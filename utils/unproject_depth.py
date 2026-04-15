import numpy as np
import trimesh
import os
import cv2
import shutil

import numpy as np
import trimesh

def unproject_rgbd_trimesh(
    depth: np.ndarray,     # (H, W)
    rgb: np.ndarray,       # (H, W, 3)
    K: np.ndarray,         # (3, 3)
    c2w: np.ndarray,       # (4, 4)
    mask: np.ndarray = None,      # (H, W)
    depth_scale: float = 1.0,
    depth_min: float = 1e-3,
    depth_max: float = 1e6,
    stride: int = 1,
):
    """
    Returns:
      cloud: trimesh.points.PointCloud (WORLD coords)
      vertices_world: (N, 3) float64
      colors_rgba: (N, 4) uint8
    """
    assert depth.ndim == 2
    assert rgb.shape[:2] == depth.shape and rgb.shape[2] == 3
    assert K.shape == (3, 3)
    assert c2w.shape == (4, 4)

    H, W = depth.shape

    # ---------- w2c -> c2w ----------
    # R_w2c = w2c[:3, :3]
    # t_w2c = w2c[:3, 3]

    # R_c2w = R_w2c.T
    # t_c2w = -R_c2w @ t_w2c

    # c2w = np.eye(4, dtype=np.float64)
    # c2w[:3, :3] = R_c2w
    # c2w[:3, 3] = t_c2w

    # ---------- depth filtering ----------
    z = depth.astype(np.float64) * depth_scale
    valid = np.isfinite(z) & (z > depth_min) & (z < depth_max)
    if mask is not None:
        valid &= (mask.astype(np.int32) == 0)

    # ---------- subsample ----------
    if stride > 1:
        z = z[::stride, ::stride]
        valid = valid[::stride, ::stride]
        rgb_s = rgb[::stride, ::stride]

        v, u = np.indices(z.shape)
        u = (u * stride).astype(np.float64)
        v = (v * stride).astype(np.float64)
    else:
        rgb_s = rgb
        v, u = np.indices((H, W))
        u = u.astype(np.float64)
        v = v.astype(np.float64)

    u = u[valid]
    v = v[valid]
    z = z[valid]

    # ---------- intrinsics ----------
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # ---------- unproject (camera coords) ----------
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    verts_cam = np.stack([x, y, z], axis=1)  # (N,3)

    # ---------- camera -> world ----------
    verts_h = np.concatenate(
        [verts_cam, np.ones((verts_cam.shape[0], 1))], axis=1
    )  # (N,4)
    verts_world = (c2w @ verts_h.T).T[:, :3]

    # ---------- colors ----------
    col = rgb_s[valid]
    if col.dtype != np.uint8:
        col = np.clip(col, 0.0, 1.0)
        col = (col * 255).astype(np.uint8)

    alpha = np.full((col.shape[0], 1), 255, dtype=np.uint8)
    colors_rgba = np.concatenate([col, alpha], axis=1)

    return verts_world, colors_rgba

def dilate_mask(binary_mask: np.ndarray, radius: int = 3, iters: int = 1, shape: str = "ellipse"):
    """
    binary_mask: (H,W) 0/1 int or bool
    radius: 커널 반지름(픽셀). 커질수록 더 많이 확장됨.
    iters: dilation 반복 횟수 (커널 고정, 반복 확장)
    shape: "ellipse" | "rect" | "cross"
    """
    binary = (binary_mask > 0).astype(np.uint8)

    k = 2 * radius + 1
    if shape == "ellipse":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    elif shape == "cross":
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (k, k))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))

    dilated = cv2.dilate(binary, kernel, iterations=iters)
    return dilated.astype(np.int32)  # 0/1

def get_single_data(data_dir):
    rgb_dir = os.path.join(data_dir, "rgb")
    depth_dir = os.path.join(data_dir, "depths")
    mask_dir = os.path.join(data_dir, "mask")
    camera_path = os.path.join(data_dir, "camera.npz")

    images = []
    for filename in sorted(os.listdir(rgb_dir)):
        image_path = os.path.join(rgb_dir, filename)
        if os.path.basename(image_path).split('.')[-1] not in ['jpg', 'jpeg', 'png']:
            continue
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    
    depths = []
    eps = 1e-6
    for filename in sorted(os.listdir(depth_dir)):
        depth_path = os.path.join(depth_dir, filename)
        if os.path.basename(depth_path).split('.')[-1] not in ['jpg', 'jpeg', 'png']:
            continue
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth /= depth.shape[1]

        # valid = np.isfinite(depth) & (depth > 0)

        # if not np.any(valid):
        #     depths.append(np.zeros_like(depth, dtype=np.float32))
        #     continue

        # d_min = depth[valid].min()
        # d_max = depth[valid].max()

        # depth_norm = np.zeros_like(depth, dtype=np.float32)
        # depth_norm[valid] = (depth[valid] - d_min) / (d_max - d_min + eps)

        depths.append(depth)

    masks = []
    for filename in sorted(os.listdir(mask_dir)):
        mask_path = os.path.join(mask_dir, filename)
        if os.path.basename(mask_path).split('.')[-1] not in ['jpg', 'jpeg', 'png']:
            continue
        mask = cv2.imread(mask_path)
        binary_mask = (mask.sum(axis=2) > 0).astype(np.int32)

        # Mask 확장
        binary_mask = dilate_mask(binary_mask, radius=6, iters=1, shape="ellipse")

        masks.append(binary_mask)
    
    cameras = np.load(camera_path)

    return images, depths, masks, cameras

# -------- Example --------
if __name__ == "__main__":
    # depth = ... (H,W)
    # rgb   = ... (H,W,3)
    # K     = ... (3,3)
    # T     = ... (4,4)  # world_from_cam
    # src_dir = "/data1/cympyc1785/data/motion_dataset/DynamicVerse/data/DynamicVerseData/DAVIS/bike-packing"
    # dst_dir = "/data1/cympyc1785/SceneData/DynamicVerse/DAVIS/bike-packing/scene_recon"

    # youtube_vis/0ae1ff65a5

    src_dir = "/data1/cympyc1785/data/motion_dataset/DynamicVerse/data/DynamicVerseData/DAVIS/blackswan"
    dst_dir = "/data1/cympyc1785/SceneData/DynamicVerse/scenes/DAVIS/blackswan/scene_recon/unproject_from_rgbd"

    os.makedirs(dst_dir, exist_ok=True)

    images, depths, masks, cameras = get_single_data(src_dir)

    print(depths[0].shape)
    print(cameras['poses'][0])
    print(cameras['intrinsics'][0])

    all_vertices = []
    all_colors = []
    for i in range(len(images)):
        verts_i, cols_i = unproject_rgbd_trimesh(
            depth=depths[i],      # (H, W)
            rgb=images[i],          # (H, W, 3)
            mask=masks[i],
            K=cameras['intrinsics'][i],              # (3, 3)
            c2w=cameras['poses'][i],          # (4, 4)
            depth_scale=1.0,
            stride=2,
        )
        all_vertices.append(verts_i)
        all_colors.append(cols_i)

    # ---- merge ----
    vertices = np.concatenate(all_vertices, axis=0)   # (N,3)
    colors   = np.concatenate(all_colors, axis=0)     # (N,4)

    cloud = trimesh.points.PointCloud(vertices=vertices, colors=colors)

    scene_save_path = os.path.join(dst_dir, "scene.ply")
    cloud.export(scene_save_path)
    print("Scene saved at:", scene_save_path)
    
    camera_save_path = os.path.join(dst_dir, "camera_params.npz")
    shutil.copyfile(os.path.join(src_dir, "camera.npz"), camera_save_path)
    print("Camera saved at:", camera_save_path)
