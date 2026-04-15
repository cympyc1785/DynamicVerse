import os
import json
import torch
import numpy as np
import cv2
import argparse
from PIL import Image
from scipy.optimize import minimize


def depth2disparity(depth, return_mask=False):
    if isinstance(depth, torch.Tensor):
        disparity = torch.zeros_like(depth)
    elif isinstance(depth, np.ndarray):
        disparity = np.zeros_like(depth)
    else:
        raise TypeError(f"Unsupported type: {type(depth)}")

    non_negtive_mask = depth > 0
    disparity[non_negtive_mask] = 1.0 / depth[non_negtive_mask]

    if return_mask:
        return disparity, non_negtive_mask
    else:
        return disparity


def absolute_error_loss(params, predicted_depth, ground_truth_depth):
    s, t = params
    predicted_aligned = s * predicted_depth + t
    abs_error = np.abs(predicted_aligned - ground_truth_depth)
    return np.sum(abs_error)


def absolute_value_scaling(predicted_depth, ground_truth_depth, s=1, t=0):
    predicted_depth_np = predicted_depth.cpu().numpy().reshape(-1)
    ground_truth_depth_np = ground_truth_depth.cpu().numpy().reshape(-1)

    initial_params = [s, t]
    result = minimize(
        absolute_error_loss,
        initial_params,
        args=(predicted_depth_np, ground_truth_depth_np),
    )

    s, t = result.x
    return s, t


def absolute_value_scaling2(
    predicted_depth,
    ground_truth_depth,
    s_init=1.0,
    t_init=0.0,
    lr=1e-4,
    max_iters=1000,
    tol=1e-6,
):
    s = torch.tensor(
        [s_init],
        requires_grad=True,
        device=predicted_depth.device,
        dtype=predicted_depth.dtype,
    )
    t = torch.tensor(
        [t_init],
        requires_grad=True,
        device=predicted_depth.device,
        dtype=predicted_depth.dtype,
    )

    optimizer = torch.optim.Adam([s, t], lr=lr)
    prev_loss = None

    for _ in range(max_iters):
        optimizer.zero_grad()

        predicted_aligned = s * predicted_depth + t
        abs_error = torch.abs(predicted_aligned - ground_truth_depth)
        loss = torch.sum(abs_error)

        loss.backward()
        optimizer.step()

        if prev_loss is not None and abs(prev_loss - loss.item()) < tol:
            break

        prev_loss = loss.item()

    return s.detach().item(), t.detach().item()


def load_background_masks(mask_scene_dir, target_h=None, target_w=None, background_threshold=1):
    """
    mask_scene_dir/
        frame_0001.png
        frame_0002.png
        ...

    mask pixel >= background_threshold -> background(True)
    """
    if not os.path.isdir(mask_scene_dir):
        raise FileNotFoundError(f"Mask directory not found: {mask_scene_dir}")

    mask_files = sorted(
        [
            os.path.join(mask_scene_dir, f)
            for f in os.listdir(mask_scene_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )

    if len(mask_files) == 0:
        raise FileNotFoundError(f"No mask files found in: {mask_scene_dir}")

    masks = []
    for mask_path in mask_files:
        m = np.array(Image.open(mask_path))
        if m.ndim == 3:
            m = m[..., 0]

        m = (m >= background_threshold).astype(np.uint8)

        if target_h is not None and target_w is not None and m.shape != (target_h, target_w):
            m = cv2.resize(m, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        masks.append(m > 0)

    masks = np.stack(masks, axis=0)  # (T, H, W), bool
    return masks


def depth_evaluation(
    predicted_depth_original,
    ground_truth_depth_original,
    max_depth=80,
    custom_mask=None,
    post_clip_min=None,
    post_clip_max=None,
    pre_clip_min=None,
    pre_clip_max=None,
    align_with_lstsq=False,
    align_with_lad=False,
    align_with_lad2=False,
    lr=1e-4,
    max_iters=1000,
    use_gpu=False,
    align_with_scale=False,
    disp_input=False,
):
    if isinstance(predicted_depth_original, np.ndarray):
        predicted_depth_original = torch.from_numpy(predicted_depth_original)
    if isinstance(ground_truth_depth_original, np.ndarray):
        ground_truth_depth_original = torch.from_numpy(ground_truth_depth_original)
    if custom_mask is not None and isinstance(custom_mask, np.ndarray):
        custom_mask = torch.from_numpy(custom_mask)

    if predicted_depth_original.dim() == 3:
        _, h, w = predicted_depth_original.shape
        predicted_depth_original = predicted_depth_original.view(-1, w)
        ground_truth_depth_original = ground_truth_depth_original.view(-1, w)
        if custom_mask is not None:
            custom_mask = custom_mask.view(-1, w)

    if use_gpu:
        predicted_depth_original = predicted_depth_original.cuda()
        ground_truth_depth_original = ground_truth_depth_original.cuda()
        if custom_mask is not None:
            custom_mask = custom_mask.cuda()

    if max_depth is not None:
        mask = (ground_truth_depth_original > 0) & (ground_truth_depth_original < max_depth)
    else:
        mask = (ground_truth_depth_original > 0)

    predicted_depth = predicted_depth_original[mask]
    ground_truth_depth = ground_truth_depth_original[mask]

    if pre_clip_min is not None:
        predicted_depth = torch.clamp(predicted_depth, min=pre_clip_min)
    if pre_clip_max is not None:
        predicted_depth = torch.clamp(predicted_depth, max=pre_clip_max)

    if disp_input:
        print("Aligning in disp space")
        real_gt = ground_truth_depth.clone()
        ground_truth_depth = 1 / (ground_truth_depth + 1e-8)

    if align_with_lstsq:
        print("Using lstsq!")
        predicted_depth_np = predicted_depth.cpu().numpy().reshape(-1, 1)
        ground_truth_depth_np = ground_truth_depth.cpu().numpy().reshape(-1, 1)

        A = np.hstack([predicted_depth_np, np.ones_like(predicted_depth_np)])
        result = np.linalg.lstsq(A, ground_truth_depth_np, rcond=None)
        s, t = result[0][0], result[0][1]

        s = torch.tensor(s, device=predicted_depth_original.device, dtype=predicted_depth_original.dtype)
        t = torch.tensor(t, device=predicted_depth_original.device, dtype=predicted_depth_original.dtype)

        predicted_depth = s * predicted_depth + t

    elif align_with_lad:
        s_init = (torch.median(ground_truth_depth) / torch.median(predicted_depth)).item()
        s, t = absolute_value_scaling(predicted_depth, ground_truth_depth, s=s_init)
        predicted_depth = s * predicted_depth + t

    elif align_with_lad2:
        print("using lad2!")
        s_init = (torch.median(ground_truth_depth) / torch.median(predicted_depth)).item()
        s, t = absolute_value_scaling2(
            predicted_depth,
            ground_truth_depth,
            s_init=s_init,
            lr=lr,
            max_iters=max_iters,
        )
        predicted_depth = s * predicted_depth + t

    elif align_with_scale:
        dot_pred_gt = torch.nanmean(ground_truth_depth)
        dot_pred_pred = torch.nanmean(predicted_depth)
        s = dot_pred_gt / dot_pred_pred

        for _ in range(10):
            residuals = s * predicted_depth - ground_truth_depth
            abs_residuals = residuals.abs() + 1e-8
            weights = 1.0 / abs_residuals

            weighted_dot_pred_gt = torch.sum(weights * predicted_depth * ground_truth_depth)
            weighted_dot_pred_pred = torch.sum(weights * predicted_depth ** 2)
            s = weighted_dot_pred_gt / weighted_dot_pred_pred

        s = s.clamp(min=1e-3).detach()
        predicted_depth = s * predicted_depth

    else:
        print("using scale!")
        scale_factor = torch.median(ground_truth_depth) / torch.median(predicted_depth)
        predicted_depth = predicted_depth * scale_factor

    if disp_input:
        ground_truth_depth = real_gt
        predicted_depth = depth2disparity(predicted_depth)

    if post_clip_min is not None:
        predicted_depth = torch.clamp(predicted_depth, min=post_clip_min)
    if post_clip_max is not None:
        predicted_depth = torch.clamp(predicted_depth, max=post_clip_max)

    if custom_mask is not None:
        assert custom_mask.shape == ground_truth_depth_original.shape
        mask_within_mask = custom_mask[mask]
        predicted_depth = predicted_depth[mask_within_mask]
        ground_truth_depth = ground_truth_depth[mask_within_mask]
    else:
        mask_within_mask = None

    num_valid_pixels = predicted_depth.numel()
    if num_valid_pixels == 0:
        abs_rel, sq_rel, rmse, threshold_1 = 0.0, 0.0, 0.0, 0.0
    else:
        abs_rel = torch.mean(torch.abs(predicted_depth - ground_truth_depth) / ground_truth_depth).item()
        sq_rel = torch.mean(((predicted_depth - ground_truth_depth) ** 2) / ground_truth_depth).item()
        rmse = torch.sqrt(torch.mean((predicted_depth - ground_truth_depth) ** 2)).item()

        predicted_depth_for_acc = torch.clamp(predicted_depth, min=1e-5)
        max_ratio = torch.maximum(
            predicted_depth_for_acc / ground_truth_depth,
            ground_truth_depth / predicted_depth_for_acc,
        )
        threshold_1 = torch.mean((max_ratio < 1.25).float()).item()

    if align_with_lstsq or align_with_lad or align_with_lad2:
        predicted_depth_original = predicted_depth_original * s + t
        if disp_input:
            predicted_depth_original = depth2disparity(predicted_depth_original)
        depth_error_parity_map = (
            torch.abs(predicted_depth_original - ground_truth_depth_original)
            / torch.clamp(ground_truth_depth_original, min=1e-8)
        )
    elif align_with_scale:
        predicted_depth_original = predicted_depth_original * s
        if disp_input:
            predicted_depth_original = depth2disparity(predicted_depth_original)
        depth_error_parity_map = (
            torch.abs(predicted_depth_original - ground_truth_depth_original)
            / torch.clamp(ground_truth_depth_original, min=1e-8)
        )
    else:
        predicted_depth_original = predicted_depth_original * scale_factor
        if disp_input:
            predicted_depth_original = depth2disparity(predicted_depth_original)
        depth_error_parity_map = (
            torch.abs(predicted_depth_original - ground_truth_depth_original)
            / torch.clamp(ground_truth_depth_original, min=1e-8)
        )

    depth_error_parity_map_full = torch.zeros_like(ground_truth_depth_original)
    depth_error_parity_map_full = torch.where(mask, depth_error_parity_map, depth_error_parity_map_full)

    predict_depth_map_full = predicted_depth_original

    gt_depth_map_full = torch.zeros_like(ground_truth_depth_original)
    gt_depth_map_full = torch.where(mask, ground_truth_depth_original, gt_depth_map_full)

    full_mask_np = mask.detach().cpu().numpy()

    results = np.array([abs_rel, threshold_1, num_valid_pixels], dtype=np.float64)

    return (
        results,
        depth_error_parity_map_full.detach().cpu().numpy(),
        predict_depth_map_full.detach().cpu().numpy(),
        gt_depth_map_full.detach().cpu().numpy(),
        full_mask_np,
    )


def get_preds(video_path, experiment_name, number_of_frames, w, h):
    full_result_dir = os.path.join(video_path, "dynamicBA", "base")

    disp_preds = []

    for i in range(number_of_frames):
        npz_path = os.path.join(full_result_dir, f"{i:04d}_simple_flow_opt_1.0_1e-4.npz")

        if not os.path.exists(npz_path):
            break

        npz = np.load(npz_path, allow_pickle=True)
        disp = npz["disp"]
        disp = cv2.resize(disp, (w, h), interpolation=cv2.INTER_CUBIC)
        disp = torch.tensor(disp)
        disp_preds.append(disp)

    if len(disp_preds) == 0:
        raise FileNotFoundError(f"No prediction files found in {full_result_dir}")

    if len(disp_preds) < number_of_frames:
        print(
            f"Warning: Only found {len(disp_preds)} prediction frames, "
            f"fewer than GT frames {number_of_frames}. Using {len(disp_preds)}."
        )

    disp_preds = np.array(disp_preds)
    return disp_preds


def get_da3_preds(video_path, model_name, number_of_frames, w, h):
    full_result_dir = os.path.join(video_path, model_name)

    npy_path = os.path.join(full_result_dir, "depth.npy")
    depth = np.load(npy_path, allow_pickle=True)
    disparity = depth2disparity(depth)

    disp_preds = []
    for i in range(len(disparity)):
        disp = disparity[i]
        disp = cv2.resize(disp, (w, h), interpolation=cv2.INTER_CUBIC)
        disp = torch.tensor(disp)
        disp_preds.append(disp)

    if len(disp_preds) == 0:
        raise FileNotFoundError(f"No prediction files found in {full_result_dir}")

    if len(disp_preds) < number_of_frames:
        print(
            f"Warning: Only found {len(disp_preds)} prediction frames, "
            f"fewer than GT frames {number_of_frames}. Using {len(disp_preds)}."
        )

    disp_preds = np.array(disp_preds)
    return disp_preds


def get_gt(video_path, dataset):
    depth_gt = []

    depth_paths = sorted(os.listdir(os.path.join(video_path, "depth_gt")))
    depth_paths = [os.path.join(video_path, "depth_gt", x) for x in depth_paths]

    number_of_frames = len(depth_paths)
    h, w = None, None

    for i in range(number_of_frames):
        if dataset == "sintel":
            depth = np.load(depth_paths[i])
        elif dataset == "kitti":
            img_pil = Image.open(depth_paths[i])
            depth_png = np.array(img_pil, dtype=int)
            assert np.max(depth_png) > 255
            depth = depth_png.astype(float) / 256.0
        elif dataset == "bonn":
            depth_png = np.asarray(Image.open(depth_paths[i]))
            assert np.max(depth_png) > 255
            depth = depth_png.astype(np.float64) / 5000.0
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        if h is None:
            h, w = depth.shape

        depth_gt.append(depth)

    depth_gt = np.stack(depth_gt, axis=0)
    return depth_gt, number_of_frames, h, w


def get_background_masks(mask_root, video_name, number_of_frames, h, w, background_threshold=1):
    """
    mask_root/
        video_name/
            frame_0001.png
            frame_0002.png
            ...
    """
    mask_scene_dir = os.path.join(mask_root, video_name)

    masks = load_background_masks(
        mask_scene_dir,
        target_h=h,
        target_w=w,
        background_threshold=background_threshold,
    )

    if len(masks) < number_of_frames:
        print(
            f"Warning: Only found {len(masks)} mask frames for {video_name}, "
            f"fewer than GT frames {number_of_frames}. Using {len(masks)}."
        )

    return masks


def eval_depth(
    dataset="sintel",
    base_path=None,
    mask_root=None,
    experiment_name="base",
    scale_only=False,
    max_depth=70,
    model="dynamicBA",
):
    if base_path is None:
        base_path = f"./data/{dataset}/"
    if mask_root is None:
        raise ValueError("mask_root must be provided")

    videos = sorted(os.listdir(base_path))
    excluded = set()

    res_save_path = f"results/depth/{dataset}/{experiment_name}"
    os.makedirs(res_save_path, exist_ok=True)

    results_txt_path = os.path.join(res_save_path, "all_results.txt")
    with open(results_txt_path, "w") as results_file:
        results_file.write(f"{'Video':<15}{'Abs_Rel':<12}{'d1 < 1.25':<12}\n")
        results_file.write("-" * 40 + "\n")

    errs = {
        "total_videos": 0,
        "errors": torch.zeros(2, dtype=torch.float64),
        "total_weights": 0.0,
    }

    video_weights = {}
    video_metrics = {}

    for i, video in enumerate(videos):
        if video in excluded:
            continue

        video_path = os.path.join(base_path, video)
        if not os.path.isdir(video_path):
            continue

        print(f"Working on {i}:{video}", flush=True)

        gt_depths, number_of_frames, h, w = get_gt(video_path, dataset)
        if model == "dynamicBA":
            pred_disp = get_preds(video_path, experiment_name, number_of_frames, w, h)
        else:
            pred_disp = get_da3_preds(video_path, model, number_of_frames, w, h)
        bg_masks = get_background_masks(mask_root, video, number_of_frames, h, w)

        total_len = min(number_of_frames, pred_disp.shape[0], gt_depths.shape[0], bg_masks.shape[0])
        gt_depths = gt_depths[:total_len]
        pred_disp = pred_disp[:total_len]
        bg_masks = bg_masks[:total_len]

        results, _, _, _, _ = depth_evaluation(
            pred_disp,
            gt_depths,
            align_with_lstsq=not scale_only,
            align_with_lad2=False,
            disp_input=True,
            max_depth=max_depth,
            use_gpu=torch.cuda.is_available(),
            post_clip_max=max_depth,
            custom_mask=bg_masks,
        )

        with open(results_txt_path, "a") as results_file:
            results_file.write(f"{video:<15}{results[0]:<12.6f}{results[1]:<12.6f}\n")

        video_weights[video] = float(results[2])
        video_metrics[video] = {"abs_rel": float(results[0]), "d1": float(results[1])}

        errs["errors"] += torch.tensor(results[:2], dtype=torch.float64) * float(results[-1])
        errs["total_videos"] += 1
        errs["total_weights"] += float(results[-1])

    if errs["total_weights"] == 0:
        total_errors_video = torch.zeros(2, dtype=torch.float64)
    else:
        total_errors_video = errs["errors"] / errs["total_weights"]

    print("\nPer-video weight information:")
    print(f"{'Video':<15}{'Abs_Rel':<12}{'d1 < 1.25':<12}{'ValidPixels':<15}{'Weight(%)':<12}")
    print("-" * 65)

    for video, weight in sorted(video_weights.items(), key=lambda x: x[1], reverse=True):
        if video in video_metrics:
            abs_rel = video_metrics[video]["abs_rel"]
            d1 = video_metrics[video]["d1"]
            weight_ratio = (weight / errs["total_weights"] * 100.0) if errs["total_weights"] > 0 else 0.0
            print(f"{video:<15}{abs_rel:<12.6f}{d1:<12.6f}{weight:<15.0f}{weight_ratio:<12.2f}")

    print("\nWeighted results:")
    print(f"Total valid pixels: {errs['total_weights']}")
    print(f"Weighted mean Abs_Rel: {total_errors_video[0].item():.6f}")
    print(f"Weighted mean d1 < 1.25: {total_errors_video[1].item():.6f}")

    with open(results_txt_path, "a") as results_file:
        results_file.write("\n" + "=" * 40 + "\n")
        results_file.write("SUMMARY\n")
        results_file.write("-" * 40 + "\n")
        results_file.write(f"{'Average Abs_Rel:':<20}{total_errors_video[0].item():.6f}\n")
        results_file.write(f"{'Average d1 < 1.25:':<20}{total_errors_video[1].item():.6f}\n")
        results_file.write(f"{'Evaluated videos:':<20}{errs['total_videos']}\n")

    final_result = {
        "video: abs_rel": total_errors_video[0].item(),
        "video: d1 < 1.25": total_errors_video[1].item(),
        "total_videos": errs["total_videos"],
    }

    print(json.dumps(final_result, indent=4))

    with open(os.path.join(res_save_path, "result.json"), "w") as f:
        f.write(json.dumps(final_result, indent=4))

    return final_result


def main():
    parser = argparse.ArgumentParser(description="Evaluate depth predictions")
    parser.add_argument("--dataset", type=str, default="sintel", help="Dataset name")
    parser.add_argument("--base_path", type=str, default="./data/sintel", help="Base path to dataset")
    parser.add_argument(
        "--mask_root",
        type=str,
        default="/data1/cympyc1785/SceneData/sintel/dataset/mask",
        help="Mask root directory. Structure: mask_root/video_name/frame_0001.png",
    )
    parser.add_argument("--experiment_name", type=str, default="base", help="Experiment name")
    parser.add_argument("--scale_only", action="store_true", help="Use only scale alignment (no shift)")
    parser.add_argument("--max_depth", type=float, default=70, help="Maximum depth value to consider")
    parser.add_argument("--model", type=str, default="dynamicBA", help="Model name")

    args = parser.parse_args()

    eval_depth(
        dataset=args.dataset,
        base_path=args.base_path,
        mask_root=args.mask_root,
        experiment_name=args.experiment_name,
        scale_only=args.scale_only,
        max_depth=args.max_depth,
        model=args.model,
    )


if __name__ == "__main__":
    main()