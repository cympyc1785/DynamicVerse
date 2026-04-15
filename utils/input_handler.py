import glob
import os, typer
import cv2
import numpy as np
from typing import List
from tqdm import tqdm

class InputHandler:
    """Base input handler class"""

    @staticmethod
    def validate_path(path: str, path_type: str = "file") -> str:
        """Validate path"""
        if not os.path.exists(path):
            raise typer.BadParameter(f"{path_type} not found: {path}")
        return path

    @staticmethod
    def handle_export_dir(export_dir: str, auto_cleanup: bool = False) -> str:
        """Handle export directory"""
        if os.path.exists(export_dir):
            if auto_cleanup:
                typer.echo(f"Auto-cleaning existing export directory: {export_dir}")
                import shutil

                shutil.rmtree(export_dir)
                os.makedirs(export_dir, exist_ok=True)
            else:
                typer.echo(f"Export directory '{export_dir}' already exists.")
                if typer.confirm("Do you want to clean it and continue?"):
                    import shutil

                    shutil.rmtree(export_dir)
                    os.makedirs(export_dir, exist_ok=True)
                    typer.echo(f"Cleaned export directory: {export_dir}")
                else:
                    typer.echo("Operation cancelled.")
                    raise typer.Exit(0)
        else:
            os.makedirs(export_dir, exist_ok=True)
        return export_dir

class ImageHandler(InputHandler):
    """Single image handler"""

    @staticmethod
    def process(image_path: str) -> List[str]:
        """Process single image"""
        InputHandler.validate_path(image_path, "Image file")
        return [image_path]

class ImagesHandler(InputHandler):
    """Image directory handler"""

    @staticmethod
    def process(images_dir: str, image_extensions: str = "png,jpg,jpeg") -> List[str]:
        """Process image directory"""
        InputHandler.validate_path(images_dir, "Images directory")

        # Parse extensions
        extensions = [ext.strip().lower() for ext in image_extensions.split(",")]
        extensions = [ext if ext.startswith(".") else f".{ext}" for ext in extensions]

        # Find image files
        image_files = []
        for ext in extensions:
            pattern = f"*{ext}"
            image_files.extend(glob.glob(os.path.join(images_dir, pattern)))
            image_files.extend(glob.glob(os.path.join(images_dir, pattern.upper())))

        image_files = sorted(list(set(image_files)))  # Remove duplicates and sort

        if not image_files:
            raise typer.BadParameter(
                f"No image files found in {images_dir} with extensions: {extensions}"
            )

        typer.echo(f"Found {len(image_files)} images to process")
        return image_files

class VideoHandler(InputHandler):
    """Video handler"""

    @staticmethod
    def process(video_path: str, output_dir: str, fps: float = 1.0, format="png", auto_cleanup=False, is_depth=False) -> List[str]:
        """Process video, extract frames"""
        InputHandler.validate_path(video_path, "Video file")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise typer.BadParameter(f"Cannot open video: {video_path}")

        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps

        # Calculate frame interval (ensure at least 1)
        frame_interval = max(1, int(video_fps / fps))
        actual_fps = video_fps / frame_interval

        typer.echo(f"Video FPS: {video_fps:.2f}, Duration: {duration:.2f}s")

        # Warn if requested FPS is higher than video FPS
        if fps > video_fps:
            typer.echo(
                f"⚠️  Warning: Requested sampling FPS ({fps:.2f}) exceeds video FPS ({video_fps:.2f})",  # noqa: E501
                err=True,
            )
            typer.echo(
                f"⚠️  Using maximum available FPS: {actual_fps:.2f} (extracting every frame)",
                err=True,
            )

        typer.echo(f"Extracting frames at {actual_fps:.2f} FPS (every {frame_interval} frame(s))")

        # Create output directory
        # frames_dir = os.path.join(output_dir, "images")
        # os.makedirs(frames_dir, exist_ok=True)
        frames_dir = output_dir
        frames_dir = InputHandler.handle_export_dir(frames_dir, auto_cleanup=auto_cleanup)

        frame_count = 0
        saved_count = 0

        with tqdm(total=total_frames, desc="Extracting frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if is_depth:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if frame_count % frame_interval == 0:
                    frame_path = os.path.join(frames_dir, f"frame_{saved_count:06d}.{format}")
                    cv2.imwrite(frame_path, frame)
                    saved_count += 1

                frame_count += 1
                pbar.update(1)

        cap.release()
        typer.echo(f"Extracted {saved_count} frames to {frames_dir}")

        # Get frame file list
        frame_files = sorted(
            [f for f in os.listdir(frames_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
        )
        if not frame_files:
            raise typer.BadParameter("No frames extracted from video")

        return [os.path.join(frames_dir, f) for f in frame_files]