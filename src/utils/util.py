import importlib
import os
import os.path as osp
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import av
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
import cv2


def save_checkpoint(model, save_dir, prefix, ckpt_num, logger, total_limit=None):
    save_path = osp.join(save_dir, f"{prefix}-{ckpt_num}.pth")

    if total_limit is not None:
        checkpoints = os.listdir(save_dir)
        checkpoints = [d for d in checkpoints if d.startswith(prefix)]
        checkpoints = sorted(
            checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
        )

        if len(checkpoints) >= total_limit:
            num_to_remove = len(checkpoints) - total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(save_dir, removing_checkpoint)
                os.remove(removing_checkpoint)

    state_dict = model.state_dict()
    torch.save(state_dict, save_path)


def create_code_snapshot(root, dst_path, extensions=(".py", ".h", ".cpp", ".cu", ".cc", ".cuh", ".json", ".sh", ".bat", ".yaml"), exclude=()):
    """Creates tarball with the source code"""
    import tarfile
    from pathlib import Path
    with tarfile.open(str(dst_path), "w:gz") as tar:
        for path in Path(root).rglob("*"):
            if '.git' in path.parts:
                continue
            exclude_flag = False
            if len(exclude) > 0:
                for k in exclude:
                    if k in path.parts:
                        exclude_flag = True
            if exclude_flag:
                continue
            if path.suffix.lower() in extensions:
                try:
                    tar.add(path.as_posix(), arcname=path.relative_to(
                        root).as_posix(), recursive=True)
                except:
                    print(path)
                    assert False, 'Error occur in create_code_snapshot'

def seed_everything(seed):
    import random

    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)


def import_filename(filename):
    spec = importlib.util.spec_from_file_location("mymodule", filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def delete_additional_ckpt(base_path, num_keep):
    dirs = []
    for d in os.listdir(base_path):
        if d.startswith("checkpoint-"):
            dirs.append(d)
    num_tot = len(dirs)
    if num_tot <= num_keep:
        return
    # ensure ckpt is sorted and delete the ealier!
    del_dirs = sorted(dirs, key=lambda x: int(x.split("-")[-1]))[: num_tot - num_keep]
    for d in del_dirs:
        path_to_dir = osp.join(base_path, d)
        if osp.exists(path_to_dir):
            shutil.rmtree(path_to_dir)


def has_audio_stream(video_path):
    """Check if a video file has an audio stream."""
    try:
        container = av.open(video_path)
        for stream in container.streams:
            if stream.type == "audio":
                container.close()
                return True
        container.close()
        return False
    except Exception:
        return False


def add_audio_to_video(video_path, audio_source_path, output_path=None, verbose=False):
    """
    Add audio from audio_source_path to video_path.

    The audio will be trimmed to match the video duration if it's longer.
    If the video is longer than the audio, the audio will end when it ends.

    Args:
        video_path: Path to the video file (without audio or with audio to replace)
        audio_source_path: Path to the source file to extract audio from
        output_path: Path for the output file. If None, replaces the original video.
        verbose: If True, print debug information

    Returns:
        True if audio was successfully added, False otherwise
    """
    if not has_audio_stream(audio_source_path):
        if verbose:
            print(f"No audio stream found in {audio_source_path}")
        return False

    if output_path is None:
        output_path = video_path

    # Create a temporary file for the output
    temp_output = None
    try:
        # Get video duration
        video_container = av.open(video_path)
        video_stream = next(s for s in video_container.streams if s.type == "video")
        video_duration = float(video_stream.duration * video_stream.time_base)
        video_container.close()

        if verbose:
            print(f"Video duration: {video_duration:.2f}s")

        # Create temp file in the same directory as output to ensure same filesystem
        output_dir = os.path.dirname(output_path) or "."
        temp_fd, temp_output = tempfile.mkstemp(suffix=".mp4", dir=output_dir)
        os.close(temp_fd)

        # Use ffmpeg to combine video and audio with proper duration handling
        # -t limits the output duration to the video duration
        # -shortest would stop when the shortest stream ends, but we use -t for more control
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_source_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-t", str(video_duration),
            "-shortest",
            temp_output
        ]

        if verbose:
            print(f"Running: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            if verbose:
                print(f"ffmpeg error: {result.stderr}")
            return False

        # Replace the original file with the new one
        shutil.move(temp_output, output_path)
        temp_output = None  # Mark as moved

        if verbose:
            print(f"Successfully added audio to {output_path}")
        return True

    except Exception as e:
        if verbose:
            print(f"Error adding audio: {e}")
        return False
    finally:
        # Clean up temp file if it wasn't moved
        if temp_output and os.path.exists(temp_output):
            os.remove(temp_output)


def save_videos_from_pil(pil_images, path, fps=8, crf=None, audio_source=None):
    """
    Save a list of PIL images as a video file.

    Args:
        pil_images: List of PIL Image objects
        path: Output path for the video
        fps: Frames per second
        crf: Constant Rate Factor for video quality (lower = better quality)
        audio_source: Optional path to a video file to extract audio from.
                     The audio will be trimmed to match the output video duration.
    """
    import av

    save_fmt = Path(path).suffix
    os.makedirs(os.path.dirname(path), exist_ok=True)
    width, height = pil_images[0].size

    if save_fmt == ".mp4":
        if True:
            codec = "libx264"
            container = av.open(path, "w")
            stream = container.add_stream(codec, rate=fps)

            stream.width = width
            stream.height = height
            if crf is not None:
                stream.options = {'crf': str(crf)}

            for pil_image in pil_images:
                # pil_image = Image.fromarray(image_arr).convert("RGB")
                av_frame = av.VideoFrame.from_image(pil_image)
                container.mux(stream.encode(av_frame))
            container.mux(stream.encode())
            container.close()
        else:

            video_writer = cv2.VideoWriter(
                path.replace('.mp4', '_cv.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)
            )
            for pil_image in pil_images:
                img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                video_writer.write(img_np)
            video_writer.release()

    elif save_fmt == ".gif":
        pil_images[0].save(
            fp=path,
            format="GIF",
            append_images=pil_images[1:],
            save_all=True,
            duration=(1 / fps * 1000),
            loop=0,
        )
    else:
        raise ValueError("Unsupported file type. Use .mp4 or .gif.")

    # Add audio from source video if provided (only for mp4)
    if audio_source is not None and save_fmt == ".mp4":
        add_audio_to_video(path, audio_source, verbose=False)


def save_videos_grid(videos_, path: str, rescale=False, n_rows=6, fps=8, crf=None, audio_source=None):
    if not isinstance(videos_, list): videos_ = [videos_]

    outputs = []
    vid_len = videos_[0].shape[2]
    for i in range(vid_len):
        output = []
        for videos in videos_:
            videos = rearrange(videos, "b c t h w -> t b c h w")
            height, width = videos.shape[-2:]

            x = torchvision.utils.make_grid(videos[i], nrow=n_rows)  # (c h w)
            x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)  # (h w c)
            if rescale:
                x = (x + 1.0) / 2.0  # -1,1 -> 0,1
            x = (x * 255).numpy().astype(np.uint8)
            output.append(x)

        output = Image.fromarray(np.concatenate(output, axis=0))
        outputs.append(output)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_videos_from_pil(outputs, path, fps, crf, audio_source=audio_source)


def save_videos_grid_ori(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    height, width = videos.shape[-2:]
    outputs = []

    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)  # (c h w)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)  # (h w c)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        x = Image.fromarray(x)

        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    save_videos_from_pil(outputs, path, fps)


def read_frames(video_path):
    container = av.open(video_path)

    video_stream = next(s for s in container.streams if s.type == "video")
    frames = []
    for packet in container.demux(video_stream):
        for frame in packet.decode():
            image = Image.frombytes(
                "RGB",
                (frame.width, frame.height),
                frame.to_rgb().to_ndarray(),
            )
            frames.append(image)

    return frames


def get_fps(video_path):
    container = av.open(video_path)
    video_stream = next(s for s in container.streams if s.type == "video")
    fps = video_stream.average_rate
    container.close()
    return fps

def draw_keypoints(keypoints, height=512, width=512, device="cuda"):
    colors = torch.tensor([
        [255, 0, 0],
        [255, 255, 0],
        [0, 255, 0],
        [0, 255, 255],
        [0, 0, 255],
        [255, 0, 255],
        [255, 0, 85],
    ], device=device, dtype=torch.float32)

    selected = torch.tensor([1, 2, 3, 4, 12, 15, 20], device=device)
    B = keypoints.shape[0]

    # [B, len(selected), 2]
    pts = keypoints[:, selected] * 0.5 + 0.5
    pts[..., 0] *= width
    pts[..., 1] *= height
    pts = pts.long()

    canvas = torch.zeros((B, 3, height, width), device=device)
    radius = 4

    for i, color in enumerate(colors):
        x = pts[:, i, 0]
        y = pts[:, i, 1]
        mask = (
            (x[:, None, None] - torch.arange(width, device=device)) ** 2
            + (y[:, None, None] - torch.arange(height, device=device)[:, None]) ** 2
        ) <= radius**2
        canvas[:, 0] += color[0] / 255.0 * mask
        canvas[:, 1] += color[1] / 255.0 * mask
        canvas[:, 2] += color[2] / 255.0 * mask

    return canvas.clamp(0, 1)

def get_boxes(keypoints, height=512, width=512):
    selected = torch.tensor([1, 2, 3, 4, 12, 15, 20])

    # [B, len(selected), 2]
    pts = keypoints[:, selected] * 0.5 + 0.5
    pts[..., 0] *= width
    pts[..., 1] *= height
    pts = pts.long()

    cx = pts[..., 0].float().mean(dim=1)   # [B]
    cy = pts[..., 1].float().mean(dim=1)   # [B]

    min_y = pts[..., 1].float().min(dim=1)[0]  # [B]

    side = (cy - min_y) * 2.0
    side = side * 1.7

    x1 = (cx - side / 2 * 0.95).clamp(0, width - 1).long()
    y1 = (cy - side / 2 * 0.95).clamp(0, height - 1).long()
    x2 = (cx + side / 2 * 1.05).clamp(0, width - 1).long()
    y2 = (cy + side / 2 * 1.05).clamp(0, height - 1).long()
    
    boxes = torch.stack([x1, y1, x2, y2], dim=1)   # [B, 4]

    return boxes

def crop_face(image_pil, face_mesh):
    image = np.array(image_pil)
    h, w = image.shape[:2]
    results = face_mesh.process(image)
    face_landmarks = results.multi_face_landmarks[0]
    coords = [(int(l.x * w), int(l.y * h)) for l in face_landmarks.landmark]
    xs, ys = zip(*coords)
    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)
    face_box = (x1, y1, x2, y2)

    left, top, right, bot = scale_bb(face_box, scale=1.1, size=image.shape[:2])

    face_patch = image[int(top) : int(bot), int(left) : int(right)]

    return face_patch

def scale_bb(bbox, scale, size):
    left, top, right, bot = bbox
    width = right - left
    height = bot - top
    length = max(width, height) * scale
    center_X = (left + right) * 0.5
    center_Y = (top + bot) * 0.5
    left, top, right, bot = [
        center_X - length / 2,
        center_Y - length / 2,
        center_X + length / 2,
        center_Y + length / 2,
    ]
    left = max(0, left)
    top = max(0, top)
    right = min(size[1] - 1, right)
    bot = min(size[0] - 1, bot)
    return np.array([left, top, right, bot])