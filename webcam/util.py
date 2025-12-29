from importlib import import_module
from types import ModuleType
from PIL import Image
import io
import time
import numpy as np
import torch
import cv2
import torchvision

def get_pipeline_class(pipeline_name: str) -> ModuleType:
    try:
        module = import_module(f"pipelines.{pipeline_name}")
    except ModuleNotFoundError:
        raise ValueError(f"Pipeline {pipeline_name} module not found")

    pipeline_class = getattr(module, "Pipeline", None)

    if pipeline_class is None:
        raise ValueError(f"'Pipeline' class not found in module '{pipeline_name}'.")

    return pipeline_class


def bytes_to_pil(image_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(image_bytes))#.transpose(Image.FLIP_LEFT_RIGHT)
    return image

def bytes_to_tensor(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    np_img = np.asarray(image)
    tensor = torch.from_numpy(np_img.copy())
    
    # nparr = np.frombuffer(image_bytes, np.uint8)
    # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # tensor = torch.from_numpy(img)
    return tensor


# def pil_to_frame(image: Image.Image) -> bytes:
#     frame_data = io.BytesIO()
#     image.save(frame_data, format="JPEG")
#     frame_data = frame_data.getvalue()
#     return (
#         b"--frame\r\n"
#         + b"Content-Type: image/jpeg\r\n"
#         + f"Content-Length: {len(frame_data)}\r\n\r\n".encode()
#         + frame_data
#         + b"\r\n"
#     )

def pil_to_frame(image: Image.Image) -> bytes:
    frame_data = io.BytesIO()
    image.save(frame_data, format="JPEG")
    return frame_data.getvalue()


def is_firefox(user_agent: str) -> bool:
    return "Firefox" in user_agent


def read_images_from_queue(queue, num_frames_needed, device, stop_event=None, prefer_latest=False):
    while queue.qsize() < num_frames_needed:
        if stop_event and stop_event.is_set():
            return None
        time.sleep(0.01)

    if prefer_latest:
        read_size = queue.qsize()
    else:
        read_size = min(queue.qsize(), num_frames_needed * 3)
    images = []
    for _ in range(read_size):
        images.append(queue.get())

    if prefer_latest:
        return images[-num_frames_needed:]
    else:
        return select_images(images, num_frames_needed)


def select_images(images, num_images: int):
    if len(images) < num_images:
        return images

    step = len(images) / (num_images - 1)
    indices = [int(i * step) for i in range(num_images - 1)] + [-1]

    selected_images = [images[i] for i in indices]
    return selected_images


def clear_queue(queue):
    while queue.qsize() > 0:
        queue.get()


def image_to_array(
        image: Image.Image,
        width: int,
        height: int,
        normalize: bool = True
    ) -> np.ndarray:
        image = image.convert("RGB").resize((width, height))
        image_array = np.array(image)
        if normalize:
            image_array = image_array / 127.5 - 1.0
        return image_array


def array_to_image(image_array: np.ndarray, normalize: bool = True) -> Image.Image:
    if normalize:
        image_array = image_array * 255.0
    image_array = image_array.astype(np.uint8)
    image = Image.fromarray(image_array)
    return image