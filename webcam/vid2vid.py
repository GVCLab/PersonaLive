import sys
import os
sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

from multiprocessing import Queue, Manager, Event, Process
from .util import read_images_from_queue, image_to_array, array_to_image, clear_queue

import time
from typing import List
import torch
from .config import Args
from pydantic import BaseModel, Field
from PIL import Image
import math
from src.wrapper import PersonaLive
import queue

page_content = """<h1 class="text-3xl font-bold">🎭 PersonaLive!</h1>
<p class="text-sm">
    This demo showcases
    <a
    href="https://github.com/GVCLab/PersonaLive"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">PersonaLive
</a>
video-to-video pipeline with a MJPEG stream server.
</p>
"""


class Pipeline:
    class Info(BaseModel):
        name: str = "PersonaLive"
        input_mode: str = "image"
        page_content: str = page_content

    class InputParams(BaseModel):
        width: int = Field(
            512, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
        )
        height: int = Field(
            512, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
        )

    def __init__(self, args: Args, device: torch.device):
        self.args = args
        self.device = device
        self.prepare()

    def prepare(self):
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.reference_queue = Queue()

        self.prepare_event = Event()
        self.stop_event = Event()
        self.restart_event = Event()
        self.reset_event = Event()

        self.process = Process(
            target=generate_process,
            args=(self.args, self.prepare_event, self.restart_event, self.stop_event, self.reset_event, self.input_queue, self.output_queue, self.reference_queue, self.device),
            daemon=True
        )
        self.process.start()
        self.processes = [self.process]
        self.prepare_event.wait()

    def reset(self):
        self.reset_event.set()
        clear_queue(self.output_queue)

    def accept_new_params(self, params: "Pipeline.InputParams"):
        if hasattr(params, "image"):
            image_pil = params.image.to(self.device).float() / 255.0
            image_pil = image_pil * 2. - 1. 
            image_pil = image_pil.permute(2, 0, 1).unsqueeze(0)
            self.input_queue.put(image_pil)

        if hasattr(params, "restart") and params.restart:
            self.restart_event.set()
            clear_queue(self.output_queue)

    def fuse_reference(self, ref_image):
        self.reference_queue.put(ref_image)

    def produce_outputs(self) -> List[Image.Image]:
        results = []
        try:
            while True:
                data = self.output_queue.get_nowait()
                results.append(array_to_image(data))
        except queue.Empty:
            pass
            
        return results

    def close(self):
        print("Setting stop event...")
        self.stop_event.set()

        print("Waiting for processes to terminate...")
        for i, process in enumerate(self.processes):
            process.join(timeout=1.0)
            if process.is_alive():
                print(f"Process {i} didn't terminate gracefully, forcing termination")
                process.terminate()
                process.join(timeout=0.5)
                if process.is_alive():
                    print(f"Force killing process {i}")
                    process.kill()
        print("Pipeline closed successfully")

def generate_process(
        args,
        prepare_event, 
        restart_event, 
        stop_event, 
        reset_event,
        input_queue, 
        output_queue, 
        reference_queue,
        device): 
    torch.set_grad_enabled(False)
    pipeline = PersonaLive(args, device)
    chunk_size = 4
    
    prepare_event.set()

    reference_img = reference_queue.get()
    pipeline.fuse_reference(reference_img)
    print('fuse reference done')
    
    while not stop_event.is_set():
        if restart_event.is_set():
            clear_queue(input_queue)
            restart_event.clear()
        print("input_queue size = ", input_queue.qsize())
        images = read_images_from_queue(input_queue, chunk_size, device, reset_event)
        if reset_event.is_set():
            pipeline.reset()
            clear_queue(input_queue)
            clear_queue(reference_queue)
            print('Waiting for reference image...')
            reference_img = reference_queue.get()
            pipeline.fuse_reference(reference_img)
            print('Fuse reference image done')
            reset_event.clear()
            continue

        images = torch.cat(images, dim=0)
        
        video = pipeline.process_input(images)
        for image in video:
            output_queue.put(image)