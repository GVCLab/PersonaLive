import os
from pathlib import Path, PurePosixPath

from huggingface_hub import hf_hub_download

def prepare_base_model():
    print(f'Preparing sd-image-variations-diffusers weights...')
    local_dir = "./pretrained_weights/sd-image-variations-diffusers"
    os.makedirs(local_dir, exist_ok=True)
    for hub_file in ["unet/config.json", 
                     "unet/diffusion_pytorch_model.bin", 
                     "image_encoder/config.json",
                     "image_encoder/pytorch_model.bin",
                     "model_index.json"]:
        path = Path(hub_file)
        saved_path = local_dir / path
        if os.path.exists(saved_path):
            continue
        hf_hub_download(
            repo_id="lambdalabs/sd-image-variations-diffusers",
            subfolder=PurePosixPath(path.parent),
            filename=PurePosixPath(path.name),
            local_dir=local_dir,
        )


def prepare_vae():
    print(f"Preparing vae weights...")
    local_dir = "./pretrained_weights/sd-vae-ft-mse"
    os.makedirs(local_dir, exist_ok=True)
    for hub_file in [
        "config.json",
        "diffusion_pytorch_model.bin",
    ]:
        path = Path(hub_file)
        saved_path = local_dir / path
        if os.path.exists(saved_path):
            continue

        hf_hub_download(
            repo_id="stabilityai/sd-vae-ft-mse",
            subfolder=PurePosixPath(path.parent),
            filename=PurePosixPath(path.name),
            local_dir=local_dir,
        )

def prepare_personalive():
    print(f"Preparing personalive weights...")
    local_dir = "./pretrained_weights/personalive"
    os.makedirs(local_dir, exist_ok=True)
    for hub_file in [
        "pretrained_weights/personalive/denoising_unet.pth",
        "pretrained_weights/personalive/motion_encoder.pth",
        "pretrained_weights/personalive/motion_extractor.pth",
        "pretrained_weights/personalive/pose_guider.pth",
        "pretrained_weights/personalive/reference_unet.pth",
        "pretrained_weights/personalive/temporal_module.pth",
    ]:
        path = Path(hub_file)
        saved_path = local_dir / path
        if os.path.exists(saved_path):
            continue

        hf_hub_download(
            repo_id="huaichang/PersonaLive",
            subfolder=PurePosixPath(path.parent),
            filename=PurePosixPath(path.name),
            local_dir=local_dir,
        )

if __name__ == '__main__':
    prepare_base_model()
    prepare_vae()
    prepare_personalive()
    
