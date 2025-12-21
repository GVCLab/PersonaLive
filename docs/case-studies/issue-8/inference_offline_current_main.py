import argparse
import gc
import os
import sys
from collections import deque
from datetime import datetime
import mediapipe as mp
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from skimage.transform import resize
from diffusers import AutoencoderKLTemporalDecoder, AutoencoderKL, AutoencoderTiny
from diffusers.image_processor import VaeImageProcessor
from src.scheduler.scheduler_ddim import DDIMScheduler
import random
from omegaconf import OmegaConf
from einops import rearrange
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.pipelines.pipeline_pose2vid import Pose2VideoPipeline
from src.utils.util import save_videos_grid, crop_face, draw_keypoints
from decord import VideoReader
from diffusers.utils.import_utils import is_xformers_available

from src.models.motion_encoder.encoder import MotEncoder
from src.liveportrait.motion_extractor import MotionExtractor
from src.models.pose_guider import PoseGuider
from tqdm import tqdm


def clear_gpu_memory():
    """Clear GPU memory cache to reduce VRAM usage."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/prompts/personalive_offline.yaml')
    parser.add_argument("--name", type=str, default='personalive_offline')
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-L", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_xformers", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=0,
                        help="Number of frames to process per sliding window iteration. Set to 0 to process all frames at once (default). "
                             "Use smaller values (e.g., 4, 8, 16) to reduce VRAM usage on GPUs with limited memory. "
                             "Uses sliding window generation for smooth video without jerky transitions. "
                             "Must be divisible by 4 (temporal_window_size). Recommended: 4 for minimal VRAM, 8-16 for balance.")
    parser.add_argument("--reference_image", type=str, default='',
                        help="Path to reference image. If provided, overrides test_cases from config file.")
    parser.add_argument("--driving_video", type=str, default='',
                        help="Path to driving video. If provided, overrides test_cases from config file.")
    args = parser.parse_args()

    return args

def main(args):
    device = args.device
    print('device', device)
    config = OmegaConf.load(args.config)

    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    vae = AutoencoderKL.from_pretrained(config.vae_path).to(device, dtype=weight_dtype)
    # if use tiny VAE
    # vae_tiny = AutoencoderTiny.from_pretrained(config.vae_tiny_path).to(device, dtype=weight_dtype)

    infer_config = OmegaConf.load(config.inference_config)
    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path,
        subfolder="unet",
    ).to(device=device, dtype=weight_dtype)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        "",
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device=device)

    motion_encoder = MotEncoder().to(dtype=weight_dtype, device=device).eval()
    pose_guider = PoseGuider().to(device=device, dtype=weight_dtype)
    pose_encoder = MotionExtractor(num_kp=21).to(device=device, dtype=weight_dtype).eval()
    
    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        config.image_encoder_path
    ).to(dtype=weight_dtype, device=device)

    sched_kwargs = OmegaConf.to_container(
        OmegaConf.load(config.inference_config).noise_scheduler_kwargs
    )
    scheduler = DDIMScheduler(**sched_kwargs)

    generator = torch.manual_seed(args.seed)
    width, height = args.W, args.H

    # load pretrained weights
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"), strict=False
    )
    reference_unet.load_state_dict(
        torch.load(
            config.denoising_unet_path.replace('denoising_unet', 'reference_unet'),
            map_location="cpu",
        ),
        strict=True,
    )
    motion_encoder.load_state_dict(
        torch.load(
            config.denoising_unet_path.replace('denoising_unet', 'motion_encoder'),
            map_location="cpu",
        ),
        strict=True,
    )
    pose_guider.load_state_dict(
        torch.load(
            config.denoising_unet_path.replace('denoising_unet', 'pose_guider'),
            map_location="cpu",
        ),
        strict=True,
    )
    denoising_unet.load_state_dict(
        torch.load(
            config.denoising_unet_path.replace('denoising_unet', 'temporal_module'),
            map_location="cpu",
        ),
        strict=False,
    )
    pose_encoder.load_state_dict(
        torch.load(
            config.denoising_unet_path.replace('denoising_unet', 'motion_extractor'),
            map_location="cpu",
        ),
        strict=False,
    )
    
    if args.use_xformers:
        if is_xformers_available(): 
            try:
                reference_unet.enable_xformers_memory_efficient_attention()
                denoising_unet.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print("Failed to enable xformers:", e)
        else:
            print("xformers is not available. Make sure it is installed correctly.")

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

    pipe = Pose2VideoPipeline(
        vae=vae,
        # vae_tiny=vae_tiny,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        motion_encoder=motion_encoder,
        pose_encoder=pose_encoder,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )
    pipe = pipe.to(device)

    date_str = datetime.now().strftime("%Y%m%d")
    if args.name is None:
        time_str = datetime.now().strftime("%H%M")
        save_dir_name = f"{date_str}--{time_str}"
    else:
        save_dir_name = f"{date_str}--{args.name}"
    save_vid_dir = os.path.join('results', save_dir_name, 'concat_vid')
    os.makedirs(save_vid_dir, exist_ok=True)
    save_split_vid_dir = os.path.join('results', save_dir_name, 'split_vid')
    os.makedirs(save_split_vid_dir, exist_ok=True)

    pose_transform = transforms.Compose(
        [transforms.Resize((height, width)), transforms.ToTensor()]
    )

    # Override test_cases from config if CLI arguments are provided
    if args.reference_image and args.driving_video:
        args.test_cases = {args.reference_image: [args.driving_video]}
    else:
        args.test_cases = OmegaConf.load(args.config)["test_cases"]

    for ref_image_path in list(args.test_cases.keys()):
        for pose_video_path in args.test_cases[ref_image_path]:
            video_name = os.path.basename(pose_video_path).split(".")[0]
            source_name = os.path.basename(ref_image_path).split(".")[0]

            vid_name = f"{source_name}_{video_name}.mp4"
            save_vid_path = os.path.join(save_vid_dir, vid_name)
            print(save_vid_path)
            if os.path.exists(save_vid_path):
                continue

            if ref_image_path.endswith('.mp4'):
                src_vid = VideoReader(ref_image_path)
                ref_img = src_vid[0].asnumpy()
                ref_img = Image.fromarray(ref_img).convert("RGB")
            else:
                ref_img = Image.open(ref_image_path).convert("RGB")

            control = VideoReader(pose_video_path)
            video_length = min(len(control) // 4 * 4, args.L)
            sel_idx = range(len(control))[:video_length]
            control = control.get_batch([sel_idx]).asnumpy() # N, H, W, C

            ref_image_pil = ref_img.copy()
            ref_patch = crop_face(ref_image_pil, face_mesh)
            ref_face_pil = Image.fromarray(ref_patch).convert("RGB")

            size = args.H
            generator = torch.Generator(device=device)
            generator.manual_seed(42)

            dri_faces = []
            ori_pose_images = []
            for idx_control, pose_image_pil in tqdm(enumerate(control[:video_length]), total=video_length, desc='cropping faces'):
                pose_image_pil = Image.fromarray(pose_image_pil).convert("RGB")
                ori_pose_images.append(pose_image_pil)
                dri_face = crop_face(pose_image_pil, face_mesh)
                dri_face_pil = Image.fromarray(dri_face).convert("RGB")
                dri_faces.append(dri_face_pil)

            face_tensor_list = []
            ori_pose_tensor_list = []
            ref_tensor_list = []

            for idx, pose_image_pil in enumerate(ori_pose_images):
                face_tensor_list.append(pose_transform(dri_faces[idx]))
                ori_pose_tensor_list.append(pose_transform(pose_image_pil))
                ref_tensor_list.append(pose_transform(ref_image_pil))

            ref_tensor = torch.stack(ref_tensor_list, dim=0)  # (f, c, h, w)
            ref_tensor = ref_tensor.transpose(0, 1).unsqueeze(0)  # (c, f, h, w)

            face_tensor = torch.stack(face_tensor_list, dim=0)  # (f, c, h, w)
            face_tensor = face_tensor.transpose(0, 1).unsqueeze(0)

            ori_pose_tensor = torch.stack(ori_pose_tensor_list, dim=0)  # (f, c, h, w)
            ori_pose_tensor = ori_pose_tensor.transpose(0, 1).unsqueeze(0)

            # Determine batch size for processing
            temporal_window_size = 4
            temporal_adaptive_step = 4
            num_inference_steps = 4
            total_frames = len(dri_faces)

            # If batch_size is 0 or larger than total frames, process all at once using the pipeline
            if args.batch_size <= 0 or args.batch_size >= total_frames:
                print("-----------")
                print(f"Processing all {total_frames} frames at once")
                print("-----------")

                # Process all frames at once (original behavior)
                gen_video = pipe(
                    ori_pose_images,
                    ref_image_pil,
                    dri_faces,
                    ref_face_pil,
                    width,
                    height,
                    len(dri_faces),
                    num_inference_steps=num_inference_steps,
                    guidance_scale=1.0,
                    generator=generator,
                    temporal_window_size=temporal_window_size,
                    temporal_adaptive_step=temporal_adaptive_step,
                ).videos
            else:
                # Sliding window generation for reduced VRAM usage
                # This approach maintains continuous latent state across windows for smooth video

                # Ensure batch_size is divisible by temporal_window_size
                frames_per_window = (args.batch_size // temporal_window_size) * temporal_window_size
                if frames_per_window < temporal_window_size:
                    frames_per_window = temporal_window_size

                padding_num = (temporal_adaptive_step - 1) * temporal_window_size  # 12 frames
                windows_per_iteration = frames_per_window // temporal_window_size

                print("-----------")
                print(f"Sliding window mode: {frames_per_window} frames per iteration ({windows_per_iteration} windows)")
                print(f"Total frames: {total_frames}, Padding: {padding_num}")
                print("-----------")

                # Initialize image processors
                vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
                ref_image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True)
                cond_image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True, do_normalize=True)
                clip_image_processor = CLIPImageProcessor()

                # Prepare reference image embeddings
                clip_image = clip_image_processor.preprocess(
                    ref_image_pil.resize((224, 224)), return_tensors="pt"
                ).pixel_values
                clip_image_embeds = image_enc(
                    clip_image.to(device, dtype=weight_dtype)
                ).image_embeds
                encoder_hidden_states = clip_image_embeds.unsqueeze(1)

                # Prepare reference image latents
                ref_image_tensor = ref_image_processor.preprocess(
                    ref_image_pil, height=height, width=width
                ).to(dtype=weight_dtype, device=device)
                ref_image_latents = vae.encode(ref_image_tensor).latent_dist.mean
                ref_image_latents = ref_image_latents * 0.18215

                # Setup reference attention control
                reference_control_writer = ReferenceAttentionControl(
                    reference_unet,
                    do_classifier_free_guidance=False,
                    mode="write",
                    batch_size=1,
                    fusion_blocks="full",
                )
                reference_control_reader = ReferenceAttentionControl(
                    denoising_unet,
                    do_classifier_free_guidance=False,
                    mode="read",
                    batch_size=1,
                    fusion_blocks="full",
                    cache_kv=True,
                )

                # Initialize reference unet
                reference_unet(
                    ref_image_latents,
                    torch.zeros((1,), dtype=weight_dtype, device=device),
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False,
                )
                reference_control_reader.update(reference_control_writer)

                # Prepare timesteps
                timesteps = torch.tensor([999, 666, 333, 0], device=device).long()
                scheduler.set_step_length(333)
                jump = num_inference_steps // temporal_adaptive_step

                # Initialize latents pile with padding
                latents_pile = deque([])
                init_latents = ref_image_latents.unsqueeze(2).repeat(1, 1, padding_num, 1, 1)
                noise = torch.randn_like(init_latents)
                init_timesteps = reversed(timesteps).repeat_interleave(temporal_window_size, dim=0)
                noisy_latents_first = scheduler.add_noise(init_latents, noise, init_timesteps[:padding_num])
                for i in range(temporal_adaptive_step - 1):
                    l = i * temporal_window_size
                    r = (i + 1) * temporal_window_size
                    latents_pile.append(noisy_latents_first[:, :, l:r])

                # Prepare reference face for motion encoding
                ref_cond_tensor = cond_image_processor.preprocess(
                    ref_image_pil, height=256, width=256
                ).to(device=device, dtype=weight_dtype)
                ref_cond_tensor = ref_cond_tensor / 2 + 0.5

                ref_face_cond_tensor = cond_image_processor.preprocess(
                    ref_face_pil, height=224, width=224
                ).to(device=device, dtype=weight_dtype)
                ref_motion = motion_encoder(ref_face_cond_tensor.unsqueeze(2))

                # Precompute all driving face motion embeddings and keypoints
                # Use memory-efficient approach: process in batches and use get_kps after first frame
                print("Precomputing motion embeddings and keypoints...")

                # First, preprocess the first frame to get reference keypoints (like wrapper.py)
                first_pose_img = ori_pose_images[0]
                first_tgt_cond = cond_image_processor.preprocess(
                    first_pose_img, height=256, width=256
                ).to(device=device, dtype=weight_dtype)
                first_tgt_cond = first_tgt_cond / 2 + 0.5

                # Get interpolated keypoints for padding (from reference to first frame)
                # This also gives us the cached kps_ref and kps_frame1 for efficient subsequent processing
                mot_bbox_param_interp, kps_ref, kps_frame1, _ = pose_encoder.interpolate_kps_online(
                    ref_cond_tensor, first_tgt_cond, num_interp=padding_num + 1
                )

                # Process all frame keypoints in batches using the efficient get_kps method
                # This avoids calling interpolate_kps for each frame which is memory-intensive
                all_mot_bbox_params_list = [mot_bbox_param_interp[padding_num:padding_num+1]]  # First frame
                all_face_cond_tensors = []

                # Process first frame's face condition
                first_face_cond = cond_image_processor.preprocess(
                    dri_faces[0], height=224, width=224
                ).to(device=device, dtype=weight_dtype)
                all_face_cond_tensors.append(first_face_cond)

                # Process remaining frames in batches to save memory
                batch_chunk_size = 32  # Process 32 frames at a time
                for batch_start in tqdm(range(1, total_frames, batch_chunk_size), desc='Preprocessing keypoints'):
                    batch_end = min(batch_start + batch_chunk_size, total_frames)

                    # Prepare batch of target condition tensors
                    batch_tgt_conds = []
                    for i in range(batch_start, batch_end):
                        tgt_cond = cond_image_processor.preprocess(
                            ori_pose_images[i], height=256, width=256
                        ).to(device=device, dtype=weight_dtype)
                        tgt_cond = tgt_cond / 2 + 0.5
                        batch_tgt_conds.append(tgt_cond)
                    batch_tgt_tensor = torch.cat(batch_tgt_conds, dim=0)

                    # Use get_kps which is more memory efficient (uses cached kps_ref, kps_frame1)
                    batch_mot_params, _ = pose_encoder.get_kps(kps_ref, kps_frame1, batch_tgt_tensor)
                    all_mot_bbox_params_list.append(batch_mot_params)

                    # Process face conditions for this batch
                    for i in range(batch_start, batch_end):
                        face_cond = cond_image_processor.preprocess(
                            dri_faces[i], height=224, width=224
                        ).to(device=device, dtype=weight_dtype)
                        all_face_cond_tensors.append(face_cond)

                    # Clear intermediate tensors
                    del batch_tgt_conds, batch_tgt_tensor, batch_mot_params
                    clear_gpu_memory()

                all_mot_bbox_params = torch.cat(all_mot_bbox_params_list, dim=0)
                del all_mot_bbox_params_list
                clear_gpu_memory()

                # Combine interpolated padding keypoints with all frame keypoints
                full_mot_bbox_params = torch.cat([mot_bbox_param_interp[:padding_num], all_mot_bbox_params], dim=0)
                del all_mot_bbox_params
                clear_gpu_memory()

                # Generate keypoints visualization in batches to save memory
                keypoints_chunks = []
                kp_batch_size = 64
                for i in range(0, full_mot_bbox_params.shape[0], kp_batch_size):
                    kp_batch = full_mot_bbox_params[i:i+kp_batch_size]
                    kp_visual = draw_keypoints(kp_batch, device=device).unsqueeze(2)
                    keypoints_chunks.append(kp_visual.cpu())  # Move to CPU to save GPU memory
                    clear_gpu_memory()

                keypoints_full = torch.cat(keypoints_chunks, dim=0).to(device=device, dtype=weight_dtype)
                del keypoints_chunks, full_mot_bbox_params
                clear_gpu_memory()

                keypoints_full = rearrange(keypoints_full, 'f c b h w -> b c f h w')

                # Generate pose features for all frames in batches
                pose_feas_full = []
                for i in range(0, keypoints_full.shape[2], 64):
                    pose_fea = pose_guider(keypoints_full[:, :, i:i+64, :, :])
                    pose_feas_full.append(pose_fea.cpu())  # Move to CPU to save memory
                    clear_gpu_memory()
                pose_feas_full = torch.cat(pose_feas_full, dim=2).to(device=device, dtype=weight_dtype)
                del keypoints_full
                clear_gpu_memory()

                # Compute motion embeddings for all frames in smaller batches
                all_face_cond_tensor = torch.cat(all_face_cond_tensors, dim=0).transpose(0, 1).unsqueeze(0)
                del all_face_cond_tensors
                clear_gpu_memory()

                motion_hidden_states_all = []
                motion_batch_size = 64  # Smaller batch size for motion encoding
                for i in range(0, all_face_cond_tensor.shape[2], motion_batch_size):
                    motion_hidden = motion_encoder(all_face_cond_tensor[:, :, i:i+motion_batch_size, :, :])
                    motion_hidden_states_all.append(motion_hidden.cpu())  # Move to CPU
                    clear_gpu_memory()
                motion_hidden_states_all = torch.cat(motion_hidden_states_all, dim=1).to(device=device, dtype=weight_dtype)
                del all_face_cond_tensor
                clear_gpu_memory()

                # Interpolate motion for padding
                def interpolate_tensors(a, b, num):
                    alphas = torch.linspace(0, 1, num, device=a.device, dtype=a.dtype)
                    view_shape = (1, num) + (1,) * (len(a.shape) - 2)
                    alphas = alphas.view(view_shape)
                    return (1 - alphas) * a + alphas * b

                init_motion_hidden = interpolate_tensors(
                    ref_motion, motion_hidden_states_all[:, :1], num=padding_num + 1
                )[:, :-1]

                # Full motion sequence: padding + all frames + trailing padding
                motion_full = torch.cat([
                    init_motion_hidden,
                    motion_hidden_states_all,
                    motion_hidden_states_all[:, -1:].repeat(1, padding_num, 1, 1)
                ], dim=1)

                # Extend pose features with trailing padding
                pose_feas_full = torch.cat([
                    pose_feas_full,
                    pose_feas_full[:, :, -1:].repeat(1, 1, padding_num, 1, 1)
                ], dim=2)

                # Initialize piles for sliding window
                pose_pile = deque([])
                motion_pile = deque([])

                for i in range(temporal_adaptive_step):
                    l = i * temporal_window_size
                    r = (i + 1) * temporal_window_size
                    pose_pile.append(pose_feas_full[:, :, l:r])
                    motion_pile.append(motion_full[:, l:r])

                # Process frames using sliding window
                num_windows = total_frames // temporal_window_size
                all_decoded_frames = []

                motion_bank = ref_motion
                num_khf = 0

                print(f"Processing {num_windows} windows with sliding window generation...")

                for window_idx in tqdm(range(num_windows), desc='Generating video'):
                    # Add new latents for this window
                    new_latents = ref_image_latents.unsqueeze(2).repeat(1, 1, temporal_window_size, 1, 1)
                    noise = torch.randn_like(new_latents)
                    new_latents = scheduler.add_noise(new_latents, noise, timesteps[:1])
                    latents_pile.append(new_latents)

                    # Get next window's pose and motion
                    window_start = (temporal_adaptive_step + window_idx) * temporal_window_size
                    if window_start < pose_feas_full.shape[2]:
                        pose_pile.append(pose_feas_full[:, :, window_start:window_start + temporal_window_size])
                        motion_pile.append(motion_full[:, window_start:window_start + temporal_window_size])

                    # Combine piles for denoising
                    latents_model_input = torch.cat(list(latents_pile), dim=2)
                    motion_hidden_state = torch.cat(list(motion_pile), dim=1)
                    pose_cond_fea = torch.cat(list(pose_pile), dim=2)

                    # Check for keyframe addition (similar to wrapper.py)
                    add_flag = False
                    if window_idx > temporal_adaptive_step * 2 and motion_bank.shape[1] < 4:
                        # Calculate distance to motion bank
                        A_flat = motion_bank.view(motion_bank.size(1), -1)
                        B_flat = motion_hidden_state.view(motion_hidden_state.size(1), -1)
                        dist = torch.cdist(B_flat[:1].to(torch.float32), A_flat.to(torch.float32), p=2)
                        min_dist = dist.min(dim=1)[0]
                        if min_dist > 17.0:
                            add_flag = True
                            motion_bank = torch.cat([motion_bank, motion_hidden_state[:, :1]], dim=1)

                    # Denoising loop
                    for j in range(jump):
                        ut = reversed(timesteps[j::jump]).repeat_interleave(temporal_window_size, dim=0)
                        ut = torch.stack([ut]).to(device)
                        ut = rearrange(ut, 'b f -> (b f)')

                        noise_pred = denoising_unet(
                            latents_model_input,
                            ut,
                            encoder_hidden_states=[encoder_hidden_states, motion_hidden_state],
                            pose_cond_fea=pose_cond_fea,
                            return_dict=False,
                        )[0]

                        clip_length = noise_pred.shape[2]
                        mid_noise_pred = rearrange(noise_pred, 'b c f h w -> (b f) c h w')
                        mid_latents = rearrange(latents_model_input, 'b c f h w -> (b f) c h w')

                        mid_latents, pred_original_sample = scheduler.step(
                            mid_noise_pred, ut, mid_latents, generator=generator, return_dict=False
                        )

                        mid_latents = rearrange(mid_latents, '(b f) c h w -> b c f h w', f=clip_length)
                        pred_original_sample = rearrange(pred_original_sample, '(b f) c h w -> b c f h w', f=clip_length)

                        latents_model_input = torch.cat([
                            pred_original_sample[:, :, :temporal_window_size],
                            mid_latents[:, :, temporal_window_size:]
                        ], dim=2)
                        latents_model_input = latents_model_input.to(dtype=weight_dtype)

                    # History keyframe mechanism
                    if add_flag and num_khf < 3:
                        reference_control_writer.clear()
                        reference_unet(
                            pred_original_sample[:, :, 0].to(weight_dtype),
                            torch.zeros((1,), dtype=weight_dtype, device=device),
                            encoder_hidden_states=encoder_hidden_states,
                            return_dict=False,
                        )
                        reference_control_reader.update_hkf(reference_control_writer)
                        num_khf += 1

                    # Update latents pile with denoised results
                    for i in range(len(latents_pile)):
                        latents_pile[i] = latents_model_input[:, :, i * temporal_window_size:(i + 1) * temporal_window_size]

                    # Pop completed window and decode
                    pose_pile.popleft()
                    motion_pile.popleft()
                    completed_latents = latents_pile.popleft()

                    # Decode latents to frames
                    completed_latents = 1 / 0.18215 * completed_latents
                    completed_latents = rearrange(completed_latents, "b c f h w -> (b f) c h w")
                    decoded_frames = vae.decode(completed_latents).sample
                    decoded_frames = rearrange(decoded_frames, "b c h w -> b h w c")
                    decoded_frames = (decoded_frames / 2 + 0.5).clamp(0, 1)
                    all_decoded_frames.append(decoded_frames.cpu().float())

                    # Clear memory periodically
                    if window_idx % 10 == 0:
                        clear_gpu_memory()

                # Cleanup
                reference_control_reader.clear()
                reference_control_writer.clear()

                # Combine all frames
                all_decoded_frames = torch.cat(all_decoded_frames, dim=0)
                all_decoded_frames = rearrange(all_decoded_frames, "f h w c -> c f h w").unsqueeze(0)
                gen_video = all_decoded_frames.numpy()
                gen_video = torch.from_numpy(gen_video)

                clear_gpu_memory()

            #Concat it with pose tensor
            video = torch.cat([ref_tensor, face_tensor, ori_pose_tensor, gen_video], dim=0)

            save_videos_grid(
                video,
                save_vid_path,
                n_rows=4,
                fps=25
            )

            if True:
                save_vid_path = save_vid_path.replace(save_vid_dir, save_split_vid_dir)
                save_videos_grid(gen_video, save_vid_path, n_rows=1, fps=25, crf=18, audio_source=pose_video_path)

if __name__ == "__main__":
    args = parse_args()
    main(args)
