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


def interpolate_tensors(a, b, num, device=None, dtype=None):
    """
    Linear interpolation between tensors a and b.
    input shape: (B, 1, D1, D2, ...)
    output shape: (B, num, D1, D2, ...)
    """
    if device is None:
        device = a.device
    if dtype is None:
        dtype = a.dtype
    alphas = torch.linspace(0, 1, num, device=device, dtype=dtype)
    view_shape = (1, num) + (1,) * (len(a.shape) - 2)
    alphas = alphas.view(view_shape)
    return (1 - alphas) * a + alphas * b


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
                        help="Number of frames per sliding window iteration. Set to 0 to process all frames at once (default). "
                             "Use smaller values (e.g., 4, 8, 16) for GPUs with limited VRAM (12-16GB). "
                             "Uses on-the-fly sliding window generation for smooth video without jerky transitions. "
                             "Must be divisible by 4 (temporal_window_size). Recommended: 4 for minimal VRAM.")
    parser.add_argument("--reference_image", type=str, default='',
                        help="Path to reference image. If provided, overrides test_cases from config file.")
    parser.add_argument("--driving_video", type=str, default='',
                        help="Path to driving video. If provided, overrides test_cases from config file.")
    args = parser.parse_args()

    return args


def crop_face_tensor(image_tensor, boxes, target_size=(224, 224)):
    """Crop face from tensor and resize to target size."""
    left, top, right, bot = boxes
    left, top, right, bottom = map(int, (left, top, right, bot))

    face_patch = image_tensor[:, top:bottom, left:right]
    face_patch = F.interpolate(
        face_patch.unsqueeze(0),
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )
    return face_patch


def get_boxes_from_kps(kps):
    """Get bounding boxes from keypoints."""
    from src.utils.util import get_boxes
    return get_boxes(kps)


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
                # Streaming sliding window generation for reduced VRAM usage
                # This approach processes frames on-the-fly, computing features just-in-time
                # instead of precomputing everything upfront (which causes OOM)

                # Ensure batch_size is divisible by temporal_window_size
                frames_per_window = (args.batch_size // temporal_window_size) * temporal_window_size
                if frames_per_window < temporal_window_size:
                    frames_per_window = temporal_window_size

                padding_num = (temporal_adaptive_step - 1) * temporal_window_size  # 12 frames
                num_windows = total_frames // temporal_window_size

                print("-----------")
                print(f"Streaming sliding window mode: {frames_per_window} frames per iteration")
                print(f"Total frames: {total_frames}, Windows: {num_windows}, Padding: {padding_num}")
                print("-----------")

                # Initialize image processors
                vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
                ref_image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True)
                cond_image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True, do_normalize=True)
                clip_image_processor = CLIPImageProcessor()

                # Prepare reference image embeddings (done once)
                clip_image = clip_image_processor.preprocess(
                    ref_image_pil.resize((224, 224)), return_tensors="pt"
                ).pixel_values
                clip_image_embeds = image_enc(
                    clip_image.to(device, dtype=weight_dtype)
                ).image_embeds
                encoder_hidden_states = clip_image_embeds.unsqueeze(1)

                # Prepare reference image latents (done once)
                ref_image_tensor = ref_image_processor.preprocess(
                    ref_image_pil, height=height, width=width
                ).to(dtype=weight_dtype, device=device)
                ref_image_latents = vae.encode(ref_image_tensor).latent_dist.mean
                ref_image_latents = ref_image_latents * 0.18215

                # Setup reference attention control (done once)
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

                # Initialize reference unet (done once)
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

                # Prepare reference conditioning tensors (done once)
                ref_cond_tensor = cond_image_processor.preprocess(
                    ref_image_pil, height=256, width=256
                ).to(device=device, dtype=weight_dtype)
                ref_cond_tensor = ref_cond_tensor / 2 + 0.5

                ref_face_cond_tensor = cond_image_processor.preprocess(
                    ref_face_pil, height=224, width=224
                ).to(device=device, dtype=weight_dtype)
                ref_motion = motion_encoder(ref_face_cond_tensor.unsqueeze(2))

                # Process first frame to get cached keypoints (like wrapper.py)
                first_pose_img = ori_pose_images[0]
                first_tgt_cond = cond_image_processor.preprocess(
                    first_pose_img, height=256, width=256
                ).to(device=device, dtype=weight_dtype)
                first_tgt_cond = first_tgt_cond / 2 + 0.5

                # Get interpolated keypoints for padding + first window + cached refs
                # We need padding_num + temporal_window_size frames to cover both the
                # padding frames and the first window's frames
                mot_bbox_param_interp, kps_ref, kps_frame1, _ = pose_encoder.interpolate_kps_online(
                    ref_cond_tensor, first_tgt_cond, num_interp=padding_num + temporal_window_size
                )

                # Initialize latents pile with padding (12 frames)
                latents_pile = deque(maxlen=temporal_adaptive_step)
                init_latents = ref_image_latents.unsqueeze(2).repeat(1, 1, padding_num, 1, 1)
                noise = torch.randn_like(init_latents)
                init_timesteps = reversed(timesteps).repeat_interleave(temporal_window_size, dim=0)
                noisy_latents_first = scheduler.add_noise(init_latents, noise, init_timesteps[:padding_num])
                for i in range(temporal_adaptive_step - 1):
                    l = i * temporal_window_size
                    r = (i + 1) * temporal_window_size
                    latents_pile.append(noisy_latents_first[:, :, l:r])
                del init_latents, noise, noisy_latents_first
                clear_gpu_memory()

                # Initialize pose and motion piles with padding
                # Generate padding keypoints visualization
                padding_keypoints = draw_keypoints(mot_bbox_param_interp[:padding_num], device=device).unsqueeze(2)
                padding_keypoints = rearrange(padding_keypoints, 'f c b h w -> b c f h w')
                padding_keypoints = padding_keypoints.to(device=device, dtype=weight_dtype)

                pose_pile = deque(maxlen=temporal_adaptive_step)
                for i in range(temporal_adaptive_step - 1):
                    l = i * temporal_window_size
                    r = (i + 1) * temporal_window_size
                    pose_fea = pose_guider(padding_keypoints[:, :, l:r])
                    pose_pile.append(pose_fea)
                del padding_keypoints
                clear_gpu_memory()

                # Initialize motion pile with interpolated motion
                motion_pile = deque(maxlen=temporal_adaptive_step)

                # First frame's motion embedding
                first_face_cond = cond_image_processor.preprocess(
                    dri_faces[0], height=224, width=224
                ).to(device=device, dtype=weight_dtype)
                first_motion = motion_encoder(first_face_cond.unsqueeze(2))

                # Interpolate motion for padding
                init_motion_hidden = interpolate_tensors(
                    ref_motion, first_motion, num=padding_num + 1
                )[:, :-1]

                for i in range(temporal_adaptive_step - 1):
                    l = i * temporal_window_size
                    r = (i + 1) * temporal_window_size
                    motion_pile.append(init_motion_hidden[:, l:r])
                del init_motion_hidden
                clear_gpu_memory()

                # Motion bank for history keyframe mechanism
                motion_bank = ref_motion
                num_khf = 0

                # Process video using streaming sliding window
                all_decoded_frames = []
                window_idx = 0
                frame_idx = 0

                print(f"Processing {num_windows} windows with streaming sliding window...")

                for window_idx in tqdm(range(num_windows), desc='Generating video'):
                    # Get frames for this window (4 frames)
                    window_start = window_idx * temporal_window_size
                    window_end = window_start + temporal_window_size

                    # Process this window's frames just-in-time
                    window_pose_images = ori_pose_images[window_start:window_end]
                    window_faces = dri_faces[window_start:window_end]

                    # Compute keypoints for this window
                    if window_idx == 0:
                        # First window uses interpolated keypoints from padding computation
                        window_mot_params = mot_bbox_param_interp[padding_num:padding_num + temporal_window_size]
                    else:
                        # Subsequent windows use efficient get_kps method
                        window_tgt_conds = []
                        for img in window_pose_images:
                            tgt_cond = cond_image_processor.preprocess(
                                img, height=256, width=256
                            ).to(device=device, dtype=weight_dtype)
                            tgt_cond = tgt_cond / 2 + 0.5
                            window_tgt_conds.append(tgt_cond)
                        window_tgt_tensor = torch.cat(window_tgt_conds, dim=0)
                        window_mot_params, _ = pose_encoder.get_kps(kps_ref, kps_frame1, window_tgt_tensor)
                        del window_tgt_conds, window_tgt_tensor

                    # Generate keypoints visualization for this window
                    window_keypoints = draw_keypoints(window_mot_params, device=device).unsqueeze(2)
                    window_keypoints = rearrange(window_keypoints, 'f c b h w -> b c f h w')
                    window_keypoints = window_keypoints.to(device=device, dtype=weight_dtype)
                    window_pose_fea = pose_guider(window_keypoints)
                    pose_pile.append(window_pose_fea)
                    del window_keypoints, window_mot_params

                    # Compute motion embeddings for this window
                    window_face_conds = []
                    for face in window_faces:
                        face_cond = cond_image_processor.preprocess(
                            face, height=224, width=224
                        ).to(device=device, dtype=weight_dtype)
                        window_face_conds.append(face_cond)
                    window_face_tensor = torch.cat(window_face_conds, dim=0).transpose(0, 1).unsqueeze(0)
                    window_motion = motion_encoder(window_face_tensor)
                    motion_pile.append(window_motion)
                    del window_face_conds, window_face_tensor

                    # Add new noisy latents for this window
                    new_latents = ref_image_latents.unsqueeze(2).repeat(1, 1, temporal_window_size, 1, 1)
                    noise = torch.randn_like(new_latents)
                    new_latents = scheduler.add_noise(new_latents, noise, timesteps[:1])
                    latents_pile.append(new_latents)
                    del noise

                    # Combine piles for denoising
                    latents_model_input = torch.cat(list(latents_pile), dim=2)
                    motion_hidden_state = torch.cat(list(motion_pile), dim=1)
                    pose_cond_fea = torch.cat(list(pose_pile), dim=2)

                    # Check for keyframe addition (history keyframe mechanism)
                    add_flag = False
                    if window_idx > temporal_adaptive_step * 2 and motion_bank.shape[1] < 4:
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
                    pile_list = list(latents_pile)
                    for i in range(len(pile_list)):
                        pile_list[i] = latents_model_input[:, :, i * temporal_window_size:(i + 1) * temporal_window_size]
                    latents_pile = deque(pile_list, maxlen=temporal_adaptive_step)

                    # Pop completed window from piles
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

                    del completed_latents, decoded_frames, latents_model_input
                    del motion_hidden_state, pose_cond_fea, noise_pred

                    # Clear memory periodically
                    if window_idx % 5 == 0:
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
