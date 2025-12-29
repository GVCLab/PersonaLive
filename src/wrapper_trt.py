from omegaconf import OmegaConf
import os
import torch
import numpy as np
from PIL import Image
import time
import gc
import cv2
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.pose_guider import PoseGuider
from src.models.motion_encoder.encoder import MotEncoder
from src.models.unet_3d import UNet3DConditionModel
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.scheduler.scheduler_ddim import DDIMScheduler
from src.liveportrait.motion_extractor import MotionExtractor
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from collections import deque
from threading import Lock, Thread
from torchvision import transforms as T
from einops import rearrange
from src.utils.util import draw_keypoints, get_boxes
import torch.nn.functional as F
from src.modeling.engine_model import EngineModel
import pycuda.driver as cuda

def map_device(device_or_str):
    return device_or_str if isinstance(device_or_str, torch.device) else torch.device(device_or_str)

class PersonaLive:
    def __init__(self, args, device=None):
        cfg = OmegaConf.load(args.config_path)
        if(device is None):
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = map_device(device)

        self.temporal_adaptive_step = cfg.temporal_adaptive_step
        self.temporal_window_size = cfg.temporal_window_size

        if cfg.dtype == "fp16":
            self.numpy_dtype = np.float16
            self.dtype = torch.float16
        elif cfg.dtype == "fp32":
            self.numpy_dtype = np.float32
            self.dtype = torch.float32

        infer_config = OmegaConf.load(cfg.inference_config)
        sched_kwargs = OmegaConf.to_container(
            infer_config.noise_scheduler_kwargs
        )

        self.num_inference_steps = cfg.num_inference_steps

        # initialize models
        self.pose_guider = PoseGuider().to(device=self.device, dtype=self.dtype)
        pose_guider_state_dict = torch.load(cfg.pose_guider_path, map_location="cpu")
        self.pose_guider.load_state_dict(pose_guider_state_dict)
        del pose_guider_state_dict

        self.motion_encoder = MotEncoder().to(dtype=self.dtype, device=self.device).eval()
        motion_encoder_state_dict = torch.load(cfg.motion_encoder_path, map_location="cpu")
        self.motion_encoder.load_state_dict(motion_encoder_state_dict)
        del motion_encoder_state_dict

        self.pose_encoder = MotionExtractor(num_kp=21).to(device=self.device, dtype=self.dtype).eval()
        pose_encoder_state_dict = torch.load(cfg.pose_encoder_path, map_location="cpu")
        self.pose_encoder.load_state_dict(pose_encoder_state_dict, strict=False)
        del pose_encoder_state_dict

        self.reference_unet = UNet2DConditionModel.from_pretrained(
            cfg.pretrained_base_model_path,
            subfolder="unet",
        ).to(dtype=self.dtype, device=self.device)
        reference_unet_state_dict = torch.load(cfg.reference_unet_weight_path, map_location="cpu")
        self.reference_unet.load_state_dict(reference_unet_state_dict)
        del reference_unet_state_dict

        self.reference_control_writer = ReferenceAttentionControl(
            self.reference_unet,
            do_classifier_free_guidance=False,
            mode="write",
            batch_size=cfg.batch_size,
            fusion_blocks="full",
        )

        self.vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(
            device=self.device, dtype=self.dtype
        )
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            cfg.image_encoder_path,
        ).to(device=self.device, dtype=self.dtype)
        
        #----------------------- TensorRT -----------------------#
        self.unet_work = EngineModel(engine_file_path=cfg.tensorrt_target_model, device_int=self.device.index)
        self.unet_work.bind({
            "motion_hidden_states_out": "motion_hidden_states",
            "pose_cond_fea_out": "pose_cond_fea",
            "latents" : "sample",
        })
        #------------------------------------------------------------#

        # miscellaneous
        self.scheduler = DDIMScheduler(**sched_kwargs)
        timesteps = torch.tensor([0, 333, 666, 999], device=self.device)
        self.timesteps = timesteps.repeat_interleave(cfg.temporal_window_size, dim=0).long()
        self.scheduler.set_step_length(333)

        self.batch_size = cfg.batch_size
        self.vae_scale_factor = 8
        self.ref_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
        )
        self.clip_image_processor = CLIPImageProcessor()
        self.cond_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=True)
        
        self.cfg = cfg
        self.reset()
        self.reference_hidden_states_names = ["d00", "d01", "d10", "d11", 
                                              "d20", "d21", "m", "u10", "u11", "u12", 
                                              "u20", "u21", "u22", "u30", "u31", "u32"]
        torch.cuda.empty_cache()

        try:
            self.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print("Failed to enable xformers:", e)

    def reset(self):
        self.first_frame = True
        self.motion_bank = None
        self.count = 0
        self.num_khf = 0
        self.reference_control_writer.clear()
        
    def enable_xformers_memory_efficient_attention(self):
        self.reference_unet.enable_xformers_memory_efficient_attention()

    def fast_resize(self, images, target_width, target_height) -> torch.Tensor:
        tgt_cond_tensor = F.interpolate(
            images,
            size=(target_width, target_height),
            mode="bilinear",
            align_corners=False,
        )
        return tgt_cond_tensor

    @torch.no_grad()
    def fuse_reference(self, ref_image): # pil input
        clip_image = self.clip_image_processor.preprocess(
            ref_image, return_tensors="pt"
        ).pixel_values
        ref_image_tensor = self.ref_image_processor.preprocess(
            ref_image, height=self.cfg.reference_image_height, width=self.cfg.reference_image_width
        )  # (bs, c, width, height)
        clip_image_embeds = self.image_encoder(
            clip_image.to(self.image_encoder.device, dtype=self.image_encoder.dtype)
        ).image_embeds
        encoder_hidden_states = clip_image_embeds.unsqueeze(1)
        torch.cuda.synchronize()
        self.unet_work.prefill(encoder_hidden_states = encoder_hidden_states)
        self.encoder_hidden_states = encoder_hidden_states

        ref_image_tensor = ref_image_tensor.to(
            dtype=self.vae.dtype, device=self.vae.device
        )
        self.ref_image_tensor = ref_image_tensor.squeeze(0)
        ref_image_latents = self.vae.encode(ref_image_tensor).latent_dist.mean
        ref_image_latents = ref_image_latents * 0.18215  # (b, 4, h, w)
        self.reference_unet(
            ref_image_latents.to(self.reference_unet.device),
            torch.zeros((self.batch_size,),dtype=self.dtype,device=self.reference_unet.device),
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )
        self.reference_hidden_states = self.reference_control_writer.output()
        torch.cuda.synchronize()
        self.unet_work.prefill(**{name: self.reference_hidden_states[name] for name in self.reference_hidden_states_names})

        ref_cond_tensor = self.cond_image_processor.preprocess(
            ref_image, height=256, width=256
        ).to(device=self.device, dtype=self.pose_encoder.dtype)  # (1, c, h, w)
        self.ref_cond_tensor = ref_cond_tensor / 2 + 0.5 # to [0, 1]
        self.ref_image_latents = ref_image_latents.unsqueeze(2).repeat(1, 1, self.temporal_window_size, 1, 1)

        padding_num = (self.temporal_adaptive_step - 1) * self.temporal_window_size
        init_latents = ref_image_latents.unsqueeze(2).repeat(1, 1, padding_num, 1, 1)
        noise = torch.randn_like(init_latents)
        self.noisy_latents_first = self.scheduler.add_noise(init_latents, noise, self.timesteps[:padding_num])
    
    def crop_face(self, image_pil, boxes):
        image = np.array(image_pil)

        left, top, right, bot = boxes

        face_patch = image[int(top) : int(bot), int(left) : int(right)]
        face_patch = Image.fromarray(face_patch).convert("RGB")
        return face_patch
    
    def crop_face_tensor(self, image_tensor, boxes):
        left, top, right, bot = boxes
        left, top, right, bottom = map(int, (left, top, right, bot))

        face_patch = image_tensor[:, top:bottom, left:right]
        face_patch = F.interpolate(
            face_patch.unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
        return face_patch
    
    def interpolate_tensors(self, a: torch.Tensor, b: torch.Tensor, num: int = 10) -> torch.Tensor:
        """
        在张量 a 和 b 之间线性插值。
        输入 shape: (B, 1, D1, D2, ...)
        输出 shape: (B, num, D1, D2, ...)
        """
        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch: a.shape={a.shape}, b.shape={b.shape}")

        B, _, *rest = a.shape
        # 插值系数 (num,) → reshape 成 (1, num, 1, 1, ...)
        alphas = torch.linspace(0, 1, num, device=a.device, dtype=a.dtype)
        view_shape = (1, num) + (1,) * len(rest)
        alphas = alphas.view(view_shape)  # (1, num, 1, 1, ...)

        # 插值 (B, num, D1, D2, ...)
        result = (1 - alphas) * a + alphas * b
        return result
    
    def calculate_dis(self, A, B, threshold=10.):
        """
        A: (b, f1, c1, c2)  bank
        B: (b, f2, c1, c2)  new data
        """

        A_flat = A.view(A.size(1), -1).clone()
        B_flat = B.view(B.size(1), -1).clone()

        dist = torch.cdist(B_flat.to(torch.float32), A_flat.to(torch.float32), p=2)

        min_dist, min_idx = dist.min(dim=1)  # (f2,)

        idx_to_add = torch.nonzero(min_dist[:1] > threshold, as_tuple=False).squeeze(1).tolist()

        if len(idx_to_add) > 0:  # 有需要添加的元素
            B_to_add = B[:, idx_to_add]  # (1, k, c1, c2)
            A_new = torch.cat([A, B_to_add], dim=1)  # (1, f1+k, c1, c2)
        else:
            A_new = A  # 没有需要添加的

        return idx_to_add, A_new, min_idx

    @torch.no_grad()
    def process_input(self, images):
        batch_size = self.batch_size
        device = self.device

        tgt_cond_tensor = self.fast_resize(images, 256, 256)
        tgt_cond_tensor = tgt_cond_tensor / 2 + 0.5

        if self.first_frame:
            mot_bbox_param, kps_ref, kps_frame1, kps_dri = self.pose_encoder.interpolate_kps_online(self.ref_cond_tensor, tgt_cond_tensor, num_interp=12+1)
            self.kps_ref = kps_ref
            self.kps_frame1 = kps_frame1
        else:
            mot_bbox_param, kps_dri = self.pose_encoder.get_kps(self.kps_ref, self.kps_frame1, tgt_cond_tensor)

        keypoints = draw_keypoints(mot_bbox_param, device=device)
        boxes = get_boxes(kps_dri)
        keypoints = rearrange(keypoints.unsqueeze(2), 'f c b h w -> b c f h w')
        keypoints = keypoints.to(device=device, dtype=self.pose_guider.dtype)

        if self.first_frame:
            pose_cond_fea = self.pose_guider(keypoints[:,:, :12])
            pose = keypoints[:,:,12:]

            ref_box = get_boxes(mot_bbox_param[:1])
            ref_face = self.crop_face_tensor(self.ref_image_tensor, ref_box[0])
            motion_face = [ref_face]
            for i, frame in enumerate(images):
                motion_face.append(self.crop_face_tensor(frame, boxes[i]))
            motion_cond_tensor = torch.cat(motion_face, dim=0).transpose(0, 1)
            motion_cond_tensor = motion_cond_tensor.unsqueeze(0)
            motion = motion_cond_tensor[:,:,1:]
            motion_hidden_states = self.motion_encoder(motion_cond_tensor[:,:,:2])
            ref_motion = motion_hidden_states[:, :1]
            dri_motion = motion_hidden_states[:, 1:]

            motion_hidden_states = self.interpolate_tensors(ref_motion, dri_motion[:,:1], num=12+1)[:,:-1]
            self.motion_bank = ref_motion

            latents = self.ref_image_latents
            noise = torch.randn_like(latents)
            latents = self.scheduler.add_noise(latents, noise, self.timesteps[-1:])
            sample =  torch.cat([self.noisy_latents_first, latents], dim=2)
            
            torch.cuda.synchronize()
            self.unet_work.prefill(latents=sample)
            self.unet_work.prefill(motion_hidden_states_out=motion_hidden_states)
            self.unet_work.prefill(pose_cond_fea_out=pose_cond_fea)
            self.first_frame = False
        else:
            pose = keypoints

            motion_face = []
            for i, frame in enumerate(images):
                motion_face.append(self.crop_face_tensor(frame, boxes[i]))
            motion = torch.cat(motion_face, dim=0).transpose(0, 1)
            motion = motion.unsqueeze(0)

        motion = motion.to(dtype = self.dtype)
        latents = self.ref_image_latents
        noise = torch.randn_like(latents)
        new_noise = self.scheduler.add_noise(latents, noise, self.timesteps[-1:])

        results = self.unet_work(output_list=["pred_video", "motion_out", "latent_first"], return_tensor=True, pose=pose, motion=motion, new_noise=new_noise)
        video = results['pred_video'].cpu().numpy()
        motion_out = results['motion_out']


        idx_to_add = []
        if self.count > 8:
            idx_to_add, self.motion_bank, idx_his = self.calculate_dis(self.motion_bank, motion_out, threshold=17.)

        if len(idx_to_add) > 0 and self.num_khf < 3:
            latents_first = results['latent_first']
            self.reference_control_writer.clear()
            self.reference_unet(
                latents_first.to(self.reference_unet.dtype),
                torch.zeros((batch_size,),dtype=self.dtype,device=self.reference_unet.device),
                encoder_hidden_states=self.encoder_hidden_states,
                return_dict=False,
            )
            reference_hidden_states = self.reference_control_writer.output()
            for name in self.reference_hidden_states_names:
                self.reference_hidden_states[name] = torch.cat([self.reference_hidden_states[name], reference_hidden_states[name]], dim=1)
            
            torch.cuda.synchronize()
            self.unet_work.prefill(**{name: self.reference_hidden_states[name] for name in self.reference_hidden_states_names})
            print('add_keyframes')
            self.num_khf += 1

        self.count += 1
        return video