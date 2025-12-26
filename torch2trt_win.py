# torch2trt.py
import os
import sys
import time
import gc
import threading
import faulthandler

import torch
from omegaconf import OmegaConf

from src.modeling.framed_models import unet_work
from diffusers import AutoencoderKL
from diffusers.models.attention_processor import AttnProcessor
from src.scheduler.scheduler_ddim import DDIMScheduler
from src.models.unet_3d_explicit_reference import UNet3DConditionModel
from src.models.motion_encoder.encoder import MotEncoder
from src.models.pose_guider import PoseGuider

from src.modeling.onnx_export import export_onnx, optimize_onnx

import tensorrt as trt

SCRIPT_START = time.time()

# ----------------------------
# (6) Watchdog: periodic stack dump
# ----------------------------
faulthandler.enable()


def start_watchdog(every_seconds: int = 300) -> threading.Thread:
    def _worker():
        while True:
            time.sleep(every_seconds)
            print("\n[WATCHDOG] dumping python stacks...", flush=True)
            faulthandler.dump_traceback(file=sys.stdout)
            sys.stdout.flush()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t


start_watchdog(300)  # every 5 minutes


def map_device(device_or_str):
    return device_or_str if isinstance(device_or_str, torch.device) else torch.device(device_or_str)


# ----------------------------
# (3)+(4 Option A) Native TensorRT build (workspace limit + disable JIT tactics)
# ----------------------------
def build_trt_engine_native(
    onnx_path: str,
    profile_obj,
    engine_path: str,
    workspace_gb: int = 8,
    fp16: bool = True,
    verbose: bool = True,
):
    """
    Build and serialize a TensorRT engine from ONNX using native TRT API.

    Implements:
      - (3) workspace limit via memory pool
      - (4) Option A: native builder + disable JIT tactics (keep EDGE_MASK_CONVOLUTIONS)
    Supports profile formats:
      - dict[name] = (min_shape, opt_shape, max_shape)
      - Polygraphy Profile-like object with `.to_trt(builder, network)` (your current case likely)
    """
    # logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)

    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)

    parser = trt.OnnxParser(network, logger)

    onnx_path = os.path.abspath(onnx_path)
    onnx_dir = os.path.dirname(onnx_path)

    print(f"[TRT] parsing ONNX: {onnx_path}", flush=True)
    print(f"[TRT] ONNX dir for external weights: {onnx_dir}", flush=True)

    cwd = os.getcwd()
    try:
        os.chdir(onnx_dir)

        # (A.1) heartbeat + timing around parser.parse()
        print("[TRT] starting parser.parse() ...", flush=True)
        t0 = time.time()
        stop_hb = False

        def _heartbeat():
            while not stop_hb:
                time.sleep(30)
                print(f"[TRT] ...still parsing ONNX ({time.time() - t0:.0f}s elapsed)", flush=True)
                sys.stdout.flush()

        hb = threading.Thread(target=_heartbeat, daemon=True)
        hb.start()

        with open(onnx_path, "rb") as f:
            try:
                ok = parser.parse(f.read())
            finally:
                stop_hb = True
                hb.join(timeout=1.0)

        if not ok:
            print("[TRT] ONNX parse failed. Errors:", flush=True)
            for i in range(parser.num_errors):
                print(parser.get_error(i), flush=True)
            raise RuntimeError("TensorRT ONNX parse failed")

        print(f"[TRT] parser.parse() done in {time.time() - t0:.1f}s", flush=True)

    finally:
        os.chdir(cwd)
    config = builder.create_builder_config()
    try:
        config.clear_flag(trt.BuilderFlag.TF32)
    except Exception:
        pass

    # (3) workspace limit (start smaller for debugging)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1024**3)))
    print(f"[TRT] workspace limit set to: {workspace_gb} GB", flush=True)

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[TRT] FP16 enabled", flush=True)

    # (4) disable JIT tactics (debug-friendly)
    try:
        config.set_tactic_sources(trt.TacticSource.EDGE_MASK_CONVOLUTIONS)
        print("[TRT] tactic sources set to EDGE_MASK_CONVOLUTIONS (JIT disabled)", flush=True)
    except Exception as e:
        print(f"[TRT] WARNING: could not set tactic sources (API not available?): {e}", flush=True)

    # Optimization profile: always use a manual clamped profile (matches working trtexec --optShapes)
    opt_profile = builder.create_optimization_profile()

    opt_profile.set_shape("sample",               (1, 4, 16, 64, 64),    (1, 4, 16, 64, 64),    (1, 4, 16, 64, 64))
    opt_profile.set_shape("encoder_hidden_states",(1, 1, 768),          (1, 1, 768),          (1, 1, 768))
    opt_profile.set_shape("motion_hidden_states", (1, 12, 32, 16),      (1, 12, 32, 16),      (1, 12, 32, 16))
    opt_profile.set_shape("motion",               (1, 3, 4, 224, 224),   (1, 3, 4, 224, 224),   (1, 3, 4, 224, 224))
    opt_profile.set_shape("pose_cond_fea",         (1, 320, 12, 64, 64), (1, 320, 12, 64, 64), (1, 320, 12, 64, 64))
    opt_profile.set_shape("pose",                 (1, 3, 4, 512, 512),   (1, 3, 4, 512, 512),   (1, 3, 4, 512, 512))
    opt_profile.set_shape("new_noise",            (1, 4, 4, 64, 64),     (1, 4, 4, 64, 64),     (1, 4, 4, 64, 64))

    opt_profile.set_shape("d00", (1, 8192, 320), (1, 8192, 320), (1, 8192, 320))
    opt_profile.set_shape("d01", (1, 8192, 320), (1, 8192, 320), (1, 8192, 320))
    opt_profile.set_shape("d10", (1, 2048, 640), (1, 2048, 640), (1, 2048, 640))
    opt_profile.set_shape("d11", (1, 2048, 640), (1, 2048, 640), (1, 2048, 640))
    opt_profile.set_shape("d20", (1, 512, 1280), (1, 512, 1280), (1, 512, 1280))
    opt_profile.set_shape("d21", (1, 512, 1280), (1, 512, 1280), (1, 512, 1280))
    opt_profile.set_shape("m",   (1, 128, 1280), (1, 128, 1280), (1, 128, 1280))

    opt_profile.set_shape("u10", (1, 512, 1280), (1, 512, 1280), (1, 512, 1280))
    opt_profile.set_shape("u11", (1, 512, 1280), (1, 512, 1280), (1, 512, 1280))
    opt_profile.set_shape("u12", (1, 512, 1280), (1, 512, 1280), (1, 512, 1280))
    opt_profile.set_shape("u20", (1, 2048, 640), (1, 2048, 640), (1, 2048, 640))
    opt_profile.set_shape("u21", (1, 2048, 640), (1, 2048, 640), (1, 2048, 640))
    opt_profile.set_shape("u22", (1, 2048, 640), (1, 2048, 640), (1, 2048, 640))
    opt_profile.set_shape("u30", (1, 8192, 320), (1, 8192, 320), (1, 8192, 320))
    opt_profile.set_shape("u31", (1, 8192, 320), (1, 8192, 320), (1, 8192, 320))
    opt_profile.set_shape("u32", (1, 8192, 320), (1, 8192, 320), (1, 8192, 320))

    config.add_optimization_profile(opt_profile)
    print("[TRT] using manual clamped optimization profile (matches trtexec --optShapes)", flush=True)


    print("[TRT] starting engine build (serialized) (can take a long time)...", flush=True)
    t0 = time.time()
    
    print("[TRT] starting build_serialized_network() ...", flush=True)
    t_build0 = time.time()
    stop_build_hb = False

    def _build_heartbeat():
        while not stop_build_hb:
            time.sleep(30)
            elapsed = time.time() - t_build0
            print(f"[TRT] ...still BUILDING engine ({elapsed:.0f}s elapsed)", flush=True)
            sys.stdout.flush()

    build_hb = threading.Thread(target=_build_heartbeat, daemon=True)
    build_hb.start()

    try:
        serialized = builder.build_serialized_network(network, config)
    finally:
        stop_build_hb = True
    
    if serialized is None:
        raise RuntimeError("TensorRT build_serialized_network returned None (build failed)")

    dt = time.time() - t0
    print(f"[TRT] engine build finished in {dt:.1f}s", flush=True)

    # Save the plan (serialized engine) directly
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(bytes(serialized))
    print(f"[TRT] engine saved to: {engine_path}", flush=True)

    # Optional: deserialize to validate it loads
    try:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(bytes(serialized))
        if engine is None:
            print("[TRT] WARNING: runtime failed to deserialize engine (saved plan may still be OK).", flush=True)
        else:
            print("[TRT] engine deserialized OK.", flush=True)
    except Exception as e:
        print(f"[TRT] WARNING: deserialize step failed: {e}", flush=True)

    return engine_path

# ----------------------------
# Main
# ----------------------------
def main():
    # parameters
    batch_size = 1
    height = 256
    width = 256
    onnx_opset = 17

    device = map_device("cuda:0")
    dtype = torch.float16

    config_path = "./configs/prompts/personalive_online.yaml"
    print(f"[DBG] Loading config: {config_path}", flush=True)
    cfg = OmegaConf.load(config_path)

    onnx_path = cfg.onnx_path
    onnx_opt_path = cfg.onnx_opt_path
    tensorrt_target_model = cfg.tensorrt_target_model

    infer_config = OmegaConf.load(cfg.inference_config)
    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)

    print("[DBG] Loading PoseGuider...", flush=True)
    pose_guider = PoseGuider().to(device=device, dtype=dtype)
    pose_guider_state_dict = torch.load(cfg.pose_guider_path, map_location="cpu")
    pose_guider.load_state_dict(pose_guider_state_dict)
    del pose_guider_state_dict

    print("[DBG] Loading MotionEncoder...", flush=True)
    motion_encoder: MotEncoder = MotEncoder().to(dtype=dtype, device=device).eval()
    motion_encoder.set_attn_processor(AttnProcessor())
    motion_encoder_state_dict = torch.load(cfg.motion_encoder_path, map_location="cpu")
    motion_encoder.load_state_dict(motion_encoder_state_dict)
    del motion_encoder_state_dict

    print("[DBG] Loading UNet3DConditionModel...", flush=True)
    denoising_unet: UNet3DConditionModel = UNet3DConditionModel.from_pretrained_2d(
        cfg.pretrained_base_model_path,
        "",
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=dtype, device=device)

    denoising_unet.load_state_dict(torch.load(cfg.denoising_unet_path, map_location="cpu"), strict=False)
    denoising_unet.load_state_dict(torch.load(cfg.temporal_module_path, map_location="cpu"), strict=False)
    denoising_unet.set_attn_processor(AttnProcessor())

    print("[DBG] Loading VAE...", flush=True)
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(device=device, dtype=dtype)
    vae.set_default_attn_processor()

    print("[DBG] Creating scheduler...", flush=True)
    scheduler = DDIMScheduler(**sched_kwargs)
    scheduler.to(device)
    timesteps = torch.tensor([0, 0, 0, 0, 333, 333, 333, 333, 666, 666, 666, 666, 999, 999, 999, 999], device=device).long()
    scheduler.set_step_length(333)

    print("[DBG] Building wrapped model...", flush=True)
    model = unet_work(
        pose_guider,
        motion_encoder,
        denoising_unet,
        vae,
        scheduler,
        timesteps,
    )

    # Ensure ONNX directories exist
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    os.makedirs(os.path.dirname(onnx_opt_path), exist_ok=True)
    os.makedirs(os.path.dirname(tensorrt_target_model), exist_ok=True)

    # Export ONNX if needed
    if not os.path.exists(onnx_path):
        print(f"[DBG] Exporting ONNX to: {onnx_path}", flush=True)
        export_onnx(
            model,
            onnx_path=onnx_path,
            opt_image_height=height,
            opt_image_width=width,
            opt_batch_size=batch_size,
            onnx_opset=onnx_opset,
            auto_cast=True,
            dtype=dtype,
            device=device,
        )
        print("[DBG] ONNX export done.", flush=True)
    else:
        print(f"[DBG] ONNX already exists: {onnx_path}", flush=True)

    # Prepare profile for TRT build (your model provides it)
    batch_size = 1
    height = 512
    width = 512

    print("[DBG] Calling get_dynamic_map() ...", flush=True)
    profile = model.get_dynamic_map(batch_size, height, width)

    # (7) Explicit flush + prints right after get_dynamic_map
    print("[DBG] get_dynamic_map() returned.", flush=True)
    print(f"[DBG] profile type: {type(profile)}", flush=True)
    try:
        if isinstance(profile, dict):
            print(f"[DBG] profile keys: {list(profile.keys())}", flush=True)
        else:
            interesting = [x for x in dir(profile) if "shape" in x.lower() or "trt" in x.lower() or "to_" in x.lower()]
            print(f"[DBG] profile dir sample: {sorted(interesting)[:40]}", flush=True)
    except Exception as e:
        print(f"[DBG] could not introspect profile: {e}", flush=True)
    sys.stdout.flush()

    # Cleanup GPU memory before ONNX optimize / TRT build
    print("[DBG] Cleaning up model + CUDA cache before ONNX optimize/TRT build...", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("[DBG] cleanup done.", flush=True)

    # Optional: keep optimize_onnx if you want the artifact, but DO NOT use it for TRT build
    do_optimize_onnx = True
    if do_optimize_onnx:
        print(f"[DBG] Optimizing ONNX: {onnx_path} -> {onnx_opt_path}", flush=True)
        optimize_onnx(onnx_path=onnx_path, onnx_opt_path=onnx_opt_path)
        print("[DBG] ONNX optimization done.", flush=True)

    # Build TRT Engine from UNOPTIMIZED ONNX (this is what worked with trtexec)
    print("[DBG] about to build TRT engine from unoptimized ONNX (native)...", flush=True)
    sys.stdout.flush()

    data_path = onnx_path + ".data"
    print(f"[DBG] Expecting external weights: {data_path}", flush=True)
    print(f"[DBG] Exists? {os.path.exists(data_path)}", flush=True)

    build_trt_engine_native(
        onnx_path=onnx_path,          # <<< critical change
        profile_obj=profile,          # kept, but we won’t rely on it
        engine_path=tensorrt_target_model,
        workspace_gb=8,
        fp16=True,
        verbose=False,
    )


    print("[DBG] TRT build step completed.", flush=True)
    gc.collect()
    torch.cuda.empty_cache()
    print("[DBG] Done.", flush=True)
    total = time.time() - SCRIPT_START
    print(f"[TIME] Total torch2trt.py runtime: {total:.1f}s ({total/60:.1f} min)", flush=True)

if __name__ == "__main__":
    main()
