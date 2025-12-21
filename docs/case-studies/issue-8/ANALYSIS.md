# Case Study: Issue #8 - Video Jerky When Using batch_size

## Executive Summary

**Issue**: When using `batch_size` parameter in `inference_offline.py`, the generated video has jerky/stuttering artifacts at batch boundaries.

**Root Cause**: Independent batch processing without temporal context sharing leads to discontinuities between consecutive batches.

**Proposed Solution**: Implement a memory-efficient sliding window approach that maintains temporal coherence while staying within VRAM limits of 12-16GB GPUs.

---

## Timeline of Events

### Phase 1: Initial Implementation (Commit 469c579 - PR #2)
**Date**: December 21, 2025 10:40 UTC

The first attempt added basic batch processing to reduce VRAM usage:
- Introduced `--batch_size` CLI parameter
- Processed frames in non-overlapping batches
- Added `gc.collect()` and `torch.cuda.empty_cache()` between batches

**Problem**: Video was created with visible jerks/stutters at batch boundaries.

### Phase 2: Overlapping Batches (Commit 8bbacd1 - PR #9)
**Date**: December 21, 2025 14:09 UTC

Attempt to fix jerky video by implementing overlapping batches:
- Used 12-frame overlap between consecutive batches
- Discarded first 12 frames of each batch after the first

**Author Comment (from AI model)**:
> "This overlap strategy will increase computational costs, and discontinuities may still occur at the connection points between batches. A better strategy would be to use sliding window generation."

**Problem**: Video still not smooth enough, and processing time increased.

### Phase 3: Sliding Window Implementation (Commit 5abd555 - PR #10)
**Date**: December 21, 2025 15:02 UTC

Implemented true sliding window generation based on `src/wrapper.py`:
- Used deques (`latents_pile`, `pose_pile`, `motion_pile`) for continuous state
- Precomputed all motion embeddings and keypoints upfront
- Included history keyframe mechanism

**Problem**: Caused CUDA Out of Memory errors during preprocessing.

### Phase 4: Memory Optimization Attempt (Commit 2d4a721 - PR #11)
**Date**: December 21, 2025 15:54 UTC

Attempted to fix OOM by:
- Using `interpolate_kps_online()` + `get_kps()` instead of `interpolate_kps()` per frame
- Processing keypoints in batches of 32-64 frames
- Moving intermediate results to CPU

**Current Status**: Still experiencing OOM errors. User requested revert to commit 435b815.

---

## Root Cause Analysis

### 1. Jerky Video Problem

The fundamental issue stems from how video diffusion models handle temporal coherence:

```
Batch 1: Frames [0-31] → Denoised independently
Batch 2: Frames [32-63] → Denoised independently (no context from Batch 1)
...
```

At each batch boundary, there's a **temporal discontinuity** because:
- The denoising UNet has no information about previously generated frames
- Latent states are reset at each batch start
- No motion interpolation between batches

### 2. CUDA OOM in Sliding Window Mode

The OOM error occurred at line 397 in PR #10's implementation:

```python
mot_param = pose_encoder.interpolate_kps(ref_cond_tensor, all_tgt_cond_tensors[i], num_interp=1)
```

**Root Causes**:
1. **Preprocessing Memory Explosion**: Precomputing all motion embeddings and keypoints for 200 frames before starting generation
2. **Tensor Accumulation**: All preprocessed tensors held in GPU memory simultaneously
3. **Inefficient Keypoint Extraction**: Calling `interpolate_kps()` per frame creates new detector tensors each time
4. **Large Batch Sizes**: Processing 256 frames at once for pose features/motion encoding

**Memory Flow**:
```
Initial state: ~5GB (models loaded)
After preprocessing: ~15GB+ (all tensors accumulated)
Result: OOM on 15.57GB GPU (RTX 4070 SUPER)
```

### 3. The Core Tradeoff

| Approach | VRAM Usage | Video Quality | Processing Time |
|----------|-----------|---------------|-----------------|
| All-at-once (original) | Very High | Smooth | Fast |
| Independent Batches | Low | Jerky | Fast |
| Overlapping Batches | Medium | Slightly Jerky | Slower |
| Full Precompute Sliding | Very High (OOM) | Smooth | Medium |
| Streaming Sliding | Low | Smooth | Medium |

---

## Analysis of src/wrapper.py

The online/real-time wrapper (`src/wrapper.py`) successfully implements sliding window generation:

**Key Patterns**:
1. **Streaming Input**: Processes 4 frames at a time, never stores all frames
2. **Rolling State**: Uses deques with fixed capacity for latents/pose/motion
3. **Lazy Computation**: Computes motion/pose features just-in-time, not upfront
4. **First Frame Bootstrap**: Uses `interpolate_kps_online()` for first frame, `get_kps()` thereafter

**Critical Code (lines 284-289)**:
```python
if self.first_frame:
    mot_bbox_param, kps_ref, kps_frame1, kps_dri = self.pose_encoder.interpolate_kps_online(...)
else:
    mot_bbox_param, kps_dri = self.pose_encoder.get_kps(self.kps_ref, self.kps_frame1, tgt_cond_tensor)
```

---

## Proposed Solution

### Strategy: On-the-Fly Sliding Window

Instead of precomputing all tensors, process frames in a streaming fashion:

```python
# Initialize once
latents_pile = deque(maxlen=temporal_adaptive_step)  # Fixed size = 4 windows
pose_pile = deque(maxlen=temporal_adaptive_step)
motion_pile = deque(maxlen=temporal_adaptive_step)

# Bootstrap with reference image
initialize_piles_with_padding()  # 12 frames from ref image

# Stream through video
for window_idx in range(num_windows):
    # 1. Get next 4 frames
    frames = get_next_window_frames(window_idx)

    # 2. Compute features just-in-time
    if window_idx == 0:
        mot_param, kps_ref, kps_frame1 = pose_encoder.interpolate_kps_online(ref, frames[0])
    else:
        mot_param = pose_encoder.get_kps(kps_ref, kps_frame1, frames)

    keypoints = draw_keypoints(mot_param)
    pose_fea = pose_guider(keypoints)
    motion_hidden = motion_encoder(face_crops)

    # 3. Add to piles (deque auto-pops old entries)
    pose_pile.append(pose_fea)
    motion_pile.append(motion_hidden)
    latents_pile.append(new_noisy_latents)

    # 4. Denoise with full context
    combined_latents = torch.cat(list(latents_pile), dim=2)
    combined_pose = torch.cat(list(pose_pile), dim=2)
    combined_motion = torch.cat(list(motion_pile), dim=1)

    denoised = denoise(combined_latents, combined_pose, combined_motion)

    # 5. Pop and decode oldest window
    completed = latents_pile.popleft()
    decoded_frames.append(vae.decode(completed))

    # 6. Clear memory
    gc.collect()
    torch.cuda.empty_cache()
```

### Memory Profile

| Stage | GPU Memory |
|-------|-----------|
| Models loaded | ~5 GB |
| 4 windows in pile (16 latent frames) | ~1.5 GB |
| Pose features (4 windows) | ~0.5 GB |
| Motion hidden states (4 windows) | ~0.3 GB |
| Temporary computation | ~2 GB |
| **Peak Total** | **~9.3 GB** |

This fits comfortably within 12-16GB GPUs.

---

## References

### Academic Literature

1. [Fast Video Generation with Sliding Tile Attention](https://arxiv.org/html/2502.04507v1) - Sliding attention for video diffusion
2. [DiffuseSlide: Training-Free High FPS Video Generation](https://arxiv.org/html/2506.01454) - Sliding window latent denoising
3. [FreeSwim: Sliding-Window Attention for Ultra-High-Resolution Video](https://arxiv.org/html/2511.14712) - Inward sliding-window attention

### PyTorch Memory Management

1. [PyTorch FAQ on Memory](https://docs.pytorch.org/docs/stable/notes/faq.html) - Official memory management guidance
2. [CUDA Semantics Documentation](https://docs.pytorch.org/docs/stable/notes/cuda.html) - GPU memory handling
3. [Solving CUDA OOM Errors](https://saturncloud.io/blog/how-to-solve-cuda-out-of-memory-error-in-pytorch/) - Best practices

---

## Files Preserved

- `inference_offline_original_435b815.py` - Original state before batch processing
- `inference_offline_current_main.py` - Current main branch state with OOM issues
- `ANALYSIS.md` - This document

---

## Conclusion

The solution requires implementing a **streaming sliding window approach** that:
1. Never precomputes all frames upfront
2. Maintains fixed-size deques for temporal context
3. Computes features just-in-time as frames are processed
4. Clears memory after each window is decoded

This matches the design pattern in `src/wrapper.py` but adapted for offline/batch video processing.
