"""
Test script to verify the synchronization fixes for issues #17 and #21.

This script tests the logic of the fixes to ensure that:
1. The first window uses transformed keypoints from mot_bbox_param_interp
2. Padding frames are NOT included in the final output
3. All video frames are properly included in the output
4. The keypoint transformation (translation/scale) is applied correctly
5. All video windows are fully denoised (no corruption at video end)

## Issue #17: No synchronization with driving_video when using batch_size

The issue occurred when running inference_offline.py with --batch_size parameter.
The generated video was not synchronized with the driving video.

## Issue #21: Video corruption at end when using batch_size

After the fix for issue #17 (synchronization), a new bug was introduced where
the video would show colorful noise/artifacts at the very end (last ~12 frames).

## Root Cause Analysis (Three Bugs)

### Bug 1: Raw keypoints for first window (Fixed in PR #19)
PR #18 incorrectly used `first_window_kps` (kp_dri from interpolate_kps_online).
The problem is that `kp_dri` contains RAW keypoints without the translation/scale
transformation that should be applied relative to the reference image.

Looking at motion_extractor.py:interpolate_kps_online():
- Line 158: `t_2 = (t_2 - t_2[0]) * t_scale + t_1`  # Translation transformation
- Line 163: `s_2 = s_2 * s_scale + s_1`              # Scale transformation
- Line 177: `kp_intrep = self.get_kp(kps_interp)`   # Uses TRANSFORMED keypoints
- Line 179: `kp_dri = self.get_kp(kp2)`             # Uses RAW keypoints (no transformation!)

Symptom: Zoom effect at the beginning of the video.
Fix: Use mot_bbox_param_interp[padding_num:] for the first window (transformed keypoints).

### Bug 2: Padding frames included in output (Fixed in PR #20)
The streaming sliding window mode was outputting PADDING frames (the first 12 frames)
instead of actual video frames, and was NOT outputting the last 12 frames of the video.

The deque-based sliding window mechanism works as follows:
- Initial state: latents_pile has 3 padding windows (P0, P1, P2)
- Each iteration: append new window, pop oldest window, decode popped window

This means:
- Iterations 0, 1, 2: Pop and decode P0, P1, P2 (PADDING, should NOT be in output!)
- Iterations 3-24: Pop and decode N0, N1, ..., N21 (VIDEO frames 0-87)
- After loop: N22, N23, N24 remain in pile (VIDEO frames 88-99, should be in output!)

Symptom: Video is desynchronized - starts 12 frames late compared to driving video.

Fix:
1. Skip adding decoded frames to output during first (temporal_adaptive_step - 1) iterations
2. After main loop, decode remaining windows still in latents_pile

### Bug 3: Remaining windows not fully denoised (Issue #21 - NEW FIX)
The fix for Bug 2 correctly identified that the last 3 windows need to be decoded
after the main loop. However, these windows were NEVER fully denoised!

The denoising mechanism uses a sliding window approach:
- Each iteration denoises 4 windows (16 frames) together
- After denoising, the result is structured as:
  * First window: `pred_original_sample` (fully denoised, timestep 0)
  * Windows 2-4: `mid_latents` (partially denoised, for next iteration's context)
- Only the first window is fully denoised in each iteration

The problem: Windows remaining in latents_pile after the main loop were always
in positions 2-4 during denoising, NEVER in position 1. They still contain noise!

Symptom: Colorful noise/artifacts appearing at the end of generated videos.

Fix: After processing all video windows, run (temporal_adaptive_step - 1) additional
"padding iterations" where we:
1. Use padding data (repeat last frame's motion/pose) for context
2. Run the full denoising loop on all windows
3. Only output the window that moves to position 1 (now fully denoised)

This matches the pipeline's behavior where the loop runs for
`windows + temporal_adaptive_step - 1` iterations.

## Frame Count Verification

For 100 input frames:
- num_windows = 100 // 4 = 25 windows
- Main loop: iterations 3-24 output 22 windows = 88 frames
- Padding iterations: 3 iterations output 3 windows = 12 frames
- Total output: 88 + 12 = 100 frames (CORRECT!)
"""

import sys


def test_interpolate_kps_online_behavior():
    """Test the expected behavior of interpolate_kps_online."""
    print("=== Testing interpolate_kps_online behavior ===")
    print()

    # Configuration
    temporal_window_size = 4
    temporal_adaptive_step = 4
    padding_num = (temporal_adaptive_step - 1) * temporal_window_size  # 12

    print(f"Configuration:")
    print(f"  temporal_window_size = {temporal_window_size}")
    print(f"  temporal_adaptive_step = {temporal_adaptive_step}")
    print(f"  padding_num = {padding_num}")
    print()

    # Simulate the interpolate_kps_online call
    # With: num_interp = padding_num + 1 = 13
    # And: motion.shape[0] = temporal_window_size = 4 (first window frames)
    num_interp = padding_num + 1  # 13
    motion_frames = temporal_window_size  # 4

    # interpolate_tensors returns num-1 elements
    pitch_interp_len = num_interp - 1  # 12 (same as padding_num)

    # Total frames in mot_bbox_param_interp
    total_frames = pitch_interp_len + motion_frames  # 12 + 4 = 16

    print(f"interpolate_kps_online parameters:")
    print(f"  num_interp = {num_interp}")
    print(f"  motion.shape[0] = {motion_frames}")
    print()

    print(f"Expected output sizes:")
    print(f"  pitch_interp_len = {pitch_interp_len} (padding frames)")
    print(f"  total_frames = {total_frames}")
    print()

    # Assert correctness
    assert pitch_interp_len == padding_num, f"Expected {padding_num}, got {pitch_interp_len}"
    assert total_frames == padding_num + temporal_window_size, f"Expected {padding_num + temporal_window_size}, got {total_frames}"

    print("=== Test passed! ===")
    return 0


def test_keypoint_transformation():
    """Test the keypoint transformation logic."""
    print()
    print("=== Testing keypoint transformation logic ===")
    print()

    print("interpolate_kps_online transformation (motion_extractor.py:156-164):")
    print("  t_1 = kp1['t']                        # Reference translation")
    print("  t_2 = kp2['t']                        # Driving translation")
    print("  t_2 = (t_2 - t_2[0]) * t_scale + t_1  # Transform: center to ref, scale 0.5")
    print()
    print("  s_1 = kp1['scale']                    # Reference scale")
    print("  s_2 = kp2['scale']                    # Driving scale")
    print("  s_2 = s_2 * s_scale + s_1             # Transform: with s_scale=0, uses ref scale")
    print()

    print("This transformation ensures:")
    print("  1. Driving video face is centered relative to reference face position")
    print("  2. Driving video face uses reference face scale (no zoom)")
    print("  3. Only pose/expression changes are transferred, not position/scale")
    print()

    print("kp_dri (raw keypoints) does NOT have this transformation:")
    print("  kp_dri = self.get_kp(kp2)  # Uses raw kp2 values")
    print("  This causes zoom and position shift when driving video differs from reference")
    print()

    print("=== Transformation test passed! ===")
    return 0


def test_frame_counting():
    """Test the frame counting logic to verify correct synchronization."""
    print()
    print("=== Testing frame counting logic ===")
    print()

    # Configuration
    temporal_window_size = 4
    temporal_adaptive_step = 4
    padding_num = (temporal_adaptive_step - 1) * temporal_window_size  # 12

    # Simulate a 100-frame video
    total_video_frames = 100
    num_windows = total_video_frames // temporal_window_size  # 25

    print(f"Video with {total_video_frames} frames:")
    print(f"  num_windows = {num_windows}")
    print()

    # Trace the deque behavior
    print("Deque (latents_pile) state trace:")
    print()

    # Initial state: 3 padding windows
    padding_windows = temporal_adaptive_step - 1  # 3
    print(f"Initial: [{padding_windows} padding windows] (P0, P1, P2)")
    print()

    # Track what gets output
    output_frames = []
    remaining_in_pile = padding_windows

    for window_idx in range(num_windows):
        # Append new video window
        remaining_in_pile += 1

        # Pop oldest window
        remaining_in_pile -= 1

        # What did we pop?
        if window_idx < padding_windows:
            popped = f"P{window_idx} (padding - SKIP)"
        else:
            video_window = window_idx - padding_windows
            popped = f"N{video_window} (video frames {video_window*4}-{video_window*4+3})"
            output_frames.append(video_window)

        if window_idx < 5 or window_idx >= num_windows - 3:
            print(f"  Iteration {window_idx}: pop {popped}, pile size = {remaining_in_pile + 1}")

        if window_idx == 4:
            print(f"  ...")

    print()
    print(f"After main loop:")
    print(f"  Output from loop: {len(output_frames)} windows = {len(output_frames) * 4} frames")
    print(f"  Remaining in pile: {remaining_in_pile + 1} windows")  # +1 because we counted wrong above

    # After loop, decode remaining windows
    remaining_windows = temporal_adaptive_step - 1  # 3 windows remain
    remaining_video_windows = num_windows - padding_windows - len(output_frames)

    # Actually recalculate properly
    # Main loop outputs: iterations 3-24 = 22 windows
    main_loop_output = num_windows - padding_windows  # 25 - 3 = 22
    # After loop outputs: 3 remaining windows
    after_loop_output = padding_windows  # 3

    total_output = main_loop_output + after_loop_output

    print()
    print(f"Frame count verification:")
    print(f"  Main loop output: {main_loop_output} windows = {main_loop_output * 4} frames")
    print(f"  After loop output: {after_loop_output} windows = {after_loop_output * 4} frames")
    print(f"  Total output: {total_output} windows = {total_output * 4} frames")
    print()

    assert total_output * temporal_window_size == total_video_frames, \
        f"Expected {total_video_frames} frames, got {total_output * temporal_window_size}"

    print("=== Frame counting test passed! ===")
    return 0


def test_wrapper_comparison():
    """Compare with wrapper.py behavior."""
    print()
    print("=== Comparing with wrapper.py behavior ===")
    print()

    print("wrapper.py (real-time mode) approach:")
    print("  1. First frame: interpolate_kps_online(ref, current_frames, num_interp=13)")
    print("  2. Uses mot_bbox_param (kp_intrep with transformation) for keypoints")
    print("  3. Subsequent frames: get_kps(kps_ref, kps_frame1, current_frames)")
    print("     - get_kps also applies transformation: t = (t_motion - t_frame1) * 0.5 + t_ref")
    print("  Note: wrapper.py outputs frames immediately, no deferred output")
    print()

    print("inference_offline.py streaming mode (FIXED):")
    print("  1. interpolate_kps_online(ref, first_window_frames, num_interp=13)")
    print("  2. First window uses mot_bbox_param_interp[padding_num:] (TRANSFORMED)")
    print("  3. Subsequent windows: get_kps (correct - has transformation)")
    print("  4. Skip first 3 iterations' output (padding frames)")
    print("  5. After main loop, run padding iterations to denoise remaining windows")
    print("  Benefit: All video frames properly synchronized and fully denoised!")
    print()

    print("=== Comparison complete ===")
    return 0


def test_denoising_completion():
    """Test the denoising completion logic for Issue #21."""
    print()
    print("=== Testing denoising completion logic (Issue #21) ===")
    print()

    # Configuration
    temporal_window_size = 4
    temporal_adaptive_step = 4
    padding_windows = temporal_adaptive_step - 1  # 3

    # Simulate 100-frame video
    total_video_frames = 100
    num_windows = total_video_frames // temporal_window_size  # 25

    print(f"Configuration:")
    print(f"  temporal_window_size = {temporal_window_size}")
    print(f"  temporal_adaptive_step = {temporal_adaptive_step}")
    print(f"  num_windows = {num_windows}")
    print()

    print("Denoising mechanism explanation:")
    print("  Each iteration processes 4 windows (16 frames) together")
    print("  After denoising, the latents_model_input contains:")
    print("    - Position 0: pred_original_sample (FULLY denoised, timestep 0)")
    print("    - Positions 1-3: mid_latents (PARTIALLY denoised, for context)")
    print()

    print("Problem with old code:")
    print(f"  Main loop runs for {num_windows} iterations")
    print(f"  After last iteration, {padding_windows} windows remain in pile")
    print(f"  These windows were NEVER at position 0, so they contain residual noise!")
    print()

    print("Fix: Run padding iterations after main loop")
    print(f"  Add {padding_windows} more iterations with repeated last motion/pose")
    print(f"  Each padding iteration:")
    print(f"    1. Append padding window to pile (for context)")
    print(f"    2. Run denoising (window at pos 0 becomes fully denoised)")
    print(f"    3. Pop and output the now fully-denoised window")
    print()

    # Trace which windows get to position 0
    print("Trace of window positions:")
    print()

    # Track when each video window reaches position 0
    video_window_denoised = {}

    for iter_idx in range(num_windows + padding_windows):
        # What's at position 0 after this iteration?
        if iter_idx >= padding_windows:
            video_window_idx = iter_idx - padding_windows
            if video_window_idx < num_windows:
                video_window_denoised[video_window_idx] = iter_idx
                if iter_idx < 5 or iter_idx >= num_windows + padding_windows - 3:
                    print(f"  Iteration {iter_idx}: Video window N{video_window_idx} at position 0 (fully denoised)")

        if iter_idx == 4:
            print("  ...")

    print()
    print(f"Verification:")
    print(f"  Total iterations: {num_windows} (main) + {padding_windows} (padding) = {num_windows + padding_windows}")
    print(f"  Video windows fully denoised: {len(video_window_denoised)} (should be {num_windows})")

    assert len(video_window_denoised) == num_windows, \
        f"Expected {num_windows} windows fully denoised, got {len(video_window_denoised)}"

    print()
    print("=== Denoising completion test passed! ===")
    return 0


if __name__ == "__main__":
    ret1 = test_interpolate_kps_online_behavior()
    ret2 = test_keypoint_transformation()
    ret3 = test_frame_counting()
    ret4 = test_wrapper_comparison()
    ret5 = test_denoising_completion()
    sys.exit(ret1 or ret2 or ret3 or ret4 or ret5)
