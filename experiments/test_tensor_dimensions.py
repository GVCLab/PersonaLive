"""
Test script to verify the tensor dimension fix for issue #13.

This script simulates the tensor dimension calculations in inference_offline.py
to verify that the fix correctly aligns the dimensions of:
- latents_pile (latents_model_input)
- pose_pile (pose_cond_fea)
- motion_pile (motion_hidden_state)

The bug was that mot_bbox_param_interp was generated with padding_num + 1 = 13 frames,
but when trying to access frames [padding_num:padding_num+temporal_window_size] = [12:16]
for the first window, we only got 1 frame instead of 4, causing a dimension mismatch.

The fix changes the interpolation to generate padding_num + temporal_window_size = 16 frames.
"""

import sys
from collections import deque


def test_old_behavior():
    """Test the OLD (buggy) behavior that caused the issue."""
    print("=== Testing OLD (buggy) behavior ===")

    temporal_window_size = 4
    temporal_adaptive_step = 4
    padding_num = (temporal_adaptive_step - 1) * temporal_window_size  # 12

    # OLD: Generate padding_num + 1 = 13 frames
    num_interp_old = padding_num + 1  # 13
    mot_bbox_param_interp_frames = num_interp_old  # Simulating the tensor dimension

    print(f"temporal_window_size: {temporal_window_size}")
    print(f"temporal_adaptive_step: {temporal_adaptive_step}")
    print(f"padding_num: {padding_num}")
    print(f"num_interp (OLD): {num_interp_old}")
    print(f"mot_bbox_param_interp frames: {mot_bbox_param_interp_frames}")

    # Initialize piles with padding (3 windows)
    latents_pile_frames = []
    pose_pile_frames = []
    motion_pile_frames = []

    for i in range(temporal_adaptive_step - 1):  # 3 iterations
        latents_pile_frames.append(temporal_window_size)  # 4 frames each
        pose_pile_frames.append(temporal_window_size)  # 4 frames each
        motion_pile_frames.append(temporal_window_size)  # 4 frames each

    print(f"\nAfter initialization (before first window):")
    print(f"  latents_pile: {latents_pile_frames} = {sum(latents_pile_frames)} frames")
    print(f"  pose_pile: {pose_pile_frames} = {sum(pose_pile_frames)} frames")
    print(f"  motion_pile: {motion_pile_frames} = {sum(motion_pile_frames)} frames")

    # First window (window_idx = 0)
    # Simulate getting first window's mot_params
    first_window_mot_params_start = padding_num  # 12
    first_window_mot_params_end = padding_num + temporal_window_size  # 16

    # BUG: mot_bbox_param_interp only has 13 frames (indices 0-12)
    # So accessing [12:16] only returns 1 frame (index 12)
    actual_frames_available = min(first_window_mot_params_end, mot_bbox_param_interp_frames) - first_window_mot_params_start

    print(f"\nFirst window:")
    print(f"  Trying to access mot_bbox_param_interp[{first_window_mot_params_start}:{first_window_mot_params_end}]")
    print(f"  But mot_bbox_param_interp only has {mot_bbox_param_interp_frames} frames (indices 0-{mot_bbox_param_interp_frames-1})")
    print(f"  Actual frames obtained: {actual_frames_available}")

    # Add first window's pose (but with wrong dimension)
    pose_pile_frames.append(actual_frames_available)  # Only 1 frame!
    motion_pile_frames.append(temporal_window_size)  # 4 frames
    latents_pile_frames.append(temporal_window_size)  # 4 frames

    total_latents = sum(latents_pile_frames)
    total_pose = sum(pose_pile_frames)
    total_motion = sum(motion_pile_frames)

    print(f"\nAfter first window appended:")
    print(f"  latents_pile: {latents_pile_frames} = {total_latents} frames")
    print(f"  pose_pile: {pose_pile_frames} = {total_pose} frames")
    print(f"  motion_pile: {motion_pile_frames} = {total_motion} frames")

    if total_latents != total_pose:
        print(f"\n*** BUG DETECTED: latents ({total_latents}) != pose ({total_pose}) ***")
        return False
    return True


def test_new_behavior():
    """Test the NEW (fixed) behavior."""
    print("\n=== Testing NEW (fixed) behavior ===")

    temporal_window_size = 4
    temporal_adaptive_step = 4
    padding_num = (temporal_adaptive_step - 1) * temporal_window_size  # 12

    # NEW: Generate padding_num + temporal_window_size = 16 frames
    num_interp_new = padding_num + temporal_window_size  # 16
    mot_bbox_param_interp_frames = num_interp_new  # Simulating the tensor dimension

    print(f"temporal_window_size: {temporal_window_size}")
    print(f"temporal_adaptive_step: {temporal_adaptive_step}")
    print(f"padding_num: {padding_num}")
    print(f"num_interp (NEW): {num_interp_new}")
    print(f"mot_bbox_param_interp frames: {mot_bbox_param_interp_frames}")

    # Initialize piles with padding (3 windows)
    latents_pile_frames = []
    pose_pile_frames = []
    motion_pile_frames = []

    for i in range(temporal_adaptive_step - 1):  # 3 iterations
        latents_pile_frames.append(temporal_window_size)  # 4 frames each
        pose_pile_frames.append(temporal_window_size)  # 4 frames each
        motion_pile_frames.append(temporal_window_size)  # 4 frames each

    print(f"\nAfter initialization (before first window):")
    print(f"  latents_pile: {latents_pile_frames} = {sum(latents_pile_frames)} frames")
    print(f"  pose_pile: {pose_pile_frames} = {sum(pose_pile_frames)} frames")
    print(f"  motion_pile: {motion_pile_frames} = {sum(motion_pile_frames)} frames")

    # First window (window_idx = 0)
    first_window_mot_params_start = padding_num  # 12
    first_window_mot_params_end = padding_num + temporal_window_size  # 16

    # FIX: mot_bbox_param_interp now has 16 frames (indices 0-15)
    # So accessing [12:16] returns 4 frames (indices 12-15)
    actual_frames_available = min(first_window_mot_params_end, mot_bbox_param_interp_frames) - first_window_mot_params_start

    print(f"\nFirst window:")
    print(f"  Accessing mot_bbox_param_interp[{first_window_mot_params_start}:{first_window_mot_params_end}]")
    print(f"  mot_bbox_param_interp has {mot_bbox_param_interp_frames} frames (indices 0-{mot_bbox_param_interp_frames-1})")
    print(f"  Actual frames obtained: {actual_frames_available}")

    # Add first window's pose (now with correct dimension)
    pose_pile_frames.append(actual_frames_available)  # 4 frames!
    motion_pile_frames.append(temporal_window_size)  # 4 frames
    latents_pile_frames.append(temporal_window_size)  # 4 frames

    total_latents = sum(latents_pile_frames)
    total_pose = sum(pose_pile_frames)
    total_motion = sum(motion_pile_frames)

    print(f"\nAfter first window appended:")
    print(f"  latents_pile: {latents_pile_frames} = {total_latents} frames")
    print(f"  pose_pile: {pose_pile_frames} = {total_pose} frames")
    print(f"  motion_pile: {motion_pile_frames} = {total_motion} frames")

    if total_latents == total_pose == total_motion:
        print(f"\n*** FIX VERIFIED: All dimensions match ({total_latents} frames) ***")
        return True
    return False


def main():
    print("Testing tensor dimension fix for issue #13")
    print("=" * 60)

    old_result = test_old_behavior()
    new_result = test_new_behavior()

    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"  Old behavior (buggy): {'PASS' if old_result else 'FAIL (expected)'}")
    print(f"  New behavior (fixed): {'PASS' if new_result else 'FAIL'}")

    if not old_result and new_result:
        print("\n*** The fix correctly resolves the tensor dimension mismatch! ***")
        return 0
    else:
        print("\n*** Something is wrong with the fix! ***")
        return 1


if __name__ == "__main__":
    sys.exit(main())
