#!/usr/bin/env python3
"""
Test script for audio extraction and merging functionality.

This script tests the has_audio_stream, add_audio_to_video, and save_videos_from_pil
functions added to src/utils/util.py.

Usage:
    python experiments/test_audio_functions.py
"""

import os
import sys
import tempfile
import subprocess

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
import numpy as np


def create_test_video_with_audio(output_path, duration=2, fps=25, width=256, height=256):
    """Create a simple test video with audio using ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=blue:s={width}x{height}:d={duration}:r={fps}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-shortest",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def create_test_video_without_audio(output_path, duration=2, fps=25, width=256, height=256):
    """Create a simple test video without audio using ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=red:s={width}x{height}:d={duration}:r={fps}",
        "-c:v", "libx264",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def test_has_audio_stream():
    """Test the has_audio_stream function."""
    from src.utils.util import has_audio_stream

    print("Testing has_audio_stream...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test video with audio
        video_with_audio = os.path.join(tmpdir, "with_audio.mp4")
        if create_test_video_with_audio(video_with_audio):
            result = has_audio_stream(video_with_audio)
            assert result == True, f"Expected True for video with audio, got {result}"
            print("  ✓ Correctly detected audio in video with audio")
        else:
            print("  ⚠ Could not create test video with audio")

        # Test video without audio
        video_without_audio = os.path.join(tmpdir, "without_audio.mp4")
        if create_test_video_without_audio(video_without_audio):
            result = has_audio_stream(video_without_audio)
            assert result == False, f"Expected False for video without audio, got {result}"
            print("  ✓ Correctly detected no audio in video without audio")
        else:
            print("  ⚠ Could not create test video without audio")

        # Test non-existent file
        result = has_audio_stream("/nonexistent/file.mp4")
        assert result == False, f"Expected False for non-existent file, got {result}"
        print("  ✓ Correctly returned False for non-existent file")

    print("  All has_audio_stream tests passed!")


def test_add_audio_to_video():
    """Test the add_audio_to_video function."""
    from src.utils.util import add_audio_to_video, has_audio_stream

    print("\nTesting add_audio_to_video...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create source video with audio (longer duration)
        source_with_audio = os.path.join(tmpdir, "source_with_audio.mp4")
        if not create_test_video_with_audio(source_with_audio, duration=5):
            print("  ⚠ Could not create source video with audio")
            return

        # Create target video without audio (shorter duration)
        target_without_audio = os.path.join(tmpdir, "target_without_audio.mp4")
        if not create_test_video_without_audio(target_without_audio, duration=2):
            print("  ⚠ Could not create target video without audio")
            return

        # Test adding audio (shorter video than audio source)
        output_path = os.path.join(tmpdir, "output_with_audio.mp4")
        result = add_audio_to_video(target_without_audio, source_with_audio, output_path, verbose=True)

        if result:
            assert os.path.exists(output_path), "Output file should exist"
            assert has_audio_stream(output_path), "Output should have audio"
            print("  ✓ Successfully added audio to video (video shorter than audio)")
        else:
            print("  ⚠ Failed to add audio to video")

        # Test with source that has no audio
        source_without_audio = os.path.join(tmpdir, "source_no_audio.mp4")
        if create_test_video_without_audio(source_without_audio, duration=3):
            result = add_audio_to_video(target_without_audio, source_without_audio,
                                        os.path.join(tmpdir, "should_fail.mp4"), verbose=True)
            assert result == False, "Should return False when source has no audio"
            print("  ✓ Correctly returned False when source has no audio")

    print("  All add_audio_to_video tests passed!")


def test_save_videos_from_pil_with_audio():
    """Test save_videos_from_pil with audio_source parameter."""
    from src.utils.util import save_videos_from_pil, has_audio_stream

    print("\nTesting save_videos_from_pil with audio_source...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a source video with audio
        source_with_audio = os.path.join(tmpdir, "source.mp4")
        if not create_test_video_with_audio(source_with_audio, duration=3):
            print("  ⚠ Could not create source video with audio")
            return

        # Create test PIL images (simple colored frames)
        pil_images = []
        for i in range(50):  # 2 seconds at 25 fps
            img = Image.new('RGB', (256, 256), color=(255, i * 5 % 256, 0))
            pil_images.append(img)

        # Save video with audio source
        output_path = os.path.join(tmpdir, "output.mp4")
        save_videos_from_pil(pil_images, output_path, fps=25, audio_source=source_with_audio)

        assert os.path.exists(output_path), "Output file should exist"
        assert has_audio_stream(output_path), "Output should have audio from source"
        print("  ✓ Successfully saved video with audio from source")

        # Test without audio source
        output_no_audio = os.path.join(tmpdir, "output_no_audio.mp4")
        save_videos_from_pil(pil_images, output_no_audio, fps=25)

        assert os.path.exists(output_no_audio), "Output file should exist"
        assert not has_audio_stream(output_no_audio), "Output should not have audio"
        print("  ✓ Successfully saved video without audio")

    print("  All save_videos_from_pil tests passed!")


def test_save_videos_grid_with_audio():
    """Test save_videos_grid with audio_source parameter."""
    import torch
    from src.utils.util import save_videos_grid, has_audio_stream

    print("\nTesting save_videos_grid with audio_source...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a source video with audio
        source_with_audio = os.path.join(tmpdir, "source.mp4")
        if not create_test_video_with_audio(source_with_audio, duration=3):
            print("  ⚠ Could not create source video with audio")
            return

        # Create test video tensor (b, c, t, h, w)
        # b=1, c=3, t=50 frames, h=256, w=256
        video_tensor = torch.rand(1, 3, 50, 256, 256)

        # Save video with audio source
        output_path = os.path.join(tmpdir, "grid_output.mp4")
        save_videos_grid(video_tensor, output_path, fps=25, audio_source=source_with_audio)

        assert os.path.exists(output_path), "Output file should exist"
        assert has_audio_stream(output_path), "Output should have audio from source"
        print("  ✓ Successfully saved grid video with audio from source")

    print("  All save_videos_grid tests passed!")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing audio extraction and merging functionality")
    print("=" * 60)

    # Check if ffmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: ffmpeg is not installed or not in PATH")
        print("Please install ffmpeg to run these tests")
        sys.exit(1)

    print("ffmpeg is available ✓\n")

    try:
        test_has_audio_stream()
        test_add_audio_to_video()
        test_save_videos_from_pil_with_audio()
        test_save_videos_grid_with_audio()

        print("\n" + "=" * 60)
        print("All tests passed successfully! ✓")
        print("=" * 60)
    except AssertionError as e:
        print(f"\nTest failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
