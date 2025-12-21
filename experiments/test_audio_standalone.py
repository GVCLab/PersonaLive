#!/usr/bin/env python3
"""
Standalone test script for audio extraction and merging functionality.
This test doesn't require torch or other heavy dependencies.

Usage:
    python experiments/test_audio_standalone.py
"""

import os
import sys
import tempfile
import subprocess
import shutil

import av


def has_audio_stream(video_path):
    """Check if a video file has an audio stream."""
    try:
        container = av.open(video_path)
        for stream in container.streams:
            if stream.type == "audio":
                container.close()
                return True
        container.close()
        return False
    except Exception:
        return False


def add_audio_to_video(video_path, audio_source_path, output_path=None, verbose=False):
    """
    Add audio from audio_source_path to video_path.

    The audio will be trimmed to match the video duration if it's longer.
    If the video is longer than the audio, the audio will end when it ends.
    """
    if not has_audio_stream(audio_source_path):
        if verbose:
            print(f"No audio stream found in {audio_source_path}")
        return False

    if output_path is None:
        output_path = video_path

    temp_output = None
    try:
        video_container = av.open(video_path)
        video_stream = next(s for s in video_container.streams if s.type == "video")
        video_duration = float(video_stream.duration * video_stream.time_base)
        video_container.close()

        if verbose:
            print(f"Video duration: {video_duration:.2f}s")

        output_dir = os.path.dirname(output_path) or "."
        temp_fd, temp_output = tempfile.mkstemp(suffix=".mp4", dir=output_dir)
        os.close(temp_fd)

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_source_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-t", str(video_duration),
            "-shortest",
            temp_output
        ]

        if verbose:
            print(f"Running: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            if verbose:
                print(f"ffmpeg error: {result.stderr}")
            return False

        shutil.move(temp_output, output_path)
        temp_output = None

        if verbose:
            print(f"Successfully added audio to {output_path}")
        return True

    except Exception as e:
        if verbose:
            print(f"Error adding audio: {e}")
        return False
    finally:
        if temp_output and os.path.exists(temp_output):
            os.remove(temp_output)


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

            # Verify output duration matches the video duration (not audio)
            container = av.open(output_path)
            video_stream = next(s for s in container.streams if s.type == "video")
            output_duration = float(video_stream.duration * video_stream.time_base)
            container.close()

            # Duration should be approximately 2 seconds (video duration, not 5 seconds audio)
            assert abs(output_duration - 2.0) < 0.5, f"Output duration should be ~2s, got {output_duration:.2f}s"
            print(f"  ✓ Output duration is {output_duration:.2f}s (correctly trimmed)")
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


def test_replace_in_place():
    """Test add_audio_to_video replacing the video file in place."""
    print("\nTesting in-place replacement...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create source video with audio
        source_with_audio = os.path.join(tmpdir, "source_with_audio.mp4")
        if not create_test_video_with_audio(source_with_audio, duration=3):
            print("  ⚠ Could not create source video with audio")
            return

        # Create target video without audio
        target_path = os.path.join(tmpdir, "target.mp4")
        if not create_test_video_without_audio(target_path, duration=2):
            print("  ⚠ Could not create target video")
            return

        # Verify it has no audio
        assert not has_audio_stream(target_path), "Target should not have audio initially"

        # Replace in place (output_path=None means replace the input)
        result = add_audio_to_video(target_path, source_with_audio, output_path=None, verbose=True)

        assert result == True, "Should succeed"
        assert has_audio_stream(target_path), "Target should now have audio"
        print("  ✓ Successfully replaced video file in place with audio")

    print("  All in-place replacement tests passed!")


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
        test_replace_in_place()

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
