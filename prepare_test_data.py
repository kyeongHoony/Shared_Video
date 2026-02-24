#!/usr/bin/env python3
"""
Minimal MPI-Sintel-format synthetic test data generator.

Generates PNG frames + .flo optical flow files that mimic the MPI-Sintel
directory structure, so the Jetson scripts can be tested without downloading
the full ~4GB dataset.

Output structure:
  ./MPI-Sintel/training/clean/test_video/frame_0001.png  ...
  ./MPI-Sintel/training/flow/test_video/frame_0001.flo   ...

Usage:
  python prepare_test_data.py [--out-dir ./MPI-Sintel] [--video-name test_video]
                              [--num-frames 16] [--width 1024] [--height 436]

After running, set:
  export SINTEL_DIR=./MPI-Sintel
  # then update video_name in jetson_baseline.py / jetson_optimizer.py to "test_video"
"""

import argparse
import struct
import numpy as np
from pathlib import Path
from PIL import Image


def write_flo_file(path: Path, flow: np.ndarray):
    """
    Write a .flo file in MPI-Sintel binary format.
    Format: [magic float32] [width int32] [height int32] [flow float32 H*W*2]
    """
    h, w = flow.shape[:2]
    with open(path, "wb") as f:
        f.write(struct.pack("f", 202021.25))   # magic number
        f.write(struct.pack("i", w))
        f.write(struct.pack("i", h))
        f.write(flow.astype(np.float32).tobytes())


def create_synthetic_video(
    out_dir: Path,
    video_name: str,
    num_frames: int,
    width: int,
    height: int,
):
    frame_dir = out_dir / "training" / "clean" / video_name
    flow_dir  = out_dir / "training" / "flow"  / video_name
    frame_dir.mkdir(parents=True, exist_ok=True)
    flow_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed=42)

    for i in range(1, num_frames + 1):
        # ── PNG frame ──────────────────────────────────────────────────────
        # Gradient background + random noise blobs to give non-trivial content
        base = np.zeros((height, width, 3), dtype=np.uint8)

        # Horizontal gradient
        base[:, :, 0] = np.linspace(30, 200, width, dtype=np.uint8)[np.newaxis, :]
        base[:, :, 1] = np.linspace(50, 150, height, dtype=np.uint8)[:, np.newaxis]
        base[:, :, 2] = (i * 15) % 256

        # Add a moving rectangle to simulate object motion across frames
        rx = int((i / num_frames) * (width - 100))
        ry = int((i / num_frames) * (height - 60))
        base[ry : ry + 60, rx : rx + 100, :] = [200, 100, 50]

        # Light noise
        noise = rng.integers(0, 20, size=(height, width, 3), dtype=np.uint8)
        frame = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        frame_name = f"frame_{i:04d}.png"
        Image.fromarray(frame).save(frame_dir / frame_name)

        # ── .flo optical flow ──────────────────────────────────────────────
        # Simulate a slow rightward + downward pan with object motion region
        flow = np.zeros((height, width, 2), dtype=np.float32)

        # Background: slow global motion
        flow[:, :, 0] = 1.5   # u (horizontal)
        flow[:, :, 1] = 0.5   # v (vertical)

        # Object region: faster motion
        flow[ry : ry + 60, rx : rx + 100, 0] = 8.0 + rng.random() * 4
        flow[ry : ry + 60, rx : rx + 100, 1] = 3.0 + rng.random() * 2

        # Add mild noise
        flow += rng.normal(0, 0.3, size=flow.shape).astype(np.float32)

        flo_name = f"frame_{i:04d}.flo"
        write_flo_file(flow_dir / flo_name, flow)

    print(f"Created {num_frames} frames  → {frame_dir}")
    print(f"Created {num_frames} .flo files → {flow_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic MPI-Sintel-format test data")
    parser.add_argument("--out-dir",    default="./MPI-Sintel", help="Root output directory")
    parser.add_argument("--video-name", default="test_video",   help="Video sequence name")
    parser.add_argument("--num-frames", type=int, default=16,   help="Number of frames to generate")
    parser.add_argument("--width",      type=int, default=1024, help="Frame width in pixels")
    parser.add_argument("--height",     type=int, default=436,  help="Frame height in pixels")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    print(f"Generating synthetic dataset:")
    print(f"  Directory : {out_dir}")
    print(f"  Video name: {args.video_name}")
    print(f"  Frames    : {args.num_frames}  ({args.width}x{args.height} px)")

    create_synthetic_video(out_dir, args.video_name, args.num_frames, args.width, args.height)

    print("\nDone. Next steps:")
    print(f"  export SINTEL_DIR={out_dir.resolve()}")
    print(f"  export OUTPUT_DIR=./results")
    print(f"  # Edit video_name in jetson_baseline.py / jetson_optimizer.py:")
    print(f'  #   video_name = "{args.video_name}"')
    print(f"  python jetson_baseline.py")
    print(f"  python jetson_optimizer.py")


if __name__ == "__main__":
    main()
