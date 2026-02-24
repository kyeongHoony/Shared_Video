#!/usr/bin/env python3
"""
Quick smoke test for jetson_baseline.py and jetson_optimizer.py.
No pytest needed — run directly:  python smoke_test.py

Creates synthetic data in a temp directory, then exercises the full pipeline
with a mocked model. Takes ~5 seconds.
"""

import struct
import sys
import tempfile
import traceback
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image


# ── Synthetic data ─────────────────────────────────────────────────────────────

def make_dataset(root: Path, video_name="test_video", n=16, w=256, h=128):
    rng = np.random.default_rng(0)
    frame_dir = root / "training" / "clean" / video_name
    flow_dir  = root / "training" / "flow"  / video_name
    frame_dir.mkdir(parents=True)
    flow_dir.mkdir(parents=True)

    for i in range(1, n + 1):
        frame = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        Image.fromarray(frame).save(frame_dir / f"frame_{i:04d}.png")

        flow = np.zeros((h, w, 2), dtype=np.float32)
        flow[:, :, 0] = 1.5
        flow[h//4:h//2, w//4:w//2, 0] = 8.0  # high-motion region
        with open(flow_dir / f"frame_{i:04d}.flo", "wb") as f:
            f.write(struct.pack("f", 202021.25))
            f.write(struct.pack("i", w))
            f.write(struct.pack("i", h))
            f.write(flow.tobytes())


# ── Mock model / processor ─────────────────────────────────────────────────────

def mock_processor():
    p = MagicMock()
    p.apply_chat_template.return_value = "<prompt>"
    p.return_value = {
        "input_ids":      torch.zeros(1, 5, dtype=torch.long),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
    }
    p.batch_decode.return_value = ["a video of something"]
    p.decode.return_value = "a video of something"
    return p


def mock_model():
    m = MagicMock()
    m.device = torch.device("cpu")
    m.eval.return_value = m
    m.generate.return_value = torch.zeros(1, 10, dtype=torch.long)
    return m


# ── Individual checks ──────────────────────────────────────────────────────────

def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
        return True
    except Exception:
        print(f"  FAIL  {name}")
        traceback.print_exc()
        return False


# ── Tests ──────────────────────────────────────────────────────────────────────

def run(root: Path, video_name: str):
    results = []

    # 1. Read .flo file
    def test_flo():
        from jetson_optimizer import MotionVectorAnalyzer
        profile = MotionVectorAnalyzer(str(root)).get_video_motion_profile(video_name)
        assert len(profile) == 16
        assert all(v >= 0 for v in profile.values())

    # 2. Load PNG frames
    def test_frames():
        from jetson_baseline import JetsonQwenBaseline
        profiler = JetsonQwenBaseline(sintel_dir=str(root))
        frame_dir = root / "training" / "clean" / video_name
        frames, indices = profiler.get_video_frames_native(frame_dir, num_frames=8)
        assert len(frames) == 8
        assert indices[0] == 0 and indices[-1] == 15

    # 3. Frame selector stays within bounds
    def test_selector():
        from jetson_optimizer import MotionVectorAnalyzer, MotionGuidedFrameSelector, OptimizationConfig
        config = OptimizationConfig(min_frames=4, max_frames=16, base_frames=8)
        profile = MotionVectorAnalyzer(str(root)).get_video_motion_profile(video_name)
        indices, _ = MotionGuidedFrameSelector(config).select_frames_by_threshold(
            profile, 16, config.base_frames
        )
        assert 4 <= len(indices) <= 16
        assert indices == sorted(indices)

    # 4. Baseline full pipeline with mock model
    def test_baseline_pipeline():
        with patch("jetson_baseline.AutoProcessor.from_pretrained", return_value=mock_processor()), \
             patch("jetson_baseline.AutoModelForVision2Seq.from_pretrained", return_value=mock_model()), \
             patch("jetson_baseline.process_vision_info", return_value=(None, None)):
            from jetson_baseline import JetsonQwenBaseline
            p = JetsonQwenBaseline(sintel_dir=str(root))
            p.load_model()
            m = p.profile_video_pipeline(video_name, num_frames_to_sample=4)
        assert m.frames_sampled == 4
        assert m.total_pipeline_time > 0
        assert len(m.stage_times) == 9

    # 5. Optimizer full pipeline with mock model
    def test_optimizer_pipeline():
        with patch("jetson_optimizer.AutoProcessor.from_pretrained", return_value=mock_processor()), \
             patch("jetson_optimizer.AutoModelForVision2Seq.from_pretrained", return_value=mock_model()), \
             patch("jetson_optimizer.AutoConfig.from_pretrained", return_value=MagicMock()), \
             patch("jetson_optimizer.process_vision_info", return_value=(None, None)):
            from jetson_optimizer import JetsonSpatioTemporalOptimizer, OptimizationConfig
            config = OptimizationConfig(base_frames=8, min_frames=4, max_frames=16)
            opt = JetsonSpatioTemporalOptimizer(sintel_dir=str(root), config=config)
            opt.load_model()
            m = opt.optimize_video(video_name)
        assert 0 < m.total_frames_selected <= 16
        assert 0.0 <= m.temporal_reduction <= 1.0

    tests = [
        ("flo file read",             test_flo),
        ("PNG frame loading",         test_frames),
        ("frame selector bounds",     test_selector),
        ("baseline pipeline (mock)",  test_baseline_pipeline),
        ("optimizer pipeline (mock)", test_optimizer_pipeline),
    ]

    print(f"\nRunning {len(tests)} smoke tests...\n")
    for name, fn in tests:
        results.append(check(name, fn))

    passed = sum(results)
    total  = len(results)
    print(f"\n{passed}/{total} passed")
    return passed == total


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_dataset(root)
        ok = run(root, "test_video")
    sys.exit(0 if ok else 1)
