#!/usr/bin/env python3
"""
Test suite for jetson_baseline.py and jetson_optimizer.py

Two test categories:
  1. Unit tests  -- no model required, fast (data processing logic only)
  2. Mock tests  -- full pipeline flow using MagicMock instead of real model

Usage:
  pip install pytest
  pytest test_jetson.py -v                # run all
  pytest test_jetson.py -v -k "unit"      # unit tests only
  pytest test_jetson.py -v -k "Mocked"    # mock pipeline tests only
"""

import struct
import json
import pytest
import numpy as np
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image

# ── Common fixture ────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_dataset(tmp_path):
    """
    Generate a minimal MPI-Sintel-format dataset under tmp_path.
    Same structure as prepare_test_data.py.
    """
    video_name = "test_video"
    num_frames = 16
    width, height = 256, 128      # smaller than real (1024x436) for speed

    frame_dir = tmp_path / "training" / "clean" / video_name
    flow_dir  = tmp_path / "training" / "flow"  / video_name
    frame_dir.mkdir(parents=True)
    flow_dir.mkdir(parents=True)

    rng = np.random.default_rng(seed=0)

    for i in range(1, num_frames + 1):
        # PNG frame with a moving rectangle to produce non-trivial motion
        frame = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        rx = int((i / num_frames) * (width - 40))
        ry = int((i / num_frames) * (height - 20))
        frame[ry:ry+20, rx:rx+40] = [200, 100, 50]
        Image.fromarray(frame).save(frame_dir / f"frame_{i:04d}.png")

        # Optical flow: slow background pan + fast object region
        flow = np.zeros((height, width, 2), dtype=np.float32)
        flow[:, :, 0] = 1.5   # horizontal
        flow[:, :, 1] = 0.5   # vertical
        flow[ry:ry+20, rx:rx+40, 0] = 8.0 + rng.random()
        flow[ry:ry+20, rx:rx+40, 1] = 3.0 + rng.random()
        _write_flo(flow_dir / f"frame_{i:04d}.flo", flow)

    return tmp_path, video_name, num_frames, width, height


def _write_flo(path: Path, flow: np.ndarray):
    """Write optical flow in MPI-Sintel binary .flo format."""
    h, w = flow.shape[:2]
    with open(path, "wb") as f:
        f.write(struct.pack("f", 202021.25))
        f.write(struct.pack("i", w))
        f.write(struct.pack("i", h))
        f.write(flow.astype(np.float32).tobytes())


# ═══════════════════════════════════════════════════════════════════════════════
# Unit Tests -- MotionVectorAnalyzer
# ═══════════════════════════════════════════════════════════════════════════════

class TestMotionVectorAnalyzerUnit:

    def _make_analyzer(self, sintel_dir=""):
        from jetson_optimizer import MotionVectorAnalyzer
        return MotionVectorAnalyzer(sintel_dir)

    def test_read_flo_roundtrip(self, tmp_path):
        """Written .flo file should deserialize to the original flow array."""
        flow = np.array([[[1.0, 2.0], [3.0, 4.0]],
                         [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
        flo_path = tmp_path / "test.flo"
        _write_flo(flo_path, flow)

        result = self._make_analyzer().read_flo_file(str(flo_path))

        assert result.shape == (2, 2, 2)
        np.testing.assert_array_almost_equal(result, flow)

    def test_invalid_flo_tag_raises(self, tmp_path):
        """File with wrong magic number should raise ValueError."""
        bad_path = tmp_path / "bad.flo"
        with open(bad_path, "wb") as f:
            f.write(struct.pack("f", 999.0))   # wrong magic number
            f.write(struct.pack("i", 2))
            f.write(struct.pack("i", 2))
            f.write(np.zeros((2, 2, 2), dtype=np.float32).tobytes())

        with pytest.raises(ValueError, match="Invalid .flo file tag"):
            self._make_analyzer().read_flo_file(str(bad_path))

    def test_compute_motion_magnitude_zero_flow(self):
        """Zero flow should produce magnitude of 0.0."""
        flow = np.zeros((10, 10, 2), dtype=np.float32)
        assert self._make_analyzer().compute_motion_magnitude(flow) == 0.0

    def test_compute_motion_magnitude_known_value(self):
        """u=3, v=4 gives magnitude=5 for all pixels regardless of percentile."""
        flow = np.full((10, 10, 2), fill_value=0.0, dtype=np.float32)
        flow[:, :, 0] = 3.0
        flow[:, :, 1] = 4.0
        assert pytest.approx(self._make_analyzer().compute_motion_magnitude(flow), abs=1e-4) == 5.0

    def test_complexity_score_range(self):
        """Complexity score must always be in [0, 1]."""
        from jetson_optimizer import MotionVectorAnalyzer
        analyzer = MotionVectorAnalyzer("")
        for _ in range(20):
            profile = {i: float(v) for i, v in enumerate(np.random.rand(20) * 200)}
            score = analyzer.compute_complexity_score(profile)
            assert 0.0 <= score <= 1.0, f"score out of range: {score}"

    def test_get_video_motion_profile_count(self, synthetic_dataset):
        """Number of entries in motion profile should equal number of .flo files."""
        sintel_dir, video_name, num_frames, *_ = synthetic_dataset
        from jetson_optimizer import MotionVectorAnalyzer
        profile = MotionVectorAnalyzer(str(sintel_dir)).get_video_motion_profile(video_name)
        assert len(profile) == num_frames


# ═══════════════════════════════════════════════════════════════════════════════
# Unit Tests -- MotionGuidedFrameSelector
# ═══════════════════════════════════════════════════════════════════════════════

class TestMotionGuidedFrameSelectorUnit:

    def _make_selector(self):
        from jetson_optimizer import OptimizationConfig, MotionGuidedFrameSelector
        config = OptimizationConfig(min_frames=6, max_frames=24, base_frames=16)
        return MotionGuidedFrameSelector(config), config

    def test_selected_count_within_config_bounds(self):
        """Selected frame count must be within [min_frames, max_frames]."""
        selector, config = self._make_selector()
        profile = {i: float(i * 5) for i in range(50)}
        indices, _ = selector.select_frames_by_threshold(profile, 50, config.base_frames)
        assert config.min_frames <= len(indices) <= config.max_frames

    def test_output_indices_are_sorted(self):
        """Returned index list must be sorted in ascending order."""
        selector, config = self._make_selector()
        profile = {i: float(np.random.rand()) for i in range(30)}
        indices, _ = selector.select_frames_by_threshold(profile, 30, config.base_frames)
        assert indices == sorted(indices)

    def test_no_duplicate_indices(self):
        """No frame should be selected more than once."""
        selector, config = self._make_selector()
        profile = {i: 1.0 for i in range(20)}
        indices, _ = selector.select_frames_by_threshold(profile, 20, config.base_frames)
        assert len(indices) == len(set(indices))

    def test_all_selected_within_profile_range(self):
        """Every selected index must be a key in the original motion profile."""
        selector, config = self._make_selector()
        profile = {i: float(i) for i in range(20)}
        indices, _ = selector.select_frames_by_threshold(profile, 20, config.base_frames)
        assert all(idx in profile for idx in indices)

    def test_adaptive_count_does_not_exceed_profile_size(self):
        """Adaptive frame count must not exceed the total number of frames."""
        selector, config = self._make_selector()
        small_profile = {i: float(i) for i in range(4)}
        indices, _ = selector.select_frames_by_threshold(
            small_profile, 4, config.base_frames, adaptive_count=True
        )
        assert len(indices) <= 4


# ═══════════════════════════════════════════════════════════════════════════════
# Unit Tests -- SpatialPatchAnalyzer
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpatialPatchAnalyzerUnit:

    def _make_analyzer(self, patch_grid=(4, 4), min_patches=4):
        from jetson_optimizer import OptimizationConfig, SpatialPatchAnalyzer
        config = OptimizationConfig(
            patch_grid=patch_grid,
            spatial_motion_threshold=2.0,
            patch_motion_ratio=0.05,
            min_patches_per_frame=min_patches,
        )
        return SpatialPatchAnalyzer(config), config

    def test_output_mask_shape_matches_grid(self):
        """encoding_mask shape must equal patch_grid (rows, cols)."""
        analyzer, _ = self._make_analyzer(patch_grid=(4, 4))
        flow = np.zeros((128, 256, 2), dtype=np.float32)
        result = analyzer.analyze_frame_patches(flow, frame_idx=0)
        assert result["encoding_mask"].shape == (4, 4)

    def test_min_patches_enforced_on_zero_flow(self):
        """Even with zero motion, at least min_patches_per_frame patches must be encoded."""
        min_patches = 4
        analyzer, _ = self._make_analyzer(min_patches=min_patches)
        flow = np.zeros((128, 256, 2), dtype=np.float32)
        result = analyzer.analyze_frame_patches(flow, frame_idx=0)
        assert result["encoding_mask"].sum() >= min_patches

    def test_high_motion_region_gets_encoded(self):
        """Patch with large motion magnitude must have encode=True."""
        analyzer, _ = self._make_analyzer(patch_grid=(2, 2))
        flow = np.zeros((128, 256, 2), dtype=np.float32)
        flow[:64, :128, 0] = 50.0   # strong motion in top-left patch
        result = analyzer.analyze_frame_patches(flow, frame_idx=0)
        assert result["encoding_mask"][0, 0] == True

    def test_spatial_savings_between_0_and_1(self):
        """spatial_savings must always be in [0, 1]."""
        analyzer, _ = self._make_analyzer()
        flow = np.random.rand(128, 256, 2).astype(np.float32) * 10
        result = analyzer.analyze_frame_patches(flow, frame_idx=0)
        assert 0.0 <= result["spatial_savings"] <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Unit Tests -- JetsonQwenBaseline (model-free)
# ═══════════════════════════════════════════════════════════════════════════════

class TestJetsonQwenBaselineUnit:

    def test_get_video_frames_linspace_count(self, synthetic_dataset):
        """get_video_frames_native must return exactly the requested number of frames."""
        sintel_dir, video_name, num_frames, *_ = synthetic_dataset
        from jetson_baseline import JetsonQwenBaseline
        profiler = JetsonQwenBaseline(sintel_dir=str(sintel_dir))
        frame_dir = sintel_dir / "training" / "clean" / video_name

        for req in [4, 8, 16]:
            frames, indices = profiler.get_video_frames_native(frame_dir, num_frames=req)
            assert len(frames) == req, f"requested {req}, got {len(frames)}"
            assert len(indices) == req

    def test_get_video_frames_indices_in_range(self, synthetic_dataset):
        """First index must be 0 and last index must be total_frames - 1."""
        sintel_dir, video_name, num_frames, *_ = synthetic_dataset
        from jetson_baseline import JetsonQwenBaseline
        profiler = JetsonQwenBaseline(sintel_dir=str(sintel_dir))
        frame_dir = sintel_dir / "training" / "clean" / video_name

        _, indices = profiler.get_video_frames_native(frame_dir, num_frames=8)
        assert indices[0] == 0
        assert indices[-1] == num_frames - 1

    def test_save_results_creates_json(self, tmp_path, synthetic_dataset):
        """save_results must produce a well-formed JSON file."""
        from jetson_baseline import JetsonQwenBaseline, DetailedLatencyMetrics
        profiler = JetsonQwenBaseline()

        metrics = DetailedLatencyMetrics(
            frames_sampled=8,
            frames_loaded=16,
            generation_time=1.0,
            total_pipeline_time=5.0,
            total_preprocessing_time=4.0,
            total_inference_time=1.0,
            peak_memory_mb=5000.0,
            system_memory_used_mb=20000.0,
            gpu_memory_allocated_mb=10000.0,
            gpu_memory_reserved_mb=12000.0,
            output_tokens=15,
            generated_text="A person walking.",
            stage_times={"1_frame_loading_sampling": 0.5, "8_model_generation": 1.0},
        )
        profiler.save_results(metrics, "test_8frames", str(tmp_path))

        out_file = tmp_path / "test_8frames_jetson_baseline.json"
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert data["platform"] == "Jetson AGX Orin"
        assert data["configuration"]["dtype"] == "float16"
        assert data["output"]["tokens"] == 15


# ═══════════════════════════════════════════════════════════════════════════════
# Mock Pipeline Tests -- full flow without a real model
# ═══════════════════════════════════════════════════════════════════════════════

def _make_mock_processor(input_ids_len=5):
    """Processor mock covering apply_chat_template / __call__ / batch_decode / decode."""
    mock_proc = MagicMock()
    mock_proc.apply_chat_template.return_value = "<user>describe video</user>"
    mock_proc.return_value = {
        "input_ids": torch.zeros(1, input_ids_len, dtype=torch.long),
        "attention_mask": torch.ones(1, input_ids_len, dtype=torch.long),
    }
    mock_proc.batch_decode.return_value = ["A walking scene near a wall."]
    mock_proc.decode.return_value = "A walking scene near a wall."
    return mock_proc


def _make_mock_model(input_ids_len=5):
    """Model mock covering device / eval / generate."""
    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")
    mock_model.eval.return_value = mock_model
    mock_model.generate.return_value = torch.zeros(
        1, input_ids_len + 5, dtype=torch.long
    )
    return mock_model


class TestBaselinePipelineMocked:

    def test_full_pipeline_returns_metrics(self, synthetic_dataset):
        """profile_video_pipeline should complete and return DetailedLatencyMetrics."""
        sintel_dir, video_name, *_ = synthetic_dataset

        mock_proc  = _make_mock_processor(input_ids_len=5)
        mock_model = _make_mock_model(input_ids_len=5)

        with patch("jetson_baseline.AutoProcessor.from_pretrained", return_value=mock_proc), \
             patch("jetson_baseline.AutoModelForVision2Seq.from_pretrained", return_value=mock_model), \
             patch("jetson_baseline.process_vision_info", return_value=(None, None)):

            from jetson_baseline import JetsonQwenBaseline
            profiler = JetsonQwenBaseline(sintel_dir=str(sintel_dir))
            profiler.load_model()
            metrics = profiler.profile_video_pipeline(video_name, num_frames_to_sample=4)

        assert metrics.frames_sampled == 4
        assert metrics.total_pipeline_time > 0
        assert metrics.output_tokens >= 0
        assert isinstance(metrics.generated_text, str)

    def test_pipeline_stage_times_all_recorded(self, synthetic_dataset):
        """All 9 pipeline stages must be present in stage_times."""
        sintel_dir, video_name, *_ = synthetic_dataset

        with patch("jetson_baseline.AutoProcessor.from_pretrained", return_value=_make_mock_processor()), \
             patch("jetson_baseline.AutoModelForVision2Seq.from_pretrained", return_value=_make_mock_model()), \
             patch("jetson_baseline.process_vision_info", return_value=(None, None)):

            from jetson_baseline import JetsonQwenBaseline
            profiler = JetsonQwenBaseline(sintel_dir=str(sintel_dir))
            profiler.load_model()
            metrics = profiler.profile_video_pipeline(video_name, num_frames_to_sample=4)

        expected_stages = [
            "1_frame_loading_sampling", "2_image_conversion", "3_message_preparation",
            "4_vision_processing", "5_text_tokenization", "6_processor_encoding",
            "7_tensor_transfer", "8_model_generation", "9_output_decoding",
        ]
        for stage in expected_stages:
            assert stage in metrics.stage_times, f"missing stage: {stage}"


class TestOptimizerPipelineMocked:

    def test_optimize_video_returns_metrics(self, synthetic_dataset):
        """optimize_video should complete and return OptimizationMetrics with valid ranges."""
        sintel_dir, video_name, num_frames, *_ = synthetic_dataset

        with patch("jetson_optimizer.AutoProcessor.from_pretrained", return_value=_make_mock_processor()), \
             patch("jetson_optimizer.AutoModelForVision2Seq.from_pretrained", return_value=_make_mock_model()), \
             patch("jetson_optimizer.AutoConfig.from_pretrained", return_value=MagicMock()), \
             patch("jetson_optimizer.process_vision_info", return_value=(None, None)):

            from jetson_optimizer import JetsonSpatioTemporalOptimizer, OptimizationConfig
            config = OptimizationConfig(base_frames=8, min_frames=4, max_frames=16)
            optimizer = JetsonSpatioTemporalOptimizer(sintel_dir=str(sintel_dir), config=config)
            optimizer.load_model()
            metrics = optimizer.optimize_video(video_name)

        assert metrics.total_frames_original == num_frames
        assert 0 < metrics.total_frames_selected <= num_frames
        assert 0.0 <= metrics.temporal_reduction <= 1.0
        assert 0.0 <= metrics.spatial_reduction <= 1.0
        assert metrics.total_time > 0

    def test_save_optimization_summary_creates_json(self, tmp_path, synthetic_dataset):
        """save_optimization_summary must produce a well-formed JSON file."""
        sintel_dir, video_name, *_ = synthetic_dataset

        with patch("jetson_optimizer.AutoProcessor.from_pretrained", return_value=_make_mock_processor()), \
             patch("jetson_optimizer.AutoModelForVision2Seq.from_pretrained", return_value=_make_mock_model()), \
             patch("jetson_optimizer.AutoConfig.from_pretrained", return_value=MagicMock()), \
             patch("jetson_optimizer.process_vision_info", return_value=(None, None)):

            from jetson_optimizer import JetsonSpatioTemporalOptimizer, OptimizationConfig
            config = OptimizationConfig(base_frames=8, min_frames=4, max_frames=16)
            optimizer = JetsonSpatioTemporalOptimizer(sintel_dir=str(sintel_dir), config=config)
            optimizer.load_model()
            metrics = optimizer.optimize_video(video_name)
            optimizer.save_optimization_summary(video_name, metrics, str(tmp_path))

        out_file = tmp_path / f"{video_name}_jetson_optimization_summary.json"
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert data["platform"] == "Jetson AGX Orin"
        assert "performance_metrics" in data
        assert "quality_metrics" in data
        assert "resource_metrics" in data
