#!/usr/bin/env python3
"""
Jetson AGX Orin (Unified Memory) Version of Spatio-Temporal Optimizer
Based on spatio_temporal_optimizer.py — adapted for Jetson ARM64 + unified memory.

Changes from original:
  - sys.path uses QWEN_BASE env var instead of hard-coded /data/Fall25/...
  - sintel_dir / output_dir / video_name passed via argparse (not hard-coded)
  - device_map="auto" → device_map="cuda:0"  (unified memory: single device)
  - max_memory={0: "20GB"} removed  (64GB unified: no artificial limit needed)
  - OOM handling added around model.generate()
  - psutil.virtual_memory() logging added for system-wide RAM monitoring
"""

import os
import sys
import numpy as np
import cv2
import json
import time
import psutil
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import argparse
import struct
import logging
from PIL import Image

# ── Path setup ───────────────────────────────────────────────────────────────
_qwen_base = os.environ.get("QWEN_BASE", "")
if _qwen_base:
    sys.path.insert(0, _qwen_base)
    sys.path.insert(0, os.path.join(_qwen_base, "qwen-vl-utils", "src"))

from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig
from qwen_vl_utils import process_vision_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Config / Metrics dataclasses (unchanged from original) ───────────────────
@dataclass
class OptimizationConfig:
    temporal_motion_threshold: float = 0.3
    min_frames: int = 6
    max_frames: int = 24
    base_frames: int = 16
    motion_percentile: float = 70.0
    patch_grid: Tuple[int, int] = (4, 4)
    spatial_motion_threshold: float = 2.0
    patch_motion_ratio: float = 0.05
    min_patches_per_frame: int = 4
    enable_adaptive_grid: bool = True
    enable_cross_optimization: bool = True
    quality_preservation_mode: str = "balanced"


@dataclass
class OptimizationMetrics:
    total_frames_original: int = 0
    total_frames_selected: int = 0
    total_patches_original: int = 0
    total_patches_encoded: int = 0
    preprocessing_time: float = 0.0
    inference_time: float = 0.0
    total_time: float = 0.0
    peak_memory_mb: float = 0.0
    system_memory_used_mb: float = 0.0   # Added: unified memory monitor
    memory_reduction_mb: float = 0.0
    shannon_entropy_retention: float = 0.0
    motion_coverage: float = 0.0
    information_retention: float = 0.0
    temporal_reduction: float = 0.0
    spatial_reduction: float = 0.0
    combined_reduction: float = 0.0
    speedup_factor: float = 0.0
    generated_text: str = ""
    output_tokens: int = 0


# ── Motion analysis (unchanged from original) ────────────────────────────────
class MotionVectorAnalyzer:
    def __init__(self, sintel_dir: str):
        self.sintel_dir = Path(sintel_dir)

    def read_flo_file(self, flo_path: str) -> np.ndarray:
        with open(flo_path, "rb") as f:
            tag = struct.unpack("f", f.read(4))[0]
            if tag != 202021.25:
                raise ValueError(f"Invalid .flo file tag: {tag}")
            width = struct.unpack("i", f.read(4))[0]
            height = struct.unpack("i", f.read(4))[0]
            flow_data = np.frombuffer(f.read(), dtype=np.float32)
            flow = flow_data.reshape((height, width, 2))
        return flow

    def compute_motion_magnitude(self, flow: np.ndarray, percentile: float = 95) -> float:
        magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
        return np.percentile(magnitude, percentile)

    def get_video_motion_profile(self, video_name: str) -> Dict[int, float]:
        flow_dir = self.sintel_dir / "training" / "flow" / video_name
        flow_files = sorted(flow_dir.glob("*.flo"))
        motion_profile = {}
        for i, flow_file in enumerate(flow_files):
            flow = self.read_flo_file(str(flow_file))
            motion_profile[i] = self.compute_motion_magnitude(flow)
        return motion_profile

    def compute_complexity_score(self, motion_profile: Dict[int, float]) -> float:
        motion_scores = np.array(list(motion_profile.values()))
        mean_motion_norm = min(np.mean(motion_scores) / 100.0, 1.0)
        variance_norm = min(np.var(motion_scores) / 5000.0, 1.0)
        range_norm = min(np.ptp(motion_scores) / 150.0, 1.0)
        high_motion_threshold = np.percentile(motion_scores, 75)
        high_motion_ratio = np.sum(motion_scores > high_motion_threshold) / len(motion_scores)
        return min(0.4 * mean_motion_norm + 0.3 * variance_norm + 0.2 * range_norm + 0.1 * high_motion_ratio, 1.0)


class MotionGuidedFrameSelector:
    def __init__(self, config: OptimizationConfig):
        self.config = config

    def adaptive_frame_count(self, motion_profile: Dict[int, float]) -> int:
        analyzer = MotionVectorAnalyzer("")
        score = analyzer.compute_complexity_score(motion_profile)
        if score > 0.8:
            return min(self.config.base_frames + 6, self.config.max_frames)
        elif score > 0.6:
            return self.config.base_frames + 3
        elif score > 0.4:
            return self.config.base_frames
        elif score > 0.2:
            return max(self.config.base_frames - 3, self.config.min_frames)
        else:
            return max(self.config.base_frames - 6, self.config.min_frames)

    def select_frames_by_threshold(
        self,
        motion_profile: Dict[int, float],
        total_frames: int,
        target_frame_count: int,
        adaptive_count: bool = True,
    ) -> Tuple[List[int], int]:
        actual_frame_count = self.adaptive_frame_count(motion_profile) if adaptive_count else target_frame_count
        motion_scores = list(motion_profile.values())
        threshold = np.percentile(motion_scores, self.config.motion_percentile)
        importance_weights = [s if s >= threshold else 0.1 for s in motion_scores]
        probabilities = np.array(importance_weights)
        prob_sum = np.sum(probabilities)
        probabilities = probabilities / prob_sum if prob_sum > 0 else np.ones(len(importance_weights)) / len(importance_weights)
        probabilities = np.clip(probabilities, 0, 1)
        probabilities /= np.sum(probabilities)
        try:
            selected_indices = np.random.choice(
                list(motion_profile.keys()),
                size=min(actual_frame_count, len(motion_profile)),
                replace=False,
                p=probabilities,
            )
        except ValueError:
            selected_indices = np.random.choice(
                list(motion_profile.keys()),
                size=min(actual_frame_count, len(motion_profile)),
                replace=False,
            )
        return sorted(selected_indices), actual_frame_count


class SpatialPatchAnalyzer:
    def __init__(self, config: OptimizationConfig):
        self.config = config

    def analyze_frame_patches(self, flow: np.ndarray, frame_idx: int) -> Dict:
        h, w = flow.shape[:2]
        rows, cols = self.config.patch_grid
        patch_h, patch_w = h // rows, w // cols
        patch_decisions = np.zeros((rows, cols), dtype=bool)
        patch_details = []
        for i in range(rows):
            for j in range(cols):
                y_start, y_end = i * patch_h, (i + 1) * patch_h
                x_start, x_end = j * patch_w, (j + 1) * patch_w
                patch_flow = flow[y_start:y_end, x_start:x_end]
                magnitude = np.sqrt(patch_flow[:, :, 0] ** 2 + patch_flow[:, :, 1] ** 2)
                motion_ratio = np.sum(magnitude > self.config.spatial_motion_threshold) / magnitude.size
                mean_motion = np.mean(magnitude)
                max_motion = np.max(magnitude)
                requires_encoding = (
                    motion_ratio > self.config.patch_motion_ratio
                    or mean_motion > self.config.spatial_motion_threshold
                    or max_motion > self.config.spatial_motion_threshold * 2
                )
                patch_decisions[i, j] = requires_encoding
                patch_details.append({
                    "position": (i, j),
                    "motion_ratio": motion_ratio,
                    "mean_motion": mean_motion,
                    "max_motion": max_motion,
                    "encode": requires_encoding,
                })
        if patch_decisions.sum() < self.config.min_patches_per_frame:
            motion_scores = [p["mean_motion"] for p in patch_details]
            for idx in np.argsort(motion_scores)[::-1]:
                if patch_decisions.sum() >= self.config.min_patches_per_frame:
                    break
                i, j = patch_details[idx]["position"]
                patch_decisions[i, j] = True
        return {
            "encoding_mask": patch_decisions,
            "patch_details": patch_details,
            "spatial_savings": 1.0 - (patch_decisions.sum() / patch_decisions.size),
        }


# ── Main optimizer ────────────────────────────────────────────────────────────
class JetsonSpatioTemporalOptimizer:
    """
    Jetson-adapted version of SpatioTemporalOptimizer.
    Uses unified memory-aware settings for Jetson AGX Orin 64GB.
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        sintel_dir: str = "./MPI-Sintel",
        config: Optional[OptimizationConfig] = None,
    ):
        self.model_path = model_path
        self.sintel_dir = sintel_dir
        self.config = config or OptimizationConfig()

        self.motion_analyzer = MotionVectorAnalyzer(sintel_dir)
        self.frame_selector = MotionGuidedFrameSelector(self.config)
        self.spatial_analyzer = SpatialPatchAnalyzer(self.config)

        self.processor = None
        self.model = None

    def _log_memory(self, tag: str = ""):
        vm = psutil.virtual_memory()
        used_gb = vm.used / 1024 ** 3
        total_gb = vm.total / 1024 ** 3
        logger.info(f"[MEM{' ' + tag if tag else ''}] System RAM: {used_gb:.1f}/{total_gb:.1f} GB")

    def load_model(self):
        """Load Qwen2.5-VL with Jetson-appropriate settings."""
        logger.info("Loading Qwen2.5-VL model (Jetson unified-memory config)...")
        self._log_memory("before load")

        self.processor = AutoProcessor.from_pretrained(self.model_path)

        config = AutoConfig.from_pretrained(self.model_path)
        config._attn_implementation = "eager"

        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            config=config,
            torch_dtype=torch.float16,      # float16: native Jetson support
            device_map="cuda:0",            # unified memory — single CUDA device
            low_cpu_mem_usage=True,
            # max_memory removed: 64GB unified memory, no artificial cap needed
            attn_implementation="eager",    # no FlashAttention on Jetson
        )
        self.model.eval()
        logger.info("Model loaded successfully")
        self._log_memory("after load")

    def calculate_entropy(self, frames: List[np.ndarray]) -> float:
        entropies = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if len(frame.shape) == 3 else frame
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_norm = hist.flatten() / hist.sum()
            entropies.append(-np.sum(hist_norm * np.log2(hist_norm + 1e-10)))
        return np.mean(entropies)

    def calculate_motion_coverage(self, motion_profile: Dict[int, float], selected_frames: List[int]) -> float:
        total_motion = sum(motion_profile.values())
        selected_motion = sum(motion_profile[i] for i in selected_frames)
        return selected_motion / total_motion if total_motion > 0 else 0.0

    def optimize_video(self, video_name: str, query: str = "Describe this video.") -> OptimizationMetrics:
        """Perform complete spatio-temporal optimization on a video."""
        start_time = time.time()
        start_memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        metrics = OptimizationMetrics()

        # Load frames
        frame_dir = Path(self.sintel_dir) / "training" / "clean" / video_name
        frame_files = sorted(frame_dir.glob("*.png"))
        original_frames = []
        for frame_file in frame_files:
            frame = cv2.imread(str(frame_file))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frames.append(frame)
        metrics.total_frames_original = len(original_frames)

        # Step 1: Temporal optimization
        logger.info(f"Analyzing motion for {video_name}...")
        motion_profile = self.motion_analyzer.get_video_motion_profile(video_name)
        selected_frame_indices, _ = self.frame_selector.select_frames_by_threshold(
            motion_profile, len(original_frames), self.config.base_frames, adaptive_count=True
        )
        selected_frames = [original_frames[i] for i in selected_frame_indices]
        metrics.total_frames_selected = len(selected_frames)
        metrics.temporal_reduction = 1.0 - (len(selected_frames) / len(original_frames))

        # Step 2: Spatial optimization
        logger.info("Performing spatial analysis...")
        flow_dir = Path(self.sintel_dir) / "training" / "flow" / video_name
        flow_files = sorted(flow_dir.glob("*.flo"))
        total_patches = len(selected_frames) * self.config.patch_grid[0] * self.config.patch_grid[1]
        encoded_patches = 0
        for frame_idx in selected_frame_indices:
            if frame_idx < len(flow_files):
                flow = self.motion_analyzer.read_flo_file(str(flow_files[frame_idx]))
                spatial_analysis = self.spatial_analyzer.analyze_frame_patches(flow, frame_idx)
                encoded_patches += spatial_analysis["encoding_mask"].sum()

        metrics.total_patches_original = total_patches
        metrics.total_patches_encoded = encoded_patches
        metrics.spatial_reduction = 1.0 - (encoded_patches / total_patches) if total_patches > 0 else 0.0
        metrics.combined_reduction = 1.0 - (
            (metrics.total_frames_selected * encoded_patches)
            / (metrics.total_frames_original * metrics.total_frames_original * self.config.patch_grid[0] * self.config.patch_grid[1])
        )
        metrics.speedup_factor = (
            1.0 / (1.0 - metrics.combined_reduction) if metrics.combined_reduction < 1.0 else float("inf")
        )
        metrics.preprocessing_time = time.time() - start_time

        # Step 3: Model inference
        if self.model is not None:
            logger.info("Running optimized inference...")
            self._log_memory("before inference")
            inference_start = time.time()

            pil_frames = [Image.fromarray(frame) for frame in selected_frames]
            messages = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": pil_frames},
                    {"type": "text", "text": query},
                ],
            }]
            image_inputs, video_inputs = process_vision_info(messages)
            text_inputs = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=text_inputs,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = {
                k: v.to(self.model.device) if torch.is_tensor(v) else v
                for k, v in inputs.items()
            }

            try:
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=50)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                logger.error(f"OOM / Runtime error during inference: {e}")
                logger.error(
                    "On Jetson unified memory, CPU fallback does NOT free memory.\n"
                    "Suggestions:\n"
                    "  1. Reduce config.base_frames / min_frames\n"
                    "  2. Use a smaller model (e.g. Qwen2.5-VL-3B-Instruct)\n"
                    "  3. Close other processes consuming RAM"
                )
                raise

            generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            metrics.generated_text = generated_text
            metrics.output_tokens = len(outputs[0])
            metrics.inference_time = time.time() - inference_start
            self._log_memory("after inference")

        # Quality metrics
        metrics.shannon_entropy_retention = (
            self.calculate_entropy(selected_frames) / self.calculate_entropy(original_frames)
        )
        metrics.motion_coverage = self.calculate_motion_coverage(motion_profile, selected_frame_indices)
        metrics.information_retention = metrics.motion_coverage

        # Memory metrics (unified memory: track system RAM)
        vm = psutil.virtual_memory()
        metrics.peak_memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        metrics.system_memory_used_mb = vm.used / 1024 / 1024
        metrics.memory_reduction_mb = start_memory_mb - metrics.peak_memory_mb
        metrics.total_time = time.time() - start_time

        logger.info(
            f"Optimization complete: {metrics.combined_reduction:.1%} reduction, "
            f"{metrics.speedup_factor:.2f}x speedup"
        )
        return metrics

    def save_optimization_summary(self, video_name: str, metrics: OptimizationMetrics, output_dir: str):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        summary = {
            "video_name": video_name,
            "platform": "Jetson AGX Orin",
            "configuration": {
                "temporal_threshold": self.config.temporal_motion_threshold,
                "patch_grid": self.config.patch_grid,
                "spatial_threshold": self.config.spatial_motion_threshold,
                "patch_ratio": self.config.patch_motion_ratio,
            },
            "performance_metrics": {
                "temporal_reduction": float(metrics.temporal_reduction),
                "spatial_reduction": float(metrics.spatial_reduction),
                "combined_reduction": float(metrics.combined_reduction),
                "speedup_factor": float(metrics.speedup_factor),
                "preprocessing_time_sec": float(metrics.preprocessing_time),
                "inference_time_sec": float(metrics.inference_time),
                "total_time_sec": float(metrics.total_time),
            },
            "quality_metrics": {
                "shannon_entropy_retention": float(metrics.shannon_entropy_retention),
                "motion_coverage": float(metrics.motion_coverage),
                "information_retention": float(metrics.information_retention),
            },
            "resource_metrics": {
                "frames_original": int(metrics.total_frames_original),
                "frames_selected": int(metrics.total_frames_selected),
                "patches_original": int(metrics.total_patches_original),
                "patches_encoded": int(metrics.total_patches_encoded),
                "process_rss_mb": float(metrics.peak_memory_mb),
                "system_ram_used_mb": float(metrics.system_memory_used_mb),
            },
        }
        out_path = Path(output_dir) / f"{video_name}_jetson_optimization_summary.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary to {out_path}")
        return summary


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jetson Qwen2.5-VL spatio-temporal optimizer")
    parser.add_argument("--model-path",  default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--sintel-dir",  default="./MPI-Sintel",
                        help="Root directory of MPI-Sintel dataset")
    parser.add_argument("--output-dir",  default="./results",
                        help="Directory to write JSON result files")
    parser.add_argument("--video-name",  default="alley_1",
                        help="Video sequence name (subdirectory under training/clean/)")
    args = parser.parse_args()

    config = OptimizationConfig(
        temporal_motion_threshold=0.3,
        patch_grid=(4, 4),
        spatial_motion_threshold=2.0,
        patch_motion_ratio=0.05,
    )

    optimizer = JetsonSpatioTemporalOptimizer(
        model_path=args.model_path,
        sintel_dir=args.sintel_dir,
        config=config,
    )
    optimizer.load_model()

    metrics = optimizer.optimize_video(args.video_name)

    print(f"\nOptimization Results for {args.video_name}:")
    print(f"  Temporal reduction:  {metrics.temporal_reduction:.1%}")
    print(f"  Spatial reduction:   {metrics.spatial_reduction:.1%}")
    print(f"  Combined reduction:  {metrics.combined_reduction:.1%}")
    print(f"  Speedup factor:      {metrics.speedup_factor:.2f}x")
    print(f"  Quality retention:   {metrics.shannon_entropy_retention:.1%}")
    print(f"  System RAM used:     {metrics.system_memory_used_mb:.0f} MB")

    optimizer.save_optimization_summary(args.video_name, metrics, args.output_dir)
    print("Done.")
