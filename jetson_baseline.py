#!/usr/bin/env python3
"""
Jetson AGX Orin (Unified Memory) Version of Qwen Native Baseline
Based on qwen_native_baseline.py — adapted for Jetson ARM64 + unified memory.

Changes from original:
  - bfloat16 → float16  (Jetson has no native bfloat16 hardware support)
  - device_map="auto" → device_map="cuda:0"  (unified memory: single device)
  - sys.path uses QWEN_BASE env var instead of hard-coded /data/Fall25/...
  - sintel_dir / output_dir / video_name passed via argparse (not hard-coded)
  - Tensor transfer uses self.model.device (not hard-coded 'cuda')
  - torch.cuda.* calls guarded with is_available()
  - CPU fallback removed (meaningless on unified memory), OOM → clear guidance
  - Added psutil.virtual_memory() for system-wide RAM monitoring
"""

import os
import sys
import numpy as np
import cv2
import json
import time
import torch
import gc
import psutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import argparse
import logging
from PIL import Image

# ── Path setup ───────────────────────────────────────────────────────────────
# Set QWEN_BASE to your qwen2.5-vl repo root if qwen_vl_utils is not pip-installed.
# e.g.  export QWEN_BASE=/home/user/Qwen2.5-VL
# If qwen-vl-utils is installed via pip, QWEN_BASE is not needed.
_qwen_base = os.environ.get("QWEN_BASE", "")
if _qwen_base:
    sys.path.insert(0, _qwen_base)
    sys.path.insert(0, os.path.join(_qwen_base, "qwen-vl-utils", "src"))

from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig
from qwen_vl_utils import process_vision_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Data classes (unchanged from original) ───────────────────────────────────
@dataclass
class DetailedLatencyMetrics:
    """Detailed latency breakdown for each pipeline stage"""

    frame_loading_time: float = 0.0
    frames_loaded: int = 0
    frame_sampling_time: float = 0.0
    frames_sampled: int = 0
    image_conversion_time: float = 0.0
    vision_processing_time: float = 0.0
    text_tokenization_time: float = 0.0
    processor_encoding_time: float = 0.0
    tensor_transfer_time: float = 0.0
    model_forward_time: float = 0.0
    generation_time: float = 0.0
    output_decoding_time: float = 0.0
    total_preprocessing_time: float = 0.0
    total_inference_time: float = 0.0
    total_pipeline_time: float = 0.0
    peak_memory_mb: float = 0.0
    gpu_memory_allocated_mb: float = 0.0
    gpu_memory_reserved_mb: float = 0.0
    system_memory_used_mb: float = 0.0   # Added: unified memory monitor
    output_tokens: int = 0
    generated_text: str = ""
    stage_times: Dict[str, float] = field(default_factory=dict)


# ── Main class ────────────────────────────────────────────────────────────────
class JetsonQwenBaseline:
    """
    Jetson-adapted version of QwenNativeBaseline.
    Uses unified memory-aware settings for Jetson AGX Orin 64GB.
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        sintel_dir: str = "./MPI-Sintel",
    ):
        self.model_path = model_path
        self.sintel_dir = sintel_dir

        self.processor = None
        self.model = None

        # Qwen default vision config
        self.max_frames = 2048
        self.sample_fps = 2
        self.total_pixels = 20480 * 32 * 32
        self.min_pixels = 64 * 32 * 32

    # ── Memory helpers ────────────────────────────────────────────────────────
    def _clear_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _log_memory(self, tag: str = ""):
        vm = psutil.virtual_memory()
        used_gb = vm.used / 1024 ** 3
        total_gb = vm.total / 1024 ** 3
        msg = f"[MEM{' ' + tag if tag else ''}] System RAM: {used_gb:.1f}/{total_gb:.1f} GB"
        if torch.cuda.is_available():
            cuda_gb = torch.cuda.memory_allocated() / 1024 ** 3
            msg += f" | CUDA alloc: {cuda_gb:.1f} GB (same pool on Jetson)"
        logger.info(msg)

    # ── Model loading ─────────────────────────────────────────────────────────
    def load_model(self):
        """Load Qwen2.5-VL with Jetson-appropriate settings."""
        logger.info("Loading Qwen2.5-VL model (Jetson unified-memory config)...")
        self._clear_cache()
        self._log_memory("before load")

        load_start = time.time()

        proc_start = time.time()
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        logger.info(f"Processor loaded in {time.time() - proc_start:.3f}s")

        model_start = time.time()
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,      # float16: native Jetson support
            device_map="cuda:0",            # unified memory — single CUDA device
            low_cpu_mem_usage=True,
            attn_implementation="eager",    # no FlashAttention on Jetson
        )
        self.model.eval()
        logger.info(f"Model loaded in {time.time() - model_start:.3f}s")
        logger.info(f"Total load time: {time.time() - load_start:.3f}s")
        self._log_memory("after load")

    # ── Frame loading ─────────────────────────────────────────────────────────
    def get_video_frames_native(
        self, video_path: Path, num_frames: int = 64
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Uniform frame sampling using np.linspace (Qwen native behavior)."""
        frame_files = sorted(video_path.glob("*.png"))
        total_frames = len(frame_files)
        indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
        frames = []
        for idx in indices:
            frame = cv2.imread(str(frame_files[idx]))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        return frames, indices

    # ── Pipeline profiling ────────────────────────────────────────────────────
    def profile_video_pipeline(
        self,
        video_name: str,
        num_frames_to_sample: int = 16,
        query: str = "Describe this video in detail.",
    ) -> DetailedLatencyMetrics:
        """Profile the complete pipeline with stage-wise latency breakdown."""

        pipeline_start = time.time()
        metrics = DetailedLatencyMetrics()

        self._clear_cache()

        # STAGE 1: Frame loading & sampling
        logger.info(f"\n[Stage 1] Loading and sampling {num_frames_to_sample} frames from {video_name}...")
        frame_load_start = time.time()

        frame_dir = Path(self.sintel_dir) / "training" / "clean" / video_name
        all_frame_files = sorted(frame_dir.glob("*.png"))
        total_frames_available = len(all_frame_files)
        logger.info(f"  Total frames available: {total_frames_available}")

        sampled_frames, selected_indices = self.get_video_frames_native(
            frame_dir, num_frames=num_frames_to_sample
        )

        metrics.frame_loading_time = time.time() - frame_load_start
        metrics.frames_loaded = total_frames_available
        metrics.frames_sampled = len(sampled_frames)
        logger.info(f"  Sampled {len(sampled_frames)} frames in {metrics.frame_loading_time:.3f}s")

        # STAGE 2: PIL conversion
        logger.info("\n[Stage 2] Converting to PIL images...")
        conversion_start = time.time()
        pil_frames = [Image.fromarray(frame) for frame in sampled_frames]
        metrics.image_conversion_time = time.time() - conversion_start
        logger.info(f"  PIL conversion in {metrics.image_conversion_time:.3f}s")

        # STAGE 3: Message preparation
        logger.info("\n[Stage 3] Preparing messages...")
        message_start = time.time()
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": pil_frames,
                    "total_pixels": self.total_pixels,
                    "min_pixels": self.min_pixels,
                    "max_frames": self.max_frames,
                    "sample_fps": self.sample_fps,
                },
                {"type": "text", "text": query},
            ],
        }]
        message_time = time.time() - message_start
        logger.info(f"  Message prep in {message_time:.3f}s")

        # STAGE 4: Vision processing
        logger.info("\n[Stage 4] Processing vision info...")
        vision_start = time.time()
        image_inputs, video_inputs = process_vision_info(messages)
        metrics.vision_processing_time = time.time() - vision_start
        logger.info(f"  Vision processing in {metrics.vision_processing_time:.3f}s")

        # STAGE 5: Text tokenization
        logger.info("\n[Stage 5] Text tokenization...")
        text_start = time.time()
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        metrics.text_tokenization_time = time.time() - text_start
        logger.info(f"  Tokenized in {metrics.text_tokenization_time:.3f}s")

        # STAGE 6: Full processor encoding
        logger.info("\n[Stage 6] Full processor encoding...")
        encoding_start = time.time()
        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        metrics.processor_encoding_time = time.time() - encoding_start
        logger.info(f"  Encoding in {metrics.processor_encoding_time:.3f}s")

        # STAGE 7: Tensor transfer to CUDA
        logger.info("\n[Stage 7] Transferring tensors to device...")
        transfer_start = time.time()
        inputs = {
            k: v.to(self.model.device) if torch.is_tensor(v) else v
            for k, v in inputs.items()
        }
        metrics.tensor_transfer_time = time.time() - transfer_start
        logger.info(f"  Transfer in {metrics.tensor_transfer_time:.3f}s")

        # STAGE 8: Model inference
        logger.info("\n[Stage 8] Model inference...")
        if torch.cuda.is_available():
            metrics.gpu_memory_allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
            metrics.gpu_memory_reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024
            logger.info(
                f"  CUDA alloc: {metrics.gpu_memory_allocated_mb:.1f} MB  "
                f"reserved: {metrics.gpu_memory_reserved_mb:.1f} MB"
            )

        inference_start = time.time()
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=False,
                )
            metrics.generation_time = time.time() - inference_start

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            logger.error(f"OOM / Runtime error during inference: {e}")
            logger.error(
                "On Jetson unified memory, CPU fallback does NOT free memory.\n"
                "Suggestions:\n"
                "  1. Reduce num_frames_to_sample (try 4 or 8)\n"
                "  2. Use a smaller model (e.g. Qwen2.5-VL-3B-Instruct)\n"
                "  3. Close other processes consuming RAM"
            )
            raise

        logger.info(f"  Generation in {metrics.generation_time:.3f}s")

        # STAGE 9: Output decoding
        logger.info("\n[Stage 9] Decoding output...")
        decode_start = time.time()
        generated_ids = [outputs[0][len(inputs["input_ids"][0]):]]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        metrics.output_decoding_time = time.time() - decode_start
        metrics.generated_text = output_text[0] if output_text else ""
        metrics.output_tokens = len(generated_ids[0])
        logger.info(f"  Decoded in {metrics.output_decoding_time:.3f}s")

        # Totals
        metrics.total_preprocessing_time = (
            metrics.frame_loading_time
            + metrics.image_conversion_time
            + message_time
            + metrics.vision_processing_time
            + metrics.text_tokenization_time
            + metrics.processor_encoding_time
            + metrics.tensor_transfer_time
        )
        metrics.total_inference_time = metrics.generation_time
        metrics.total_pipeline_time = time.time() - pipeline_start

        # Memory (unified memory: track system RAM)
        vm = psutil.virtual_memory()
        metrics.peak_memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        metrics.system_memory_used_mb = vm.used / 1024 / 1024

        metrics.stage_times = {
            "1_frame_loading_sampling": metrics.frame_loading_time,
            "2_image_conversion": metrics.image_conversion_time,
            "3_message_preparation": message_time,
            "4_vision_processing": metrics.vision_processing_time,
            "5_text_tokenization": metrics.text_tokenization_time,
            "6_processor_encoding": metrics.processor_encoding_time,
            "7_tensor_transfer": metrics.tensor_transfer_time,
            "8_model_generation": metrics.generation_time,
            "9_output_decoding": metrics.output_decoding_time,
        }

        return metrics

    # ── Reporting ─────────────────────────────────────────────────────────────
    def print_detailed_report(self, metrics: DetailedLatencyMetrics, video_name: str):
        print(f"\n{'='*80}")
        print("JETSON QWEN BASELINE - DETAILED LATENCY BREAKDOWN")
        print(f"Video: {video_name}")
        print(f"{'='*80}")

        print(f"\nINPUT STATISTICS:")
        print(f"  Frames available: {metrics.frames_loaded}")
        print(f"  Frames sampled (np.linspace): {metrics.frames_sampled}")

        print(f"\nSTAGE-WISE LATENCY:")
        print(f"  {'Stage':<35} {'Time (s)':<12} {'%':<8}")
        print(f"  {'-'*55}")
        total_time = metrics.total_pipeline_time
        for stage_name, stage_time in metrics.stage_times.items():
            pct = (stage_time / total_time) * 100
            print(f"  {stage_name.replace('_',' ').title():<35} {stage_time:>8.3f}s  {pct:>6.1f}%")
        print(f"  {'-'*55}")
        print(f"  {'TOTAL':<35} {total_time:>8.3f}s  {100.0:>6.1f}%")

        print(f"\nMEMORY (Unified):")
        print(f"  Process RSS:        {metrics.peak_memory_mb:.1f} MB")
        print(f"  System RAM used:    {metrics.system_memory_used_mb:.1f} MB")
        print(f"  CUDA allocated:     {metrics.gpu_memory_allocated_mb:.1f} MB")
        print(f"  CUDA reserved:      {metrics.gpu_memory_reserved_mb:.1f} MB")

        print(f"\nOUTPUT:")
        print(f"  Tokens: {metrics.output_tokens}")
        print(f"  Text:   {metrics.generated_text[:200]}")
        print(f"\n{'='*80}\n")

    def save_results(self, metrics: DetailedLatencyMetrics, video_name: str, output_dir: str):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        total_time = metrics.total_pipeline_time
        results = {
            "video_name": video_name,
            "platform": "Jetson AGX Orin",
            "configuration": {
                "model": self.model_path,
                "dtype": "float16",
                "device_map": "cuda:0",
                "frames_sampled": metrics.frames_sampled,
                "sampling_method": "np.linspace",
            },
            "latency_breakdown": {
                name.replace("_", " ").title(): {
                    "time_s": float(t),
                    "percentage": float((t / total_time) * 100),
                }
                for name, t in metrics.stage_times.items()
            },
            "summary": {
                "total_preprocessing_s": float(metrics.total_preprocessing_time),
                "total_inference_s": float(metrics.total_inference_time),
                "total_pipeline_s": float(total_time),
            },
            "memory": {
                "process_rss_mb": float(metrics.peak_memory_mb),
                "system_ram_used_mb": float(metrics.system_memory_used_mb),
                "cuda_allocated_mb": float(metrics.gpu_memory_allocated_mb),
                "cuda_reserved_mb": float(metrics.gpu_memory_reserved_mb),
            },
            "output": {
                "tokens": metrics.output_tokens,
                "text": metrics.generated_text,
            },
        }
        out_path = Path(output_dir) / f"{video_name}_jetson_baseline.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {out_path}")
        return results


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jetson Qwen2.5-VL baseline profiler")
    parser.add_argument("--model-path",  default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--sintel-dir",  default="./MPI-Sintel",
                        help="Root directory of MPI-Sintel dataset")
    parser.add_argument("--output-dir",  default="./results",
                        help="Directory to write JSON result files")
    parser.add_argument("--video-name",  default="ambush_5",
                        help="Video sequence name (subdirectory under training/clean/)")
    parser.add_argument("--num-frames",  type=int, nargs="+", default=[8, 16, 32],
                        help="Frame counts to profile, e.g. --num-frames 8 16 32")
    args = parser.parse_args()

    profiler = JetsonQwenBaseline(model_path=args.model_path, sintel_dir=args.sintel_dir)
    profiler.load_model()

    for num_frames in args.num_frames:
        logger.info(f"\nProfiling {args.video_name} — {num_frames} frames...")
        try:
            profiler._clear_cache()
            metrics = profiler.profile_video_pipeline(args.video_name, num_frames_to_sample=num_frames)
            profiler.print_detailed_report(metrics, args.video_name)
            profiler.save_results(metrics, f"{args.video_name}_{num_frames}frames", args.output_dir)
        except Exception as e:
            logger.error(f"Failed with {num_frames} frames: {e}")
            continue

    print("Profiling complete.")
