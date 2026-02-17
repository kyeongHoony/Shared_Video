#!/usr/bin/env python3
"""
Native Qwen2.5-VL Baseline Implementation with Detailed Latency Breakdown
Follows Qwen's exact video processing approach - using np.linspace to sample frames uniformly
"""

import os
import sys
import numpy as np
import cv2
import json
import time
import torch
import gc
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import logging
from PIL import Image

# Add paths
sys.path.append('/data/Fall25/multimodalRAG/VLM/Qwen2.5-VL')
sys.path.append('/data/Fall25/multimodalRAG/VLM/Qwen2.5-VL/qwen-vl-utils/src')

from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig
from qwen_vl_utils import process_vision_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DetailedLatencyMetrics:
    """Detailed latency breakdown for each pipeline stage"""
    
    # Frame loading stage
    frame_loading_time: float = 0.0
    frames_loaded: int = 0
    
    # Frame sampling stage  
    frame_sampling_time: float = 0.0
    frames_sampled: int = 0
    
    # Image preprocessing stage
    image_conversion_time: float = 0.0  # Converting to PIL
    vision_processing_time: float = 0.0  # process_vision_info
    
    # Tokenization stage
    text_tokenization_time: float = 0.0
    processor_encoding_time: float = 0.0  # Full processor encoding
    
    # Model transfer stage
    tensor_transfer_time: float = 0.0  # Moving to GPU
    
    # Inference stage
    model_forward_time: float = 0.0
    generation_time: float = 0.0
    
    # Decoding stage
    output_decoding_time: float = 0.0
    
    # Total metrics
    total_preprocessing_time: float = 0.0
    total_inference_time: float = 0.0
    total_pipeline_time: float = 0.0
    
    # Memory metrics
    peak_memory_mb: float = 0.0
    gpu_memory_allocated_mb: float = 0.0
    gpu_memory_reserved_mb: float = 0.0
    
    # Output info
    output_tokens: int = 0
    generated_text: str = ""
    
    # Additional detailed metrics
    stage_times: Dict[str, float] = field(default_factory=dict)

class QwenNativeBaseline:
    """Native Qwen2.5-VL implementation following exact Qwen approach"""
    
    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 sintel_dir: str = "/data/Fall25/multimodalRAG/VLM/Qwen2.5-VL/MPI-Sintel"):
        
        self.model_path = model_path
        self.sintel_dir = sintel_dir
        
        # Model components
        self.processor = None
        self.model = None
        
        # Following Qwen's default configuration
        self.max_frames = 2048  # Qwen's default max_frames
        self.sample_fps = 2  # Qwen's default sample_fps (not used for pre-loaded frames)
        self.total_pixels = 20480 * 32 * 32  # Qwen's default total_pixels
        self.min_pixels = 64 * 32 * 32  # Qwen's default min_pixels
        
    def load_model(self):
        """Load Qwen2.5-VL model with timing"""
        logger.info("Loading Qwen2.5-VL model (native configuration)...")
        
        # Clear GPU cache before loading
        torch.cuda.empty_cache()
        gc.collect()
        
        load_start = time.time()
        
        # Load processor
        processor_start = time.time()
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        processor_time = time.time() - processor_start
        logger.info(f"Processor loaded in {processor_time:.3f}s")
        
        # Load model with minimal memory
        model_start = time.time()
        
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,  # Using bfloat16 for better stability
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation="eager"  # Use eager attention
        )
        
        self.model.eval()
        model_time = time.time() - model_start
        logger.info(f"Model loaded in {model_time:.3f}s")
        
        total_load_time = time.time() - load_start
        logger.info(f"Total model loading time: {total_load_time:.3f}s")
        
    def get_video_frames_native(self, video_path: Path, num_frames: int = 64) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Load video frames following Qwen's exact approach
        Uses np.linspace to uniformly sample frames (Qwen's native behavior)
        
        Args:
            video_path: Path to video directory
            num_frames: Number of frames to sample uniformly
        
        Returns:
            frames: List of sampled frames
            indices: Array of selected frame indices
        """
        frame_files = sorted(video_path.glob("*.png"))
        total_frames = len(frame_files)
        
        # Qwen's exact sampling: np.linspace for uniform sampling
        indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            frame = cv2.imread(str(frame_files[idx]))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        return frames, indices
    
    def profile_video_pipeline(self, video_name: str, num_frames_to_sample: int = 16,
                              query: str = "Describe this video in detail.") -> DetailedLatencyMetrics:
        """
        Profile the complete pipeline with detailed latency breakdown
        Following Qwen's exact implementation
        
        Args:
            video_name: Name of video directory
            num_frames_to_sample: Number of frames to sample using np.linspace (Qwen default behavior)
            query: Text prompt
        """
        
        pipeline_start = time.time()
        metrics = DetailedLatencyMetrics()
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        # ===== STAGE 1: Frame Loading and Sampling (Combined as in Qwen) =====
        logger.info(f"\n[Stage 1] Loading and sampling {num_frames_to_sample} frames from {video_name}...")
        frame_load_start = time.time()
        
        frame_dir = Path(self.sintel_dir) / "training" / "clean" / video_name
        
        # Load all frames first to get total count
        all_frame_files = sorted(frame_dir.glob("*.png"))
        total_frames_available = len(all_frame_files)
        logger.info(f"  Total frames available: {total_frames_available}")
        
        # Use Qwen's native sampling with np.linspace
        sampled_frames, selected_indices = self.get_video_frames_native(frame_dir, num_frames=num_frames_to_sample)
        
        metrics.frame_loading_time = time.time() - frame_load_start
        metrics.frames_loaded = total_frames_available  # Total available
        metrics.frames_sampled = len(sampled_frames)  # Actually used
        logger.info(f"  Loaded and sampled {len(sampled_frames)} frames in {metrics.frame_loading_time:.3f}s")
        logger.info(f"  Selected indices: {selected_indices.tolist()}")
        
        # ===== STAGE 2: Image Conversion to PIL =====
        logger.info(f"\n[Stage 2] Converting to PIL images...")
        conversion_start = time.time()
        
        pil_frames = [Image.fromarray(frame) for frame in sampled_frames]
        
        metrics.image_conversion_time = time.time() - conversion_start
        logger.info(f"  Converted to PIL in {metrics.image_conversion_time:.3f}s")
        
        # ===== STAGE 3: Prepare Messages (Qwen format) =====
        logger.info(f"\n[Stage 3] Preparing messages in Qwen format...")
        message_start = time.time()
        
        # Following Qwen's exact message format for video
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "video", 
                    "video": pil_frames,
                    "total_pixels": self.total_pixels,
                    "min_pixels": self.min_pixels,
                    "max_frames": self.max_frames,
                    "sample_fps": self.sample_fps  # This is metadata when frames are pre-loaded
                },
                {"type": "text", "text": query}
            ]
        }]
        
        message_time = time.time() - message_start
        logger.info(f"  Message preparation in {message_time:.3f}s")
        
        # ===== STAGE 4: Vision Processing =====
        logger.info(f"\n[Stage 4] Processing vision information...")
        vision_start = time.time()
        
        # Process vision info with metadata
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Video inputs are handled directly by process_vision_info
        
        metrics.vision_processing_time = time.time() - vision_start
        logger.info(f"  Vision processing in {metrics.vision_processing_time:.3f}s")
        
        # ===== STAGE 5: Text Tokenization =====
        logger.info(f"\n[Stage 5] Text tokenization...")
        text_start = time.time()
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        metrics.text_tokenization_time = time.time() - text_start
        logger.info(f"  Text tokenized in {metrics.text_tokenization_time:.3f}s")
        
        # ===== STAGE 6: Full Processor Encoding =====
        logger.info(f"\n[Stage 6] Full processor encoding...")
        encoding_start = time.time()
        
        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        metrics.processor_encoding_time = time.time() - encoding_start
        logger.info(f"  Processor encoding in {metrics.processor_encoding_time:.3f}s")
        
        # ===== STAGE 7: Tensor Transfer to GPU =====
        logger.info(f"\n[Stage 7] Transferring tensors to GPU...")
        transfer_start = time.time()
        
        inputs = inputs.to('cuda')
        
        metrics.tensor_transfer_time = time.time() - transfer_start
        logger.info(f"  Tensor transfer in {metrics.tensor_transfer_time:.3f}s")
        
        # ===== STAGE 8: Model Inference =====
        logger.info(f"\n[Stage 8] Model inference...")
        
        # Check GPU memory before inference
        if torch.cuda.is_available():
            metrics.gpu_memory_allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
            metrics.gpu_memory_reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024
            logger.info(f"  GPU Memory - Allocated: {metrics.gpu_memory_allocated_mb:.1f}MB, Reserved: {metrics.gpu_memory_reserved_mb:.1f}MB")
        
        inference_start = time.time()
        
        with torch.no_grad():
            try:
                # Generate output with limited tokens
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=30,  # Reduced for memory
                    do_sample=False,
                    temperature=0.7
                )
                
                metrics.generation_time = time.time() - inference_start
                
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"CUDA OOM: {e}")
                # Try with CPU fallback
                logger.info("Falling back to CPU inference...")
                inputs = {k: v.cpu() for k, v in inputs.items()}
                self.model = self.model.cpu()
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=False
                )
                
                metrics.generation_time = time.time() - inference_start
        
        logger.info(f"  Model generation in {metrics.generation_time:.3f}s")
        
        # ===== STAGE 9: Output Decoding =====
        logger.info(f"\n[Stage 9] Decoding output...")
        decode_start = time.time()
        
        generated_ids = [outputs[0][len(inputs.input_ids[0]):]]
        output_text = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )
        
        metrics.output_decoding_time = time.time() - decode_start
        metrics.generated_text = output_text[0] if output_text else ""
        metrics.output_tokens = len(generated_ids[0])
        logger.info(f"  Output decoded in {metrics.output_decoding_time:.3f}s")
        
        # ===== Calculate Total Metrics =====
        metrics.total_preprocessing_time = (
            metrics.frame_loading_time + 
            metrics.image_conversion_time +
            message_time +
            metrics.vision_processing_time +
            metrics.text_tokenization_time +
            metrics.processor_encoding_time +
            metrics.tensor_transfer_time
        )
        
        metrics.total_inference_time = metrics.generation_time
        metrics.total_pipeline_time = time.time() - pipeline_start
        
        # Memory metrics
        import psutil
        metrics.peak_memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Store detailed stage times
        metrics.stage_times = {
            "1_frame_loading_sampling": metrics.frame_loading_time,
            "2_image_conversion": metrics.image_conversion_time,
            "3_message_preparation": message_time,
            "4_vision_processing": metrics.vision_processing_time,
            "5_text_tokenization": metrics.text_tokenization_time,
            "6_processor_encoding": metrics.processor_encoding_time,
            "7_tensor_transfer": metrics.tensor_transfer_time,
            "8_model_generation": metrics.generation_time,
            "9_output_decoding": metrics.output_decoding_time
        }
        
        return metrics
    
    def print_detailed_report(self, metrics: DetailedLatencyMetrics, video_name: str):
        """Print formatted detailed latency report"""
        
        print(f"\n{'='*80}")
        print(f"QWEN NATIVE BASELINE - DETAILED LATENCY BREAKDOWN")
        print(f"Video: {video_name}")
        print(f"{'='*80}")
        
        print(f"\n📊 INPUT STATISTICS:")
        print(f"  • Frames available: {metrics.frames_loaded}")
        print(f"  • Frames sampled (np.linspace): {metrics.frames_sampled}")
        print(f"  • Sampling method: Uniform (Qwen native)")
        
        print(f"\n⏱️  STAGE-WISE LATENCY BREAKDOWN:")
        print(f"  {'Stage':<35} {'Time (s)':<12} {'Percentage':<12}")
        print(f"  {'-'*59}")
        
        total_time = metrics.total_pipeline_time
        
        for stage_name, stage_time in metrics.stage_times.items():
            percentage = (stage_time / total_time) * 100
            stage_display = stage_name.replace('_', ' ').title()
            print(f"  {stage_display:<35} {stage_time:>8.3f}s    {percentage:>6.1f}%")
        
        print(f"  {'-'*59}")
        print(f"  {'TOTAL':<35} {total_time:>8.3f}s    {100.0:>6.1f}%")
        
        print(f"\n📈 SUMMARY METRICS:")
        print(f"  • Total Preprocessing: {metrics.total_preprocessing_time:.3f}s ({(metrics.total_preprocessing_time/total_time)*100:.1f}%)")
        print(f"  • Total Inference: {metrics.total_inference_time:.3f}s ({(metrics.total_inference_time/total_time)*100:.1f}%)")
        print(f"  • Total Pipeline: {metrics.total_pipeline_time:.3f}s")
        
        print(f"\n💾 MEMORY METRICS:")
        print(f"  • Peak System Memory: {metrics.peak_memory_mb:.1f} MB")
        print(f"  • GPU Memory Allocated: {metrics.gpu_memory_allocated_mb:.1f} MB")
        print(f"  • GPU Memory Reserved: {metrics.gpu_memory_reserved_mb:.1f} MB")
        
        print(f"\n📝 OUTPUT:")
        print(f"  • Tokens generated: {metrics.output_tokens}")
        print(f"  • Text: {metrics.generated_text[:200]}...")
        
        print(f"\n{'='*80}\n")
    
    def save_detailed_results(self, metrics: DetailedLatencyMetrics, video_name: str,
                            output_dir: str):
        """Save detailed results to JSON"""
        
        total_time = metrics.total_pipeline_time
        
        results = {
            "video_name": video_name,
            "configuration": {
                "model": self.model_path,
                "frames_available": metrics.frames_loaded,
                "frames_sampled": metrics.frames_sampled,
                "sampling_method": "np.linspace (Qwen native)",
                "max_frames": self.max_frames,
                "sample_fps": self.sample_fps,
                "total_pixels": self.total_pixels,
                "min_pixels": self.min_pixels
            },
            "latency_breakdown": {
                stage_name.replace('_', ' ').title(): {
                    "time_s": float(stage_time),
                    "percentage": float((stage_time / total_time) * 100)
                }
                for stage_name, stage_time in metrics.stage_times.items()
            },
            "summary": {
                "total_preprocessing_s": float(metrics.total_preprocessing_time),
                "total_inference_s": float(metrics.total_inference_time),
                "total_pipeline_s": float(metrics.total_pipeline_time),
                "preprocessing_percentage": float((metrics.total_preprocessing_time / total_time) * 100),
                "inference_percentage": float((metrics.total_inference_time / total_time) * 100)
            },
            "memory": {
                "peak_system_mb": float(metrics.peak_memory_mb),
                "gpu_allocated_mb": float(metrics.gpu_memory_allocated_mb),
                "gpu_reserved_mb": float(metrics.gpu_memory_reserved_mb)
            },
            "output": {
                "tokens": metrics.output_tokens,
                "text": metrics.generated_text
            }
        }
        
        output_path = Path(output_dir) / f"{video_name}_qwen_native_latency.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved results to {output_path}")
        return results

if __name__ == "__main__":
    # Initialize profiler
    profiler = QwenNativeBaseline()
    
    # Load model
    profiler.load_model()
    
    # Profile ambush_5 (50 frames available)
    video_name = "ambush_5"
    
    # Test with different sampling configurations
    sampling_configs = [
        8,   # Sample 8 frames from 50
        16,  # Sample 16 frames from 50 (common default)
        32,  # Sample 32 frames from 50
    ]
    
    for num_frames in sampling_configs:
        logger.info(f"\n🎬 Profiling {video_name} - Sampling {num_frames} frames from 50...")
        
        try:
            # Clear cache between runs
            torch.cuda.empty_cache()
            gc.collect()
            
            # Run profiling
            metrics = profiler.profile_video_pipeline(
                video_name, 
                num_frames_to_sample=num_frames
            )
            
            # Print detailed report
            profiler.print_detailed_report(metrics, video_name)
            
            # Save results
            output_dir = "/data/Fall25/multimodalRAG/VLM/Qwen2.5-VL/motionVector/spatio-temporal-working/baseline"
            profiler.save_detailed_results(metrics, f"{video_name}_{num_frames}frames", output_dir)
            
        except Exception as e:
            logger.error(f"Error profiling with {num_frames} frames: {e}")
            continue
    
    print("✅ Profiling complete!")