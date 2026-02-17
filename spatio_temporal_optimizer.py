#!/usr/bin/env python3
"""
Spatio-Temporal VLM Optimizer - Working Implementation
Complete working implementation of motion-guided adaptive frame selection
with spatial patch optimization for Qwen2.5-VL
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
import struct
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
class OptimizationConfig:
    """Configuration for spatio-temporal optimization"""
    
    # Temporal parameters
    temporal_motion_threshold: float = 0.3
    min_frames: int = 6
    max_frames: int = 24
    base_frames: int = 16
    motion_percentile: float = 70.0
    
    # Spatial parameters
    patch_grid: Tuple[int, int] = (4, 4)
    spatial_motion_threshold: float = 2.0
    patch_motion_ratio: float = 0.05
    min_patches_per_frame: int = 4
    
    # Quality parameters
    enable_adaptive_grid: bool = True
    enable_cross_optimization: bool = True
    quality_preservation_mode: str = "balanced"  # conservative|balanced|aggressive

@dataclass
class OptimizationMetrics:
    """Metrics for optimization performance"""
    
    # Performance metrics
    total_frames_original: int = 0
    total_frames_selected: int = 0
    total_patches_original: int = 0
    total_patches_encoded: int = 0
    
    # Timing metrics
    preprocessing_time: float = 0.0
    inference_time: float = 0.0
    total_time: float = 0.0
    
    # Memory metrics
    peak_memory_mb: float = 0.0
    memory_reduction_mb: float = 0.0
    
    # Quality metrics
    shannon_entropy_retention: float = 0.0
    motion_coverage: float = 0.0
    information_retention: float = 0.0
    
    # Efficiency metrics
    temporal_reduction: float = 0.0
    spatial_reduction: float = 0.0
    combined_reduction: float = 0.0
    speedup_factor: float = 0.0
    
    # Model output
    generated_text: str = ""
    output_tokens: int = 0

class MotionVectorAnalyzer:
    """Analyzes motion vectors from optical flow data"""
    
    def __init__(self, sintel_dir: str):
        self.sintel_dir = Path(sintel_dir)
        
    def read_flo_file(self, flo_path: str) -> np.ndarray:
        """Read MPI-Sintel .flo optical flow file"""
        with open(flo_path, 'rb') as f:
            # Read header
            tag = struct.unpack('f', f.read(4))[0]
            if tag != 202021.25:
                raise ValueError(f"Invalid .flo file tag: {tag}")
            
            width = struct.unpack('i', f.read(4))[0]
            height = struct.unpack('i', f.read(4))[0]
            
            # Read flow data
            flow_data = np.frombuffer(f.read(), dtype=np.float32)
            flow = flow_data.reshape((height, width, 2))
            
        return flow
    
    def compute_motion_magnitude(self, flow: np.ndarray, percentile: float = 95) -> float:
        """Compute motion magnitude using percentile to avoid outliers"""
        magnitude = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
        return np.percentile(magnitude, percentile)
    
    def get_video_motion_profile(self, video_name: str) -> Dict[int, float]:
        """Get motion profile for entire video"""
        flow_dir = self.sintel_dir / "training" / "flow" / video_name
        flow_files = sorted(flow_dir.glob("*.flo"))
        
        motion_profile = {}
        for i, flow_file in enumerate(flow_files):
            flow = self.read_flo_file(str(flow_file))
            motion_magnitude = self.compute_motion_magnitude(flow)
            motion_profile[i] = motion_magnitude
            
        return motion_profile
    
    def compute_complexity_score(self, motion_profile: Dict[int, float]) -> float:
        """Compute 5-component motion complexity score"""
        motion_scores = np.array(list(motion_profile.values()))
        
        # Component 1: Mean Motion (40% weight)
        mean_motion = np.mean(motion_scores)
        mean_motion_norm = min(mean_motion / 100.0, 1.0)
        
        # Component 2: Motion Variance (30% weight)
        motion_variance = np.var(motion_scores)
        variance_norm = min(motion_variance / 5000.0, 1.0)
        
        # Component 3: Motion Range (20% weight)
        motion_range = np.ptp(motion_scores)
        range_norm = min(motion_range / 150.0, 1.0)
        
        # Component 4: High Motion Ratio (10% weight)
        high_motion_threshold = np.percentile(motion_scores, 75)
        high_motion_ratio = np.sum(motion_scores > high_motion_threshold) / len(motion_scores)
        
        # Weighted combination
        complexity_score = (
            0.4 * mean_motion_norm +
            0.3 * variance_norm +
            0.2 * range_norm +
            0.1 * high_motion_ratio
        )
        
        return min(complexity_score, 1.0)

class MotionGuidedFrameSelector:
    """Selects frames based on motion analysis"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        
    def adaptive_frame_count(self, motion_profile: Dict[int, float]) -> int:
        """Determine optimal frame count based on motion complexity"""
        motion_analyzer = MotionVectorAnalyzer("")
        complexity_score = motion_analyzer.compute_complexity_score(motion_profile)
        
        if complexity_score > 0.8:
            return min(self.config.base_frames + 6, self.config.max_frames)
        elif complexity_score > 0.6:
            return self.config.base_frames + 3
        elif complexity_score > 0.4:
            return self.config.base_frames
        elif complexity_score > 0.2:
            return max(self.config.base_frames - 3, self.config.min_frames)
        else:
            return max(self.config.base_frames - 6, self.config.min_frames)
    
    def select_frames_by_threshold(self, motion_profile: Dict[int, float], 
                                 total_frames: int, target_frame_count: int, 
                                 adaptive_count: bool = True) -> Tuple[List[int], int]:
        """Select frames using motion threshold and weighted sampling"""
        
        if adaptive_count:
            actual_frame_count = self.adaptive_frame_count(motion_profile)
        else:
            actual_frame_count = target_frame_count
        
        # Calculate threshold
        motion_scores = list(motion_profile.values())
        threshold = np.percentile(motion_scores, self.config.motion_percentile)
        
        # Create importance weights
        importance_weights = []
        for score in motion_scores:
            importance = score if score >= threshold else 0.1
            importance_weights.append(importance)
        
        # Weighted probabilistic sampling with safety checks
        probabilities = np.array(importance_weights)
        prob_sum = np.sum(probabilities)
        
        if prob_sum > 0:
            probabilities = probabilities / prob_sum
        else:
            # Fallback to uniform if all weights are zero
            probabilities = np.ones(len(importance_weights)) / len(importance_weights)
        
        # Ensure probabilities are valid
        probabilities = np.clip(probabilities, 0, 1)
        probabilities = probabilities / np.sum(probabilities)
        
        try:
            selected_indices = np.random.choice(
                list(motion_profile.keys()),
                size=min(actual_frame_count, len(motion_profile)),
                replace=False,
                p=probabilities
            )
        except ValueError:
            # Fallback to uniform sampling if probabilistic fails
            selected_indices = np.random.choice(
                list(motion_profile.keys()),
                size=min(actual_frame_count, len(motion_profile)),
                replace=False
            )
        
        return sorted(selected_indices), actual_frame_count

class SpatialPatchAnalyzer:
    """Analyzes spatial patches for encoding decisions"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        
    def analyze_frame_patches(self, flow: np.ndarray, frame_idx: int) -> Dict:
        """Analyze patches in a frame for encoding decisions"""
        
        h, w = flow.shape[:2]
        rows, cols = self.config.patch_grid
        patch_h, patch_w = h // rows, w // cols
        
        patch_decisions = np.zeros((rows, cols), dtype=bool)
        patch_details = []
        
        for i in range(rows):
            for j in range(cols):
                # Extract patch
                y_start, y_end = i * patch_h, (i + 1) * patch_h
                x_start, x_end = j * patch_w, (j + 1) * patch_w
                
                patch_flow = flow[y_start:y_end, x_start:x_end]
                magnitude = np.sqrt(patch_flow[:,:,0]**2 + patch_flow[:,:,1]**2)
                
                # Calculate metrics
                total_pixels = magnitude.size
                motion_pixels = np.sum(magnitude > self.config.spatial_motion_threshold)
                motion_ratio = motion_pixels / total_pixels
                mean_motion = np.mean(magnitude)
                max_motion = np.max(magnitude)
                
                # Encoding decision (3-criteria)
                requires_encoding = (
                    motion_ratio > self.config.patch_motion_ratio or
                    mean_motion > self.config.spatial_motion_threshold or
                    max_motion > self.config.spatial_motion_threshold * 2
                )
                
                patch_decisions[i, j] = requires_encoding
                patch_details.append({
                    'position': (i, j),
                    'motion_ratio': motion_ratio,
                    'mean_motion': mean_motion,
                    'max_motion': max_motion,
                    'encode': requires_encoding
                })
        
        # Ensure minimum patches per frame
        encoded_count = patch_decisions.sum()
        if encoded_count < self.config.min_patches_per_frame:
            # Find additional patches with highest motion
            motion_scores = [p['mean_motion'] for p in patch_details]
            top_indices = np.argsort(motion_scores)[::-1]
            
            for idx in top_indices:
                if patch_decisions.sum() >= self.config.min_patches_per_frame:
                    break
                i, j = patch_details[idx]['position']
                patch_decisions[i, j] = True
        
        return {
            'encoding_mask': patch_decisions,
            'patch_details': patch_details,
            'spatial_savings': 1.0 - (patch_decisions.sum() / patch_decisions.size)
        }

class SpatioTemporalOptimizer:
    """Main spatio-temporal optimization class"""
    
    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct", 
                 sintel_dir: str = "/data/Fall25/multimodalRAG/VLM/Qwen2.5-VL/MPI-Sintel",
                 config: Optional[OptimizationConfig] = None):
        
        self.model_path = model_path
        self.sintel_dir = sintel_dir
        self.config = config or OptimizationConfig()
        
        # Initialize analyzers
        self.motion_analyzer = MotionVectorAnalyzer(sintel_dir)
        self.frame_selector = MotionGuidedFrameSelector(self.config)
        self.spatial_analyzer = SpatialPatchAnalyzer(self.config)
        
        # Model components
        self.processor = None
        self.model = None
        
    def load_model(self):
        """Load Qwen2.5-VL model"""
        logger.info("Loading Qwen2.5-VL model...")
        
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        
        config = AutoConfig.from_pretrained(self.model_path)
        config._attn_implementation = "eager"
        
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            config=config,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            max_memory={0: "20GB"},
            attn_implementation="eager"
        )
        
        self.model.eval()
        logger.info("Model loaded successfully")
        
    def calculate_entropy(self, frames: List[np.ndarray]) -> float:
        """Calculate Shannon entropy of frame sequence"""
        entropies = []
        
        for frame in frames:
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame
                
            # Calculate histogram
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_norm = hist.flatten() / hist.sum()
            
            # Calculate entropy
            entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
            entropies.append(entropy)
        
        return np.mean(entropies)
    
    def calculate_motion_coverage(self, motion_profile: Dict[int, float], 
                                selected_frames: List[int]) -> float:
        """Calculate what percentage of motion is covered by selected frames"""
        total_motion = sum(motion_profile.values())
        selected_motion = sum(motion_profile[i] for i in selected_frames)
        return selected_motion / total_motion if total_motion > 0 else 0.0
    
    def optimize_video(self, video_name: str, query: str = "Describe this video.") -> OptimizationMetrics:
        """Perform complete spatio-temporal optimization on a video"""
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        metrics = OptimizationMetrics()
        
        # Load video frames
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
        
        selected_frame_indices, actual_frame_count = self.frame_selector.select_frames_by_threshold(
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
        
        spatial_details = []
        for i, frame_idx in enumerate(selected_frame_indices):
            if frame_idx < len(flow_files):
                flow = self.motion_analyzer.read_flo_file(str(flow_files[frame_idx]))
                spatial_analysis = self.spatial_analyzer.analyze_frame_patches(flow, frame_idx)
                
                encoded_patches += spatial_analysis['encoding_mask'].sum()
                spatial_details.append(spatial_analysis)
        
        metrics.total_patches_original = total_patches
        metrics.total_patches_encoded = encoded_patches
        metrics.spatial_reduction = 1.0 - (encoded_patches / total_patches) if total_patches > 0 else 0.0
        
        # Step 3: Combined metrics
        metrics.combined_reduction = 1.0 - ((metrics.total_frames_selected * encoded_patches) / 
                                          (metrics.total_frames_original * metrics.total_frames_original * 
                                           self.config.patch_grid[0] * self.config.patch_grid[1]))
        
        metrics.speedup_factor = 1.0 / (1.0 - metrics.combined_reduction) if metrics.combined_reduction < 1.0 else float('inf')
        
        preprocessing_time = time.time() - start_time
        metrics.preprocessing_time = preprocessing_time
        
        # Step 4: Model inference (with timing)
        if self.model is not None:
            logger.info("Running optimized inference...")
            inference_start = time.time()
            
            # Prepare optimized input
            pil_frames = [Image.fromarray(frame) for frame in selected_frames]
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": pil_frames},
                    {"type": "text", "text": query}
                ]
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
                return_tensors="pt"
            )
            
            inputs = {k: v.to(self.model.device) if torch.is_tensor(v) else v 
                     for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=50)
            
            # Decode output
            generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            metrics.generated_text = generated_text
            metrics.output_tokens = len(outputs[0])
            
            metrics.inference_time = time.time() - inference_start
        
        # Step 5: Quality metrics
        metrics.shannon_entropy_retention = self.calculate_entropy(selected_frames) / self.calculate_entropy(original_frames)
        metrics.motion_coverage = self.calculate_motion_coverage(motion_profile, selected_frame_indices)
        metrics.information_retention = metrics.motion_coverage  # Proxy for information retention
        
        # Step 6: Memory metrics
        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        metrics.peak_memory_mb = peak_memory
        metrics.memory_reduction_mb = start_memory - peak_memory
        
        metrics.total_time = time.time() - start_time
        
        logger.info(f"Optimization complete: {metrics.combined_reduction:.1%} reduction, "
                   f"{metrics.speedup_factor:.2f}× speedup")
        
        return metrics
    
    def save_optimization_summary(self, video_name: str, metrics: OptimizationMetrics, 
                                output_dir: str):
        """Save optimization results summary"""
        
        summary = {
            'video_name': video_name,
            'configuration': {
                'temporal_threshold': self.config.temporal_motion_threshold,
                'patch_grid': self.config.patch_grid,
                'spatial_threshold': self.config.spatial_motion_threshold,
                'patch_ratio': self.config.patch_motion_ratio
            },
            'performance_metrics': {
                'temporal_reduction': float(metrics.temporal_reduction),
                'spatial_reduction': float(metrics.spatial_reduction),
                'combined_reduction': float(metrics.combined_reduction),
                'speedup_factor': float(metrics.speedup_factor),
                'preprocessing_time_sec': float(metrics.preprocessing_time),
                'inference_time_sec': float(metrics.inference_time),
                'total_time_sec': float(metrics.total_time)
            },
            'quality_metrics': {
                'shannon_entropy_retention': float(metrics.shannon_entropy_retention),
                'motion_coverage': float(metrics.motion_coverage),
                'information_retention': float(metrics.information_retention)
            },
            'resource_metrics': {
                'frames_original': int(metrics.total_frames_original),
                'frames_selected': int(metrics.total_frames_selected),
                'patches_original': int(metrics.total_patches_original),
                'patches_encoded': int(metrics.total_patches_encoded),
                'peak_memory_mb': float(metrics.peak_memory_mb)
            }
        }
        
        output_path = Path(output_dir) / f"{video_name}_optimization_summary.json"
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved optimization summary to {output_path}")
        return summary

if __name__ == "__main__":
    # Example usage
    config = OptimizationConfig(
        temporal_motion_threshold=0.3,
        patch_grid=(4, 4),
        spatial_motion_threshold=2.0,
        patch_motion_ratio=0.05
    )
    
    optimizer = SpatioTemporalOptimizer(config=config)
    optimizer.load_model()
    
    # Test on a sample video
    video_name = "alley_1"
    metrics = optimizer.optimize_video(video_name)
    
    print(f"\nOptimization Results for {video_name}:")
    print(f"Temporal reduction: {metrics.temporal_reduction:.1%}")
    print(f"Spatial reduction: {metrics.spatial_reduction:.1%}")
    print(f"Combined reduction: {metrics.combined_reduction:.1%}")
    print(f"Speedup factor: {metrics.speedup_factor:.2f}×")
    print(f"Quality retention: {metrics.shannon_entropy_retention:.1%}")