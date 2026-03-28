#!/usr/bin/env python3
"""
Memory logging utility for Jetson AGX Orin inference diagnosis.
Tracks system RAM and CUDA allocations at each pipeline stage.

Usage:
    from mem_logger import MemLogger
    ml = MemLogger()
    ml.log("after model load")
    ml.log("after processor")
    ...
"""

import os
import time
import psutil
import torch
import logging

logger = logging.getLogger(__name__)


class MemLogger:
    def log(self, tag: str):
        """Log system RAM + CUDA memory at a named stage."""
        vm    = psutil.virtual_memory()
        alloc = torch.cuda.memory_allocated() / 1<<30   # tensors actually holding data
        pool  = torch.cuda.memory_reserved()  / 1<<30   # PyTorch CUDA pool (alloc + cached free blocks)

        logger.info(
            f"[MEM {tag}] "
            f"sys={vm.used/1<<30:.2f}/{vm.total/1<<30:.1f}GB "
            f"avail={vm.available/1<<30:.2f}GB "
            f"cuda_alloc={alloc:.2f}GB "
            f"cuda_pool={pool:.2f}GB"
        )
