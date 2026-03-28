#!/usr/bin/env python3
"""
Memory logging utility for Jetson AGX Orin inference diagnosis.
Tracks system RAM and CUDA allocations at each pipeline stage.

Each log() call writes to both stderr (via logging) and a file with os.fsync(),
so entries survive OOM kills that happen before process exit.

Usage:
    from mem_logger import MemLogger
    ml = MemLogger()          # default log: ~/mem_stage.log
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

LOG_PATH = os.path.expanduser("~/mem_stage.log")


class MemLogger:
    def __init__(self, path: str = LOG_PATH):
        self._path = path

    def log(self, tag: str):
        """Log system RAM + CUDA memory at a named stage.
        Writes to logger (stderr) AND fsyncs to file — survives OOM kills.
        """
        GB    = 1 << 30
        vm    = psutil.virtual_memory()
        alloc = torch.cuda.memory_allocated() / GB   # tensors actually holding data
        pool  = torch.cuda.memory_reserved()  / GB   # PyTorch CUDA pool (alloc + cached free blocks)

        line = (
            f"{time.strftime('%H:%M:%S')} "
            f"[MEM {tag}] "
            f"sys={vm.used/GB:.2f}/{vm.total/GB:.1f}GB "
            f"avail={vm.available/GB:.2f}GB "
            f"cuda_alloc={alloc:.2f}GB "
            f"cuda_pool={pool:.2f}GB"
        )

        logger.info(line)

        # fsync write — survives SIGKILL from OOM killer
        with open(self._path, "a") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
