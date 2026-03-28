#!/usr/bin/env python3
"""
Quick test: check if mem_efficient_sdp is available on this Jetson,
and measure memory reduction vs standard eager attention.
"""

import torch
from mem_logger import MemLogger

ml = MemLogger()

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
print(f"Device          : {torch.cuda.get_device_name(0)}")
print(f"Compute cap     : sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}")
print()

# --- SDP backend availability ---
print("SDP backend support:")
print(f"  flash_sdp        : {torch.backends.cuda.flash_sdp_enabled()}")
print(f"  mem_efficient_sdp: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
print(f"  math_sdp         : {torch.backends.cuda.math_sdp_enabled()}")
print()

# --- Simulate prefill attention with a large sequence ---
# Approximate 13 frames × ~540 tokens = ~7000 tokens
N   = 7000   # sequence length
H   = 28     # attention heads (Qwen2.5-VL-7B)
D   = 128    # head dim
dtype = torch.float16
device = "cuda:0"

q = torch.randn(1, H, N, D, dtype=dtype, device=device)
k = torch.randn(1, H, N, D, dtype=dtype, device=device)
v = torch.randn(1, H, N, D, dtype=dtype, device=device)

# Test 1: math (standard n×n — current baseline)
torch.cuda.empty_cache()
ml.log(f"before math sdp  (N={N})")
try:
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        out_math = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    ml.log("after  math sdp")
except Exception as e:
    print(f"math sdp failed: {e}")

torch.cuda.empty_cache()

# Test 2: mem_efficient (tiled — target)
ml.log(f"before mem_efficient sdp (N={N})")
try:
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
        out_eff = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    ml.log("after  mem_efficient sdp")
    print("mem_efficient_sdp: OK")
except Exception as e:
    print(f"mem_efficient_sdp failed: {e}")
