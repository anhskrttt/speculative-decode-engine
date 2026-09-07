# Source: https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch?utm_source=chatgpt.com

import torch
from time import perf_counter

steps = 10
times = []
sync_times = []

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Running on device: {device}")

a = torch.randn(8192, 8192, device=device)
b = torch.randn(8192, 8192, device=device)

def run_kernel():
    out = a @ b

# Warm up
print("Warming up...")
for _ in range(3):
    run_kernel()


# This won't time the CUDA kernel - only the launch overhead
print("Timing without synchronize...")
for _ in range(steps):
    start_time = perf_counter()

    run_kernel() # Some kernel

    end_time = perf_counter()
    times.append(end_time - start_time)

# This measures what we actually care about
print("Timing with synchronize...")
for _ in range(steps):
    start_time = perf_counter()

    run_kernel()
    torch.cuda.synchronize()

    end_time = perf_counter()
    sync_times.append(end_time - start_time)
    
print(f"Average time without synchronize: {sum(times)/steps*1000:.3f} ms")
print(f"Average time with synchronize: {sum(sync_times)/steps*1000:.3f} ms")