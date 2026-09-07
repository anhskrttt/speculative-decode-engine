import time
import torch

assert torch.cuda.is_available(), "CUDA GPU not available"

device = "cuda"

# Large enough that GPU work takes noticeable time
a = torch.randn(8192, 8192, device=device)
b = torch.randn(8192, 8192, device=device)

# Warm up CUDA, cuBLAS, and memory allocator first
for _ in range(3):
    _ = a @ b
torch.cuda.synchronize()

print("GPU:", torch.cuda.get_device_name(0))

# ❌ Incorrect GPU timing: measures mostly CPU launch overhead
start = time.perf_counter()
out = a @ b
end = time.perf_counter()

print(f"Without synchronize: {(end - start) * 1000:.3f} ms")

# The GPU work is still in progress here; wait for it.
wait_start = time.perf_counter()
torch.cuda.synchronize()
wait_end = time.perf_counter()

print(f"Time spent waiting afterward: {(wait_end - wait_start) * 1000:.3f} ms")

# ✅ Correct GPU timing
torch.cuda.synchronize()
start = time.perf_counter()
out = a @ b
torch.cuda.synchronize()
end = time.perf_counter()

print(f"With synchronize: {(end - start) * 1000:.3f} ms")