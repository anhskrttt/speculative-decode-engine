import torch
from time import perf_counter

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Running on device: {device}")

a = torch.randn(8192, 8192, device=device)
b = torch.randn(8192, 8192, device=device)

def run_kernel():
    out = a @ b

steps = 10

# Warmup steps
for _ in range(steps):
    run_kernel() # don't record time

start_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
end_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]

for i in range(steps):
    start_events[i].record()
    run_kernel()
    end_events[i].record()

torch.cuda.synchronize()
times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
# print(f"All times (ms): {times}")
# print(f"Average time with CUDA events: {sum(times)/steps:.3f} ms")