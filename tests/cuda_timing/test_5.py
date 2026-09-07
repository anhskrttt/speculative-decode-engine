import statistics
import torch
import torch.nn as nn

assert torch.cuda.is_available(), "CUDA GPU is required"

device = "cuda"

# A simple operation to benchmark.
# The weights of this layer are reused on every call.
linear = nn.Linear(4096, 4096, bias=False, device=device, dtype=torch.float16)
input_tensor = torch.randn(1, 4096, device=device, dtype=torch.float16)

# This tensor is deliberately larger than GPU L2 cache.
# Writing to it fills/evicts much of L2, so the next Linear call is less likely
# to reuse cached weights/data from the previous call.
#
# 128 MiB is a simple conservative size for a local example.
cache_flush_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)


def flush_cache():
    # Launches a GPU operation that writes 128 MiB of zeros.
    cache_flush_buffer.zero_()


def measure_linear(flush_before_each_run: bool, repeats: int = 30) -> list[float]:
    times_ms = []

    # Warm up: initialization, memory allocation, and compilation effects
    # should not be included in the benchmark.
    for _ in range(10):
        _ = linear(input_tensor)
    torch.cuda.synchronize()

    for _ in range(repeats):
        if flush_before_each_run:
            flush_cache()

        # The CPU can enqueue CUDA work asynchronously, so CUDA events measure
        # elapsed time on the GPU itself.
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        _ = linear(input_tensor)
        end.record()

        # Wait until all GPU work up to `end` has completed.
        torch.cuda.synchronize()

        times_ms.append(start.elapsed_time(end))

    return times_ms


no_flush = measure_linear(flush_before_each_run=False)
with_flush = measure_linear(flush_before_each_run=True)

print(f"GPU: {torch.cuda.get_device_name()}")
print()
print("Linear only (cache may be warm):")
print(f"  median: {statistics.median(no_flush):.3f} ms")
print(f"  min:    {min(no_flush):.3f} ms")
print()
print("Linear only after flushing cache:")
print(f"  median: {statistics.median(with_flush):.3f} ms")
print(f"  min:    {min(with_flush):.3f} ms")