import torch
from torch.profiler import profile, ProfilerActivity

from pathlib import Path
import csv

from engine import SpeculativeEngine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROFILES_DIR = PROJECT_ROOT / "profiles"
PROFILES_DIR.mkdir(exist_ok=True)

trace_path = PROFILES_DIR / "speculative_trace.json"
csv_path = PROFILES_DIR / "speculative_ops.csv"

decoder = SpeculativeEngine()
prompt = "The rapid development of artificial intelligence has led to"

# Warm-up: do not profile initialization / first-use cost.
decoder.generate(prompt, max_new_tokens=5, gamma=4)
torch.cuda.synchronize()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    with_stack=False,
    record_shapes=False,
    profile_memory=False,
) as prof:
    decoder.generate(prompt, max_new_tokens=8, gamma=4)

prof.export_chrome_trace(str(trace_path))
print(f"Trace saved to: {trace_path}")

# print(prof.key_averages().table(
#     sort_by="self_cuda_time_total",
#     row_limit=20,
# ))

events = prof.key_averages()

with csv_path.open("w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "operation",
            "calls",
            "self_cpu_ms",
            "cpu_total_ms",
            "self_cuda_ms",
            "cuda_total_ms",
        ],
    )
    writer.writeheader()

    # for event in events:
    #     writer.writerow({
    #         "operation": event.key,
    #         "calls": event.count,
    #         "self_cpu_ms": event.self_cpu_time_total / 1000,
    #         "cpu_total_ms": event.cpu_time_total / 1000,
    #         "self_cuda_ms": event.self_cuda_time_total / 1000,
    #         "cuda_total_ms": event.cuda_time_total / 1000,
    #     })
    for event in events:
        writer.writerow({
            "operation": event.key,
            "calls": event.count,
            "self_cpu_ms": event.self_cpu_time_total / 1000,
            "cpu_total_ms": event.cpu_time_total / 1000,
            "self_cuda_ms": event.self_device_time_total / 1000,
            "cuda_total_ms": event.device_time_total / 1000,
        })
        
print(f"CSV saved to: {csv_path}")

