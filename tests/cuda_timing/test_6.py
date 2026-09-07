import statistics
import torch

# ---------- Configuration ----------
DEVICE = "cuda"
MATRIX_SIZE = 4096
WARMUP_STEPS = 10
MEASURE_STEPS = 10
REPEATS_PER_GRAPH = 20
# -----------------------------------


def run_kernel(a, b, out):
    """
    One GPU workload.

    `out` is preallocated, which is important:
    CUDA Graph capture requires stable tensor addresses and avoids
    allocating new output tensors during each replay.
    """
    torch.mm(a, b, out=out)


def main():
    assert torch.cuda.is_available(), "CUDA GPU is required"

    print("GPU:", torch.cuda.get_device_name(0))
    print(f"Matrix size: {MATRIX_SIZE} x {MATRIX_SIZE}")
    print(f"Each graph replay: {REPEATS_PER_GRAPH} matrix multiplications")

    torch.manual_seed(0)

    # Static GPU tensors: allocated once and reused.
    a = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=DEVICE)
    b = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=DEVICE)
    out = torch.empty(MATRIX_SIZE, MATRIX_SIZE, device=DEVICE)

    # 1. Warm up CUDA, cuBLAS, allocator, and GPU clocks.
    print("\nWarming up...")
    for _ in range(WARMUP_STEPS):
        run_kernel(a, b, out)

    torch.cuda.synchronize()

    # 2. Capture a fixed sequence of GPU operations.
    #
    # The graph contains 20 matmuls. A later replay launches the
    # pre-recorded sequence with much less CPU launch overhead.
    print("Capturing CUDA graph...")
    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph):
        for _ in range(REPEATS_PER_GRAPH):
            run_kernel(a, b, out)

    torch.cuda.synchronize()

    # 3. Create GPU-side timestamp markers.
    start_events = [
        torch.cuda.Event(enable_timing=True)
        for _ in range(MEASURE_STEPS)
    ]
    end_events = [
        torch.cuda.Event(enable_timing=True)
        for _ in range(MEASURE_STEPS)
    ]

    # 4. Measure each replay.
    #
    # `record()` inserts events into the same CUDA stream:
    #
    # start event → graph's 20 matmuls → end event
    #
    # So elapsed_time measures GPU execution between those markers.
    for start, end in zip(start_events, end_events):
        start.record()
        graph.replay()
        end.record()

    # Wait only once, after all measurements have been queued.
    torch.cuda.synchronize()

    # elapsed_time returns milliseconds.
    graph_times_ms = [
        start.elapsed_time(end)
        for start, end in zip(start_events, end_events)
    ]

    # Each replay contains multiple matmuls, so divide to estimate
    # time for one matrix multiplication.
    kernel_times_ms = [
        graph_time / REPEATS_PER_GRAPH
        for graph_time in graph_times_ms
    ]

    print("\n--- Results ---")
    print("Graph replay times (ms):")
    print([round(t, 3) for t in graph_times_ms])

    print("\nEstimated one matmul time (ms):")
    print([round(t, 3) for t in kernel_times_ms])

    print(f"\nMedian per matmul: {statistics.median(kernel_times_ms):.3f} ms")
    print(f"Mean per matmul:   {statistics.mean(kernel_times_ms):.3f} ms")
    print(
        f"Min / Max:          "
        f"{min(kernel_times_ms):.3f} / {max(kernel_times_ms):.3f} ms"
    )


if __name__ == "__main__":
    main()