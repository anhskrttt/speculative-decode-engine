# Speculative Decoding Benchmark

## Goal

Autoregressive decoding normally emits one token per expensive target-model forward pass. Speculative decoding uses a smaller **draft model** to propose a short continuation, then uses one **target-model verification forward pass** to score every proposed position. If sufficient proposals are accepted, the target produces multiple output tokens per call without changing its intended distribution.

## Status

**In progress...**

Implemented / running:

- Draft-token generation and batched target verification.
- Throughput and draft acceptance-rate reporting.

Planned extensions:
- Auto tuning gamma for each architecture.
- GPU profiling data.
- GPU-synchronized timing, warm-up.
- Peak-VRAM, draft-time, and target-verification-time measurements.
- Greedy token-ID equality tests against target-only decoding.
- KV-cache-aware draft generation and before/after profiling.


## Benchmark methodology

### Hardware and model configuration

| Item | Configuration |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24 GB VRAM |
| Target model | `facebook/opt-1.3b` (default; configurable) |
| Draft model | `facebook/opt-125m` (default; configurable) |
| Precision | FP16 recommended on CUDA |
| Workload | Fixed prompt and fixed generated-token budget |

The draft and target models must share a tokenizer and vocabulary. Use models in the same family.

## Quick start

```bash
git clone 
cd speculative-decoding-benchmark

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run a baseline-versus-speculative comparison:

```bash
cd src
python compare.py --gamma 4 --max-new-tokens 40
```

Run the gamma sweep:

```bash
python experiments.py
```

Expected outputs:

```text
results/gamma_analysis.png
results/gamma_results.csv
```

## Project structure

```text
.
├── README.md
├── requirements.txt
├── src/
│   ├── config.py        # Model and sampling configuration
│   ├── baseline.py      # Target-only autoregressive decoding
│   ├── sampling.py      # Token sampling and rejection sampling
│   ├── engine.py        # Drafting and target verification loop
│   ├── compare.py       # Baseline vs. chosen gamma
│   └── experiments.py   # Gamma sweep and result plot
└── results/
    ├── gamma_analysis.png
    └── gamma_results.csv
```

## Results

- The longer draft assistant's proposed squence, the worse performance (for both TPS and acceptance rate). TODO: Is there any solution?
- Average improvement is only aroun ~1.1x (not really significant). TODO: Why? How does it perform on different accelerators?

![Gamma analysis](results/gamma_analysis.png)

## References and attribution

- Leviathan, Kalman, and Matias. [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192).
- Chen et al. [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318).
- Hugging Face. [Assisted Generation](https://huggingface.co/blog/assisted-generation).
- The initial educational implementation was informed by [kunal51107/Speculative-decoding-engine](https://github.com/kunal51107/Speculative-decoding-engine). This repository extends that starting point with a reproducible consumer-GPU benchmark and cache-aware optimization study.