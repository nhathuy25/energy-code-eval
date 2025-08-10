# Evaluate Energy Efficiency for Code Generation with Open LLMs

This repository evaluates the energy cost of code-generation benchmarks using open-source large language models (LLMs).  
It extends [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness) and relies on [vLLM](https://github.com/vllm-project/vllm) for high-throughput inference, with built‑in energy and power monitoring.

---

## Requirements

Main dependencies:
- `vllm==0.8.4`
- `transformers>=4.51.0`
- `hqq==0.2.7`
- `gemlite==0.4.7`
- `sentence-transformers==4.1.0`
- `datasets==3.2.0`

See `requirements.txt` for the full list.

---

## Repository Structure

```
energy-code-eval/
├── code_eval/            # Core evaluation package
│   ├── monitor/          # Energy/power monitoring (NVIDIA GPUs only)
│   │   ├── device/
│   │   ├── energy.py
│   │   └── power.py
│   ├── tasks/            # Benchmark definitions and metrics
│   ├── arguments.py      # CLI argument definitions
│   ├── evaluator.py
│   └── generation.py
├── data/                 # Experiment inputs & results
│   ├── final_result/
│   └── sub_experiments/
├── results/              # Raw experiment outputs
├── slurm/                # SLURM job scripts
├── main.py               # Entry point for running evaluations
└── setup.py
```

---

## Basic Usage

All arguments are defined in `code_eval/arguments.py`.

### Simple Inference with vLLM
```bash
python3 main.py \
  --model <MODEL_DIRECTORY> \
  --tasks humaneval \
  --n_samples 5 \
  --temperature 0 \
  --top_p 1 \
  --max_tokens 512 \
  --max_model_len 16384 \
  --allow_code_execution \
  --trust_remote_code \
  --enforce_eager \
  --max_num_seqs 128 \
  --num_scheduler_steps 1 \
  --enable_chunked_prefill False \
  --no_monitor
```

### Energy & Power Monitoring (pyNVML)
```bash
python3 main.py \
  --model <MODEL_DIRECTORY> \
  --tasks humaneval \
  --n_samples 5 \
  --temperature 0 \
  --top_p 1 \
  --max_tokens 512 \
  --no_stop \
  --generation_only \
  --trust_remote_code \
  --enforce_eager \
  --max_model_len 16384 \
  --max_num_seqs 128 \
  --num_scheduler_steps 1 \
  --enable_chunked_prefill False \
  --save_monitoring_folder <SAVE_MONITOR_RESULT_PATH>
```

**Monitoring output structure**
```
<SAVE_MONITOR_RESULT_PATH>/
├── energy/model_name_energy.csv    # energy and throughput data
├── metrics/model_name_metrics.json # correctness & configuration info
└── power/model_name_power.csv      # power measurements over time
```

### HumanEvalExplain Benchmark

1. **Describe phase** – generate natural-language summaries from reference code:

```bash
LANGUAGE=python   # python | java | javascript
python3 main.py \
  --model <MODEL_DIRECTORY> \
  --tasks humanevalexplaindescribe-$LANGUAGE \
  --n_samples 5 \
  --temperature 0 \
  --top_p 1 \
  --max_tokens 512 \
  --max_num_seqs 128 \
  --allow_code_execution \
  --trust_remote_code \
  --enforce_eager \
  --max_model_len 16384 \
  --num_scheduler_steps 1 \
  --enable_chunked_prefill False \
  --save_generations \
  --save_generations_path <FILE_NAME>.json \
  --save_monitoring_folder <SAVE_MONITOR_RESULT_PATH> \
  --generation_only
```

2. **Synthesize phase** – generate code from descriptions and evaluate pass@1:

```bash
LANGUAGE=python
python3 main.py \
  --model <MODEL_DIRECTORY> \
  --tasks humanevalexplainsynthesize-$LANGUAGE \
  --n_samples 5 \
  --temperature 0 \
  --top_p 1 \
  --max_tokens 512 \
  --allow_code_execution \
  --enforce_eager \
  --max_model_len 16384 \
  --num_scheduler_steps 1 \
  --enable_chunked_prefill False \
  --max_num_seqs 128 \
  --load_data_path <FILE_NAME>_humanevalexplaindescribe-$LANGUAGE.json \
  --trust_remote_code \
  --no_monitor \
  --save_monitoring_folder <SAVE_MONITOR_RESULT_PATH>
```

---

## SLURM Jobs

Example scripts for running distributed experiments on HPC clusters are provided in `slurm/`.

---

## Notes on Backends

- Base instruction‑tuned models use **FlashAttention‑2**.
- Quantized models switch to **X‑formers** due to FlashAttention‑2 incompatibilities with AWQ quantization.
- HQQ models run faster with their PyTorch backend than with Gemlite, but still lag in efficiency on vLLM 0.8.4.

---

## Experimental Studies

`data/sub_experiments/` contains exploratory studies:

- **exp2_diffInputTokens**  
  Investigates how varying input sequence lengths affect prefilling. Prefill time depends only on total tokens, not per‑sequence variance.

- **exp5_warmupGPU**  
  Tests whether GPU warm‑up improves throughput and examines run‑to‑run variability on the same GPU.

- **exp6_diffN-queries**  
  Observes how different numbers of queries (n‑samples) influence latency and whether intermediate batch sizes align with batching‑experiment trends.

---

## V1 Experiments (vLLM v0.8.4)

A separate branch `vllm_v1` implements the re‑architected vLLM v1 engine:
- Multi-step & chunked prefill are always active.
- Engine internals are restructured.
- `max_num_seqs` cannot be adjusted after initialization.

---

## License

This project builds on open-source tools; check upstream repositories and individual files for license details.

