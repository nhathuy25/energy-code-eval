# Evaluate Energy-Efficient for Code Generation Tasks with Open-Sourced LLM

Based on [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness).
Applying faster LLMs inferences with [vLLM](https://github.com/vllm-project/vllm).

## Versions

- vllm == 0.8.4
- transformers >= 4.51.0
- hqq == 0.2.7
- gemlite == 0.4.7
- sentence-transformers == 4.1.0
- datasets == 3.2.0

See more in `requirements.txt`

##
```
energy-code-eval/
├── code_eval/
│   ├── monitor/							:
│   │   ├── device/
│   │   ├── __init__.py
│   │   ├── energy.py
│   │   ├── power.py
│   │   └── utils.py
│   ├── tasks/								:
│   │   ├── custom_metrics/
│   │   ├── __init__.py
│   │   ├── <TASKS_FILE>.py
│   │   └── utils.py
│   ├── __init__.py
│   ├── arguments.py
│   ├── base.py
│   ├── evaluator.py
│   ├── generation.py
│   └── utils.py
├── results/								:
│   ├── batching/
│   │   └── <N_SAMPLES VALUES>/
│   ├── correctness/
│   ├── correctness_hee/
│   ├── scheduler/
│   ├── export_energy.py					:
│   └── export_metrics.py					:
├── slurm/									:
├── main.py									:
└── setup.py
```

## How to use

Please refer to [this file](./code_eval/arguments.py) for arguments' informations.

### Basic load and inference with vLLM
```cmd
python3 main.py \
	--model <MODEL_DIRECTORY> \
	--tasks humaneval \ # To execute multiple task, seperate the tasks by ',' (eg. humaneval,mbpp,codesearchnet-python)
	--n_samples 5 \
	--temperature 0 \
	--top_p 1 \
	--max_tokens 512 \
	--max_model_len 16384 \
	--allow_code_execution \ # To execute pass@k evaluation after generation
	--trust_remote_code \ 
	--enforce_eager \ # Disable compute CUDA graph for all the experiments
	--max_num_seqs 128 \
	--num_scheduler_steps 1 \
	--enable_chunked_prefill False \
	--no_monitor
```

Exist `--save_generations` and `--save_generations_path` to save the generated code.

### Energy and power consumption monitoring with pyNVML
```cmd
python3 main.py \
	--model <MODEL_DIRECTORY> \
	--tasks humaneval \
	--n_samples 5 \
	--temperature 0 \
	--top_p 1 \
	--max_tokens 512 \
	--no_stop \ # Ignore end-of-sequences token to generate exact 'max_tokens' for homogeneity of throughput mesurement
	--generation_only \ # Skip the correctness (pass@k score) evaluation 
	--trust_remote_code \
	--enforce_eager \
	--max_model_len 16384 \
	--max_num_seqs 128 \ # Modify here for 'batching' experiments
	--num_scheduler_steps 1 \ # Modify here and --enable_chunked_prefill for 'scheduler' experiment
	--enable_chunked_prefill False \
	--save_monitoring_folder <SAVE_MONITOR_RESULT_PATH> 
```

The structure of `monitoring folder`:
- <SAVE_MONITOR_RESULT_PATH>
  - energy
    - model_name_energy.csv
  - metrics
    - model_name_metrics.json
  - power
    - model_name_power.csv

Where 
- `model_name_energy.csv` including informations about energy consumed, throughput of different model with multiple batch and/or multiple tasks during inference.
- `model_name_metrics.json` for saving correctness results and configurations and parameters.
- `model_name_power.csv` monitoring power consumption of selected GPU(s) over time.

### HumanEvalExplain benchmarks for evaluating code summerization
HumanEvalExplain require 2 times execution, first for generating description D from the reference 'canonical_solution' code C1 from original dataset and, second, for generating synthesized code C2 and measure the pass@k of C2 to assess the capbability of the model in summarization D of original code C1.

Original dataset -> C1 -> D -> C2 -> pass@k

```cmd
# HumanEvalExplainDescribe : For generating natural language description D from the canonical solution code C1 of the original dataset HumanEval.

## Exist 3 available language : python, java, javascript
LANGUAGE=python 
python3 main.py \
	--model <MODEL_DIRECTORY> \
	--tasks humanevalexplaindescribe-${LANGUAGE} \
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
	--save_generations \ # Boolean parameter to save a version of generated summarizations D for the next synthesized code C2
	--save_generations_path <FILE_NAME.json> \ # Path and name of the .json file
	--save_monitoring_folder <SAVE_MONITOR_RESULT_PATH>  \ # Path to the directory to save energy monitoring information
	--generation_only
```

```
# HumanEvalExplainSynthesize : For generating synthesized code C2 and evaluation with pass@1 score.

LANGUAGE=python 
python3 main.py  \
	--model $CONTAINER_DATASETS/$MODEL_NAME \
	--tasks humanevalexplainsynthesize-${LANGUAGE} \
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

## Launching SLURM's jobs

Please refers to some SLURM examples [here](./slurm/)

## Notes on kernel's backend
- With base (instruction finetuned) models, we use `FlashAttention-2` by default.
- With quantized models, we change the backend to `X-formers`, which is developed on `FlashAttention`. This is necessary because `FlashAttention-2` in this version of vLLM leads to CUDA errors ([source](https://github.com/vllm-project/vllm/issues/5376)), especially with AWQ quantization.
- For HQQ models, implementation with PyTorch backend ([main.py - line 128](./main.py#L128)) is faster than Gemlite, which is recommended by the author. However, with vLLM v0.8.4, we still observe underperformance of HQQ in terms of throughput and energy efficiency.

## V1 Experiments

Switch to branch `vllm_v1` for V1 implementation of vLLM version 0.8.4 ([source](https://developers.redhat.com/articles/2025/04/28/performance-boosts-vllm-081-switching-v1-engine#architectural_changes_and_simplifications))
Majors changes
- Multi-step and Chunked prefill are always active.
- Engine re-architecture.
- etc.


