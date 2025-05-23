import os
import fnmatch
import json
import warnings
import time

import datasets
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.config import ModelConfig

from code_eval.arguments import parse_args
from code_eval.evaluator import Evaluator
from code_eval.tasks import ALL_TASKS
from code_eval.monitor import PowerMonitor
from code_eval.utils import RESULT_DIR
# Energy measurement
from pynvml import *

# April 2025: Use the V0 version of vLLM since V1 is still experimental
os.environ['VLLM_USE_V1'] = '0'

nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

#Create the results directory if they don't exist
for path in RESULT_DIR.values():
    os.makedirs(path, exist_ok=True)

MODEL_NAME_TO_LOCAL_DIR = {
    "codellama7": '/workdir/models/CodeLlama-7b-hf',
    "codellama7i": '/workdir/models/CodeLlama-7b-Instruct-hf',
    "codellama34i" : '/workdir/models/CodeLlama-34b-Instruct-hf',
    "deepseek_base" : '/workdir/models/DeepSeek-Coder-V2-Lite-Base',
    "deepseek_instruct" : '/workdir/models/DeepSeek-Coder-V2-Lite-Instruct',
    "codestral" : '/workdir/models/Codestral-22B-v0.1',
}


def pattern_match(patterns, source_list):
    """Returns a list containing all values of the source_list that
    match at least one of the patterns"""
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)


def get_gpus_max_memory(max_memory, num_gpus):
    max_memory = {i: max_memory for i in range(num_gpus)}
    print("Loading model via these GPUs & max memories: ", max_memory)
    return max_memory


def main():
    args = parse_args()
    transformers.logging.set_verbosity_error()
    datasets.logging.set_verbosity_error()


    if args.tasks is None:
        task_names = 'humaneval'
    else:
        task_names = pattern_match(args.tasks.split(","), ALL_TASKS)

    print(f"Selected Tasks: {task_names}")

    results = {}
    if args.load_generations_path:
        # here we don't generate code but only evaluate previously computed generations
        print("evaluation only mode")
        evaluator = Evaluator(None, None, args)
        for task in task_names:
            results[task] = evaluator.evaluate(task)
    else:        
        model_kwargs = {
            "revision": args.revision,
            "trust_remote_code": args.trust_remote_code,
            "tensor_parallel_size": args.tensor_parallel_size,
            "dtype" : args.dtype,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "num_scheduler_steps": args.num_scheduler_steps,
            "enable_chunked_prefill": args.enable_chunked_prefill,
        }
        
        if args.model in MODEL_NAME_TO_LOCAL_DIR:
            args.model = MODEL_NAME_TO_LOCAL_DIR[args.model]
        model_name = os.path.basename(args.model)
    
        if args.gpu_memory_utilization:
            if args.gpu_memory_utilization != "auto":
                model_kwargs["gpu_memory_utilization"] = get_gpus_max_memory(
                    args.gpu_memory_utilization, args.tensor_parallel_size
                )
            else:
                model_kwargs["gpu_memory_utilization"] = "auto"
                print("Loading model in auto mode")
        
        # TODO: Quantization replace with vLLM 
        if args.load_in_8bit:
            print("Loading model in 8bit - Using HQQ 8W8A")

            from hqq.utils.vllm import set_vllm_hqq_backend, VLLM_HQQ_BACKEND
            set_vllm_hqq_backend(backend=VLLM_HQQ_BACKEND.MARLIN)

            from hqq.utils.vllm import set_vllm_onthefly_hqq_quant
            set_vllm_onthefly_hqq_quant(weight_bits=8, group_size=None, quant_mode='dynamic', skip_modules=['lm_head']) #dynamic A8W8
            
            model_kwargs['dtype'] = torch.bfloat16
            
        elif args.load_in_4bit:
            print("Loading model in 4bit - Using HQQ 4W16A")
            # Source https://github.com/mobiusml/hqq?tab=readme-ov-file#vllm

            from hqq.utils.vllm import set_vllm_hqq_backend, VLLM_HQQ_BACKEND
            set_vllm_hqq_backend(backend=VLLM_HQQ_BACKEND.MARLIN)

            from hqq.utils.vllm import set_vllm_onthefly_hqq_quant
            set_vllm_onthefly_hqq_quant(weight_bits=4, group_size=64, quant_mode='static', skip_modules=['lm_head']) #A16W4 

            model_kwargs['dtype'] = torch.bfloat16

        else:
            print(f"Loading model in {args.dtype}")

        bclock = time.time()
        start_energy = nvmlDeviceGetTotalEnergyConsumption(handle)

        model = LLM(
            args.model,
            enforce_eager = args.enforce_eager,
            **model_kwargs
            )
        torch.cuda.synchronize()
        eclock = time.time()
        model_loading_time = eclock - bclock
        model_loading_energy = (nvmlDeviceGetTotalEnergyConsumption(handle) - start_energy) / 1000

        
        # TODO: decide whether to use Peft or not
        """
        if args.peft_model:
            from peft import PeftModel  # dynamic import to avoid dependency on peft

            model = PeftModel.from_pretrained(model, args.peft_model)
            print("Loaded PEFT model. Merging...")
            model.merge_and_unload()
            print("Merge complete.")
        """
        if args.left_padding:
            # left padding is required for some models like chatglm3-6b
            tokenizer = AutoTokenizer.from_pretrained(
                args.model,
                revision=args.revision,
                trust_remote_code=args.trust_remote_code,
                token=args.use_auth_token,
                padding_side="left",  
            )
        else:
            # used by default for most models
            tokenizer = AutoTokenizer.from_pretrained(
                args.model,
                revision=args.revision,
                trust_remote_code=args.trust_remote_code,
                token=args.use_auth_token,
                truncation_side="left",
                padding_side="right",  
            )
        if not tokenizer.eos_token:
            if tokenizer.bos_token:
                tokenizer.eos_token = tokenizer.bos_token
                print("bos_token used as eos_token")
            else:
                raise ValueError("No eos_token or bos_token found")
        try:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Some models like CodeGeeX2 have pad_token as a read-only property
        except AttributeError:
            print("Not setting pad_token to eos_token")
            pass
            
        if not args.no_monitor:
            power_monitor = PowerMonitor(
                gpu_indices=args.gpu_indices,
                update_period=args.update_period,
                power_csv_path=RESULT_DIR["energy"] +f'power/{model_name}.csv'
            )
            
        evaluator = Evaluator(model, tokenizer, args)

        if (
            args.load_generations_intermediate_paths
            and len(args.load_generations_intermediate_paths) != len(task_names)
        ):
            raise ValueError(
                "If passing --load_generations_intermediate_paths, \
                must pass equal number of files as number of tasks"
            )

        for idx, task in enumerate(task_names):
            intermediate_generations = None
            # For completed generated file, evaluator evaluate only instead of generate + evaluate.
            if args.load_generations_intermediate_paths:
                with open(args.load_generations_intermediate_paths[idx], "r") as f_in:
                    # intermediate_generations: list[list[str | None]] of len n_tasks
                    # where list[i] = generated codes or empty
                    intermediate_generations = json.load(f_in)

            gen_bclock = time.time()

            if args.generation_only:
                print("generation mode only")
                generations, references = evaluator.generate_text(
                    task, intermediate_generations=intermediate_generations
                )
                save_generations_path = f"{os.path.splitext(args.save_generations_path)[0]}_{task}.json"
                save_references_path = f"references_{task}.json"
                evaluator.save_json_files(
                    generations,
                    references,
                    save_generations_path,
                    save_references_path,
                )
            else:
                results[task] = evaluator.evaluate(
                    task, intermediate_generations=intermediate_generations
                )
                gen_eclock = time.time()
                results[task]["generation_time"] = gen_eclock - gen_bclock
        
        power_monitor._stop()
        tasks_execution_time = time.time() - gen_bclock
        total_execution_time = time.time() - eclock

    measurements = dict(total_execution_time=total_execution_time,
                        model_loading_time=model_loading_time,
                        peak_memory=0.0,
                        flops=0.0,
                        model_loading_energy=model_loading_energy,
                        total_num_tokens=0,)

    # Save all args to config
    results["config"] = vars(args)
    results["main.py measurements"] = measurements
    if not args.generation_only:
        dumped = json.dumps(results, indent=2)
        print(dumped)

        with open(args.metric_output_path, "w") as f:
            f.write(dumped)

if __name__ == "__main__":
    main()

    
nvmlShutdown()
