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
from code_eval.monitor import PowerMonitor, EnergyMonitor, Measurement
from code_eval.utils import RESULT_DIR
# Energy measurement
from pynvml import *

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

#Create the results directory if they don't exist
for path in RESULT_DIR.values():
    os.makedirs(path, exist_ok=True)


MODEL_NAME_TO_PROMPT = {
    "CodeLlama": "codellama",
    "DeepSeek-Coder": "deepseek",
    "Codestral": "codestral", 
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
    args.task_names = task_names

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
        
        model_name = os.path.basename(args.model)
    
        if args.gpu_memory_utilization:
            if args.gpu_memory_utilization != "auto":
                model_kwargs["gpu_memory_utilization"] = get_gpus_max_memory(
                    args.gpu_memory_utilization, args.tensor_parallel_size
                )
            else:
                model_kwargs["gpu_memory_utilization"] = "auto"
                print("Loading model in auto mode")
        
        # WARNING: the following 2 arguments are used for inflight quantization, which is deprecated
        if args.load_in_8bit:
            # TODO: hqq in-flight quantization is deprecated with vLLM >= v0.7 to find the alternative way
            print("Loading model in 8bit - Using HQQ 8W8A")

            from hqq.utils.vllm import set_vllm_hqq_backend, VLLM_HQQ_BACKEND
            set_vllm_hqq_backend(backend=VLLM_HQQ_BACKEND.GEMLITE) # 8-bit HQQuantization is suggested to be used with gemlite backend
            #set_vllm_hqq_backend(backend=VLLM_HQQ_BACKEND.PYTORCH)

            from hqq.utils.vllm import set_vllm_onthefly_hqq_quant
            set_vllm_onthefly_hqq_quant(weight_bits=8, group_size=None, quant_mode='dynamic', skip_modules=['lm_head']) #dynamic A8W8
            
            model_kwargs['dtype'] = torch.bfloat16
            raise warnings.warn(
                "HQQ on-the-fly quantization is deprecated with vLLM >= 0.7, inference time is much slower" \
                "than the non-quantized model."
            )
            
        elif args.load_in_4bit:
            print("Loading model in 4bit - Using HQQ 4W16A")
            # Source https://github.com/mobiusml/hqq?tab=readme-ov-file#vllm

            from hqq.utils.vllm import set_vllm_hqq_backend, VLLM_HQQ_BACKEND
            set_vllm_hqq_backend(backend=VLLM_HQQ_BACKEND.GEMLITE)
            #set_vllm_hqq_backend(backend=VLLM_HQQ_BACKEND.PYTORCH)

            from hqq.utils.vllm import set_vllm_onthefly_hqq_quant
            set_vllm_onthefly_hqq_quant(weight_bits=4, group_size=64, quant_mode='static', skip_modules=['lm_head']) #A16W4 

            model_kwargs['dtype'] = torch.bfloat16
            raise warnings.warn(
                "HQQ on-the-fly quantization is deprecated with vLLM >= 0.7, inference time is much slower" \
                "than the non-quantized model."
            )

        # Change backend to GEMLITE of HQQuantization models
        config_path = os.path.join(args.model, "config.json")
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            if "quantization_config" in config:
                if "quant_method" in config["quantization_config"]:
                    quant_method = config["quantization_config"]["quant_method"]
                    # Using float16 precision for AWQ and GPTQ models
                    model_kwargs['dtype'] = torch.float16 
                else: quant_method = None
            else:
                quant_method = None

            if quant_method in ('awq', 'gptq'):
                # Avoid error with FlashAttention
                # Source: https://github.com/vllm-project/vllm/issues/5376
                print("Changing awq/gptq attention backend to Xformers instead of FlashAttention")
                os.environ["VLLM_ATTENTION_BACKEND"]="XFORMERS"
            
            if quant_method == 'hqq':
                print("Changing hqq backend for vLLM inference")
                from hqq.utils.vllm import set_vllm_hqq_backend, VLLM_HQQ_BACKEND
                #set_vllm_hqq_backend(backend=VLLM_HQQ_BACKEND.GEMLITE)
                set_vllm_hqq_backend(backend=VLLM_HQQ_BACKEND.PYTORCH)

                # It is suggested to load HQQ model in float16 precision for Gemlite backend and bfloat16 for Torchao's tiny_gemm backend
                # Source: https://github.com/mobiusml/hqq?tab=readme-ov-file#optimized-inference
                model_kwargs['dtype'] = torch.float16 

        except FileNotFoundError:
            print(f"Config file not found at {config_path}")
            return None
        except json.JSONDecodeError:
            print(f"Error parsing JSON in {config_path}")
            return None

        # Measuring energy consumption of the whole process
        main_emonitor = EnergyMonitor(
            gpu_indices=args.gpu_indices,
            cpu_indices=args.cpu_indices,
            log_file=None,
        )

        # Initialization of measuring window for model loading 
        main_emonitor.begin_window('loading_model')
        print("Loading model in precision", model_kwargs['dtype'])
        model = LLM(
            args.model,
            enforce_eager = args.enforce_eager,
            **model_kwargs
            )

        # Energy measurements of the loading process
        ms_loading : Measurement = main_emonitor.end_window('loading_model')

        ### Initial tokenizer setup from bigcode-evaluation-harness
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
        
        ### end
            
        # Addjust the prompt formulation (for code summarization tasks only!)
        if args.prompt == None:
            # Check if any key in MODEL_NAME_TO_PROMPT is a prefix of model_name
            if any(model_name.startswith(key) for key in MODEL_NAME_TO_PROMPT):
                matching_key = next(key for key in MODEL_NAME_TO_PROMPT if model_name.startswith(key))
                args.prompt = MODEL_NAME_TO_PROMPT.get(matching_key)
            else:
                args.prompt = "instruct" # Default prompt formulation for most models

        # Inference with different configuration within a single model load.
        if args.all_max_num_seqs:
            list_max_num_seqs: list[int] = [int(x) for x in args.all_max_num_seqs.split(',')]
        else: 
            list_max_num_seqs = [args.max_num_seqs]
        if args.all_max_tokens:
            list_max_tokens: list[int] = [int(x) for x in args.all_max_tokens.split(',')]
        else: 
            list_max_tokens = [args.max_tokens]
        if args.all_n_samples:
            list_n_samples: list[int] = [int(x) for x in args.all_n_samples.split(',')]
        else: 
            list_n_samples = [args.n_samples]
            
        # Initialize Evaluator for generation and evaluation
        evaluator = Evaluator(model, tokenizer, args)

        if (
            args.load_generations_intermediate_paths
            and len(args.load_generations_intermediate_paths) != len(task_names)
        ):
            raise ValueError(
                "If passing --load_generations_intermediate_paths, \
                must pass equal number of files as number of tasks"
            )
        
        # Dummy generation to warm up the GPU and model
        print("Generating dummy text to warm up the model and GPU")
        model.generate("This is a dummy generation to warm up the model and GPU",
                       sampling_params=SamplingParams(
                           max_tokens=128,
                           temperature=0.0,
                           top_p=1.0,
                       ),
        )
        time.sleep(3)

        # Generation and evaluation for each task
        main_emonitor.begin_window('inference')
        for idx, task in enumerate(task_names):
            print(f"Processing task: {task}")
            intermediate_generations = None
            # For completed generated file, evaluator evaluate only instead of generate + evaluate.
            if args.load_generations_intermediate_paths:
                with open(args.load_generations_intermediate_paths[idx], "r") as f_in:
                    # intermediate_generations: list[list[str | None]] of len n_tasks
                    # where list[i] = generated codes or empty
                    intermediate_generations = json.load(f_in)
            
            torch.cuda.synchronize()
            gen_bclock = time.time()

            
            # Generation part for each set of configuration
            for n_samples in list_n_samples:
                for max_tokens in list_max_tokens:
                    for mns in list_max_num_seqs:
                        args.n_samples = n_samples
                        args.max_tokens = max_tokens
                        args.max_num_seqs = mns

                        print(f'Generaion for n-samples{n_samples}, max-tokens{max_tokens}, batch-size{mns}')

                        # Update evaluator arguments
                        evaluator.args = args
                        # - Change LLM Engine scheduler batch size
                        model.llm_engine.vllm_config.scheduler_config.max_num_seqs=mns

                        # Set up the power monitoring with multiprocessing FOR EACH CONFIGURATION OF EXP.
                        if not args.no_monitor:
                            os.makedirs(args.save_monitoring_folder, exist_ok=True)
                            power_dir = os.path.join(args.save_monitoring_folder, 'power')
                            os.makedirs(power_dir, exist_ok=True)
                            power_monitor = PowerMonitor(
                                gpu_indices=args.gpu_indices,
                                update_period=args.update_period,
                                power_csv_path= os.path.join(power_dir, f'{model_name}_{','.join(task_names)}_mns{args.max_num_seqs}_max-toks{args.max_tokens}_n{args.n_samples}.csv')
                            )
                            time.sleep(10)

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

                        if not args.no_monitor:
                            try:
                                time.sleep(10)
                                power_monitor._stop()
                            except Exception as e:
                                print(f'Failed to stop power monitor: {e}')
        
        ms_inference = main_emonitor.end_window('inference')
        

    # Over all measurements of main.py, for detailed batch-level measurements, enable --save_monitoring_folder
    measurements = dict(total_execution_time = ms_loading.time + ms_inference.time,
                        model_loading_time = ms_loading.time,
                        model_loading_energy = ms_loading.total_energy,
                        tasks_execution_time = ms_inference.time,
                        overall_energy = ms_inference.total_energy + ms_loading.total_energy)

    # Save all args to config
    results["config"] = vars(args)
    results["main.py measurements"] = measurements

    dumped = json.dumps(results, indent=2)
    print(dumped)
    metrics_dir = os.path.join(args.save_monitoring_folder, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)

    if 'humanevalexplainsynthesize' in args.task_names[0]:
        # If it is humanevalexplainsynthesize task, we need to specify the language
        from code_eval import tasks
        # Suppose that the 'tasks' argument has only one task for HEE  
        task_var = tasks.get_task(args.task_names[0], args)
        language = task_var.DATASET_NAME
        try:
            describe_model = os.path.basename(args.load_data_path).replace(f'_humanevalexplaindescribe-{language}.json','')
        except Exception as e:
            print(f"Failed to get describe model name from {args.load_data_path}: {e}")
        print(f"Saving metrics for describe model {describe_model}")
        metrics_path = os.path.join(metrics_dir, f'{describe_model}_{model_name}_{','.join(task_names)}.json')
    else:
        metrics_path = os.path.join(metrics_dir, f'{model_name}_{','.join(task_names)}.json')
    with open(metrics_path, "w") as f:
        f.write(dumped)


if __name__ == "__main__":
    main()

    # Sleep time for power usage return to normal level
    time.sleep(10)

    
