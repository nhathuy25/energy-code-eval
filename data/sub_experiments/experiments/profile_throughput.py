"""
2025-06-20: Monitoring the number of tokens generated over time (over sequences)
"""
import os
from code_eval.utils import *
from code_eval.generation import parallel_generations
from code_eval import tasks
from code_eval.arguments import parse_args
from code_eval.monitor import EnergyMonitor, PowerMonitor, Measurement

from typing import Union, List, Optional

import argparse
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader

from vllm import LLM, SamplingParams, LLMEngine, EngineArgs
from vllm.outputs import RequestOutput, PoolingRequestOutput

import json

import multiprocessing as mp
import pandas as pd


os.environ['VLLM_USE_V1'] = '0' # De-activate vLLM V1 Inference Engine

def create_test_prompts(
        task_name: str,
        n_questions: int,
        n_samples: int) -> list[str]:
    """Create a list of test prompts with their sampling parameters."""
    task = tasks.get_task(task_name)
    dataset = task.get_dataset()

    # Take the first 10 questions in HumanEval dataset for test
    test_prompts = [task.get_prompt(dataset[i]) for i in range(n_questions) for _ in range(n_samples)] 

    return test_prompts

def add_requests(engine: LLMEngine, prompts, sampling_params: SamplingParams):
    request_id = 0
    while prompts:
        prompt = prompts.pop(0)
        #print(f"[Add] Adding request {request_id} has {len(engine.tokenizer.encode(prompt))} tokens")
        engine.add_request(str(request_id), prompt, sampling_params)
        request_id += 1

def run_engine(
        llm_engine: LLMEngine,
        use_tqdm: bool
    ) -> list[Union[RequestOutput, PoolingRequestOutput]]:
    # Initialize tqdm.
    if use_tqdm:
        num_requests = llm_engine.get_num_unfinished_requests()
        pbar = tqdm(
            total=num_requests,
            desc="Processed prompts",
            dynamic_ncols=True,
            postfix=(f"est. speed input: {0:.2f} toks/s, "
                        f"output: {0:.2f} toks/s"),
        )

    # Run the engine.
    outputs: list[Union[RequestOutput, PoolingRequestOutput]] = []
    total_in_toks = 0
    total_out_toks = 0
    while llm_engine.has_unfinished_requests():
        step_outputs = llm_engine.step()
        for output in step_outputs:
            if output.finished:
                outputs.append(output)
                if use_tqdm:
                    if isinstance(output, RequestOutput):
                        # Calculate tokens only for RequestOutput
                        n = len(output.outputs)
                        assert output.prompt_token_ids is not None
                        total_in_toks += len(output.prompt_token_ids) * n
                        in_spd = total_in_toks / pbar.format_dict["elapsed"]
                        total_out_toks += sum(
                            len(stp.token_ids) for stp in output.outputs)
                        out_spd = (total_out_toks /
                                    pbar.format_dict["elapsed"])
                        pbar.postfix = (
                            f"est. speed input: {in_spd:.2f} toks/s, "
                            f"output: {out_spd:.2f} toks/s")
                        pbar.update(n)
                    else:
                        pbar.update(1)

    if use_tqdm:
        pbar.close()

    # Sort the outputs by request ID.
    # This is necessary because some requests may be finished earlier than
    # its previous requests.
    return sorted(outputs, key=lambda x: int(x.request_id))

def main(args: argparse.Namespace):
    config_path = os.path.join(args.model, "config.json")
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        if "quantization_config" in config:
            if "quant_method" in config["quantization_config"]:
                quant_method = config["quantization_config"]["quant_method"]
            else: quant_method = None
        else:
            quant_method = None

        """
        if quant_method == 'awq':
            # Avoid error with 
            # Source: https://github.com/vllm-project/vllm/issues/5376
            print("Changing awq attention backend to Xformers instead of FlashAttention")
            os.environ["export VLLM_ATTENTION_BACKEND"]="XFORMERS"
        """
        if quant_method == 'hqq':
            print("Changing hqq backend for vLLM inference")
            from hqq.utils.vllm import set_vllm_hqq_backend, VLLM_HQQ_BACKEND
            #set_vllm_hqq_backend(backend=VLLM_HQQ_BACKEND.GEMLITE)
            set_vllm_hqq_backend(backend=VLLM_HQQ_BACKEND.PYTORCH)

            # It is suggested to load HQQ model in float16 precision for Gemlite backend and bfloat16 for Torchao's tiny_gemm backend
            # Source: https://github.com/mobiusml/hqq?tab=readme-ov-file#optimized-inference
            args.dtype = 'float16'
    except FileNotFoundError:
        print(f"Config file not found at {config_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error parsing JSON in {config_path}")
        return None

    # Initialized engine
    engine_args = EngineArgs(
        model = args.model,
        max_num_seqs = args.max_num_seqs,
        max_model_len = args.max_model_len,
        enforce_eager = args.enforce_eager,
        trust_remote_code= args.trust_remote_code,
        dtype = args.dtype
        #gpu_memory_utilization= args.gpu_memory_utilization if args.gpu_memory_utilization else "auto",
    )
    engine = LLMEngine.from_engine_args(engine_args)

    # Import task
    task = tasks.get_task(args.tasks)
    task_name = task.name
    dataset = task.get_dataset()
    n_tasks = len(dataset)
    
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        #stop=task.stop_words,
        ignore_eos=args.no_stop,
    )

    # Initialize the energy monitor
    if not args.no_monitor:
        energy_dir = os.path.join(os.path.curdir, 'data')
        os.makedirs(energy_dir, exist_ok=True)
        energy_monitor = EnergyMonitor(
            gpu_indices=None,
            log_file=os.path.join(energy_dir, f'{os.path.basename(args.model)}_mns{args.max_num_seqs}_max-tokens{args.max_tokens}.csv')
        )
        energy_monitor.begin_window(key=f'{task_name}')
    """
    # EXP1: Different max-tokens values
    # Adding requests
    prompts = create_test_prompts(task_name=task_name, n_questions=n_tasks, n_samples=args.n_samples)
    add_requests(engine, prompts, sampling_params)
    """
    # EXP2: Different nb input values
    prompts = create_test_prompts(task_name=task_name, n_questions=1, n_samples=args.n_samples) # Duplicated the first question of HumanEval for n_samples {512, 1024, 2048} times
    add_requests(engine, prompts, sampling_params)
    
    # Generation    
    outputs = run_engine(
        llm_engine=engine,
        use_tqdm=True
    )
    generated_outputs = engine.validate_outputs(outputs, RequestOutput)

    if not args.no_monitor:
        measurement:Measurement = energy_monitor.end_window(
            key=f'{task_name}',
            generated_outputs=generated_outputs
        )
    
    # Extract information in list
    if generated_outputs is not None:
        list_num_in_tokens = [len(generated_outputs[i].prompt_token_ids) for i in range(len(generated_outputs))]
        list_num_out_tokens = [len(generated_outputs[i].outputs[0].token_ids) for i in range(len(generated_outputs))]
        # Collect the first generated token time of the first served sequence in the batch
        list_first_token_time = [generated_outputs[i].metrics.first_token_time for i in range(len(generated_outputs))]
        # Collect the end time of the batch
        list_end_time = [generated_outputs[i].metrics.finished_time for i in range(len(generated_outputs))]

    # Save outputs
    with open(os.path.join(os.path.curdir, 'data/outputs.jsonl'), 'wb') as fp:
        fp.write(
                    f"start_time,window_name,elapsed_time,{','.join(map(lambda i: f'gpu{i}_energy', energy_monitor.gpu_indices))},num_in_tokens,num_out_tokens,first_token_time,finished_time\n"
                )

        for output in generated_outputs:
            fp.write((json.dumps(output.outputs[0].text) + "\n").encode('utf-8'))

if __name__ == "__main__":

    args = parse_args()
    main(args)