import json
from math import ceil

from typing import List, Optional
import os

from torch.utils.data.dataloader import DataLoader
from transformers import StoppingCriteria, StoppingCriteriaList

from code_eval.utils import TokenizedDataset, complete_code
from code_eval.monitor import EnergyMonitor, Measurement, PowerMonitor

"""
Huy's note: This 2 classes below is originally from bigcode-evaluation-harness, for adding custom stopping criteria with HF inference. 
However, it is not used with vLLM inference since vLLM has integrated stopping words inside generate(stop=[], eos_token=[]) method.
I keep them here in case we reuse HF Transformers inference to compare with vLLM in the report.
"""
class EndOfFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""
    def __init__(self, start_length, eof_strings, tokenizer, check_fn=None):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer
        if check_fn is None:
            check_fn = lambda decoded_generation: any(
                [stop_string in decoded_generation for stop_string in self.eof_strings]
            )
        self.check_fn = check_fn

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(input_ids[:, self.start_length :])
        return all([self.check_fn(decoded_generation) for decoded_generation in decoded_generations])

class TooLongFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if the generated function is too long by a certain multiplier based on input length."""

    def __init__(self, input_length, multiplier):
        self.input_length = input_length
        self.multiplier = multiplier

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if generated sequence is too long."""
        return input_ids.shape[1] > int(self.input_length * self.multiplier)
        
def parallel_generations(
        task,
        dataset,
        model,
        tokenizer,
        n_tasks,
        args,
        curr_sample_idx: int = 0,
        save_every_k_tasks: int = -1,
        intermediate_generations: Optional[List[Optional[List[Optional[str]]]]] = None,
        intermediate_save_generations_path: Optional[str] = None,
        energy_monitor: Optional[EnergyMonitor] = None,
):
    """
    Set up the dataset loader from task and corresponding dataset for generation.
    Define generation settings before calling `complete_code` to generate.
    - temperature, top_p, top_k, max_tokens     : for generation configurations
    - stop_words, stop_token_ids                : for stopping criteria
    - ignore_eos                                : to ignore eos tokens for generation to maximum number of tokens
    """ 
    if args.load_generations_path:
        # load generated code
        with open(args.load_generations_path) as fp:
            generations = json.load(fp)
            print(
                f"generations loaded, {n_tasks} selected from {len(generations)} with {len(generations[0])} candidates"
            )
        return generations[:n_tasks]
    # Setup generation settings
    gen_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "stop_token_ids" : [tokenizer.eos_token_ids]
    }

    # Check if the task has a custom check_fn method for the stopping criteria
    if task.stop_words and tokenizer.eos_token:
        task.stop_words.append(tokenizer.eos_token)   

    if not args.no_stop:
        gen_kwargs["stop"] = task.stop_words 
    else:
        gen_kwargs["ignore_eos"] = True

    # Instruction tokens is a set of 3 tokens: begin_token, end_token, assistant_token
    # TODO: define the use for instruction tokens
    if args.instruction_tokens:
        instruction_tokens = args.instruction_tokens.split(",")
        if len(instruction_tokens) != 3:
            raise ValueError(
                "Instruction tokens should contain exactly 3 tokens separated by a comma. If a token is empty, represent it as ''"
            )
        for token in instruction_tokens:
            if token.strip() != "":
                task.stop_words.append(token)
    else:
        instruction_tokens = None


    # Define dataset loader for each line of task
    ds_tokenized = TokenizedDataset(
        task,
        dataset,
        tokenizer,
        num_devices=args.tensor_parallel_size, #os.environ('SLURM_GPUS_PER_NODE'), TODO: define later for detect GPUs on current node
        max_length=args.max_tokens,
        limit_start=args.limit_start + curr_sample_idx,
        n_tasks=n_tasks,
        n_copies=args.n_samples,
        prefix=args.prefix,
        instruction_tokens=instruction_tokens
        )
    
    # Load dataset using torch.DataLoader
    """
    Huy's note: batch size is set by default to None, which means the whole dataset will be load in one batch and vLLM's scheduler
    will handle the inference with max_num_seqs and other parameters.
    If batch size is set, the input sequences with be loaded in batches and sent to vLLM engine group by group (one after another) of batch size 
        - this reduce the throughput of vLLM inference since it limit the use of iteration-level scheduling.
    """
    if args.batch_size:
        ds_loader = DataLoader(ds_tokenized, batch_size=args.batch_size)
    else:
        # Load in one batch and execute all sequences in one go with vLLM
        ds_loader = DataLoader(ds_tokenized, batch_size=len(ds_tokenized))

    generations = complete_code(
        task,
        model,
        tokenizer,
        ds_loader,
        n_tasks=n_tasks,
        limit_start=args.limit_start + curr_sample_idx,
        batch_size=args.batch_size,
        prefix=args.prefix,
        instruction_tokens=instruction_tokens,
        postprocess=args.postprocess,
        save_every_k_tasks=save_every_k_tasks,
        intermediate_generations=intermediate_generations,
        intermediate_save_generations_path=intermediate_save_generations_path,
        energy_monitor=energy_monitor,
        **gen_kwargs,
    )
    

    return generations