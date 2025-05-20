import json
import math
import re
import warnings
from collections import defaultdict
from typing import List, Optional, Iterable, Dict

import torch
from torch.utils.data import IterableDataset
from tqdm import tqdm

from vllm import SamplingParams
from vllm.outputs import RequestOutput

from pynvml import *
from time import time, sleep

INFILL_MODE = False
INSTRUCTION_MODE = False


class TokenizedDataset(IterableDataset):
    """Tokenize and preprocess the dataset
    Multiple copies of the same prompt are sent sequentially. See compute_code for more details.
    The prompt can either be:
    - one prompt: normal code completion
    - two prompts: for infilling mode (prefix, suffix) or instructin-tuning mode (instruction, context)
    """

    def __init__(
        self,
        task,
        dataset,
        tokenizer,
        num_devices,
        max_length,
        limit_start=0,
        n_tasks=None,
        n_copies=1,
        prefix="",
        instruction_tokens=None,
    ):
        self.task = task
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.num_devices = num_devices
        self.max_length = max_length
        self.limit_start = limit_start
        self.n_tasks = n_tasks
        self.n_copies = n_copies
        self.prefix = prefix
        self.instruction_tokens = instruction_tokens

    def __iter__(self):
        prompts = []
        infill = []
        instruction = []
        for sample in range(self.limit_start, self.limit_start + self.n_tasks):
            prompt_contents = self.task.get_prompt(self.dataset[sample])
            if isinstance(prompt_contents, str):
                # Normal code completion mode
                infill.append(False)
                instruction.append(False)
                prompt = self.prefix + prompt_contents
            elif isinstance(prompt_contents, dict):
                pass
                # TODO: for code infilling and instruction-tuning mode
            else:
                raise ValueError(f"Unsupported prompt format: {type(prompt_contents)}")
            
            # Create list of prompts from the dataset 
            prompts.append(prompt)

        global INFILL_MODE
        global INSTRUCTION_MODE
        INFILL_MODE = infill[0]
        INSTRUCTION_MODE = instruction[0]
        if INFILL_MODE:
            return_token_type_ids = False
        else:
            return_token_type_ids = None  # default

        outputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt", # input_ids in tensor type might deprecated in vLLM
            max_length=self.max_length,
            return_token_type_ids=return_token_type_ids
        )

        for sample in range(self.n_tasks):
            for _ in range(self.n_copies):
                yield {
                    "content": str(prompts[sample]),
                    "ids": outputs.input_ids[sample],
                    "task_id": int(sample),
                    "input_len": sum(outputs.attention_mask[sample])
                }
        
        def _make_instruction_prompt(self, instruction, context, prefix=""):
            pass
            # TODO: Define later for prompt instruction fine-tunning

    def __len__(self):
        return int(self.n_tasks * self.n_copies)

def complete_code(
    task,
    model,
    tokenizer,
    dataloader,
    n_tasks,
    limit_start=0,
    batch_size=20,
    prefix="", 
    instruction_tokens=None,
    postprocess=True, 
    save_every_k_tasks: int = -1,
    intermediate_generations: Optional[List[Optional[List[Optional[str]]]]] = None,
    intermediate_save_generations_path: Optional[str] = None,
    **gen_kwargs
):
    """
    Complete the code base on dataset Loader, supported continuous generation and post processing step.
    Return 
    """
    code_gens: List[List[Optional[str]]] = [[] for _ in range(n_tasks)]
    generations = [] if not intermediate_generations else intermediate_generations
    gen_token_dict = defaultdict(list) # dict of list of generated tokens

    # Initialize GPU monitor
    handle = nvmlDeviceGetHandleByIndex(0)
    mesurements = []
    num_tokens_batch = 0

    for step, batch in tqdm(
        enumerate(dataloader),
        total=math.ceil(
            #n_tasks * dataloader.dataset.n_copies / num_processes # Define later for parallel distribution
            (n_tasks * dataloader.dataset.n_copies) / batch_size
        ),
    ):
        #input_tensors = [b["ids"][:, : b["input_len"]] if tokenizer.padding_side == "right" else b["ids"] for b in batch]
        # Define the generations here
        try:
            #inputs = input_tensors.cpu().detach().numpy().tolist()
            inputs = batch["content"]
            start_time = time()
            start_energy = nvmlDeviceGetTotalEnergyConsumption(handle)
            generated_outputs = model.generate(
                prompts=inputs,
                sampling_params=SamplingParams(**gen_kwargs),
                use_tqdm=True
            )
            num_tokens_batch = sum(len(generated_outputs[i].outputs[0].token_ids) for i in range(len(generated_outputs)))
            torch.cuda.synchronize()
            elapsed_time = time() - start_time
            energy = (nvmlDeviceGetTotalEnergyConsumption(handle) - start_energy) / 1000 # convert to Joules
            mesurements.append({
                "batch": step,
                "num_tokens_generated": num_tokens_batch,
                "start_time": start_time,
                "elapsed_time": elapsed_time,
                "energy": energy, 
                "energy_per_token": energy / num_tokens_batch,
            })
        except ValueError as e:
            raise e
        # -- end generation --
        generated_tasks = batch["task_id"]
        
        for sample, generated_outputs in zip(generated_tasks, generated_outputs):
            gen_token_dict[sample].append(generated_outputs)

        if save_every_k_tasks >= 1 and (step + 1) % save_every_k_tasks == 0:
            # If exist path to save intermediate generations, save the generation every 50 tasks.
            code_gens = update_code_gens(
                task,
                tokenizer,
                limit_start,
                prefix,
                instruction_tokens,
                postprocess,
                code_gens,
                gen_token_dict,
            )
            with open(intermediate_save_generations_path, "w") as fp:
                json.dump(generations + code_gens, fp)
                print(
                    f"intermediate generations were saved at {intermediate_save_generations_path}"
                )
            # reset gen_token_dict - prevent redundant decoding
            gen_token_dict = defaultdict(list)

    # Adding and organise generated outputs to 2D list 'code_gens' 
    code_gens = update_code_gens(
        task,
        tokenizer,
        limit_start,
        prefix,
        instruction_tokens,
        postprocess,
        code_gens,
        gen_token_dict
    )

    
    generations.extend(code_gens)
    return generations, mesurements

# TODO : define post processing step
def update_code_gens(
    task,
    tokenizer,
    limit_start,
    prefix,
    instruction_tokens,
    postprocess,
    code_gens,
    gen_token_dict
):
    """
    Take the generated outputs stocked in temporary 'gen_token_dict' and reorganised it in to 2D list of 'code_gens'.
    Support intermediate_generations and postprocessing step to prepare the code for evaluation.
    Original version of bigcode-evaluation-harness use this function to decode the generated tokens to string.
    """
    for sample, generated_outputs in gen_token_dict.items():
        for s in generated_outputs:
            if INFILL_MODE or tokenizer.eos_token in task.stop_words:
                pass
            # If generated code isn't in the output format of vLLM, process it
            if not isinstance(s, RequestOutput):
                gen_code = tokenizer.decode(
                    s, skip_special_tokens=True
                )
            if isinstance(s, RequestOutput):
                # Unlike HFTransformers, vLLM doesn't concatenate prompt and generated outputs, we concatenate it manually
                gen_code = generated_outputs[0].prompt + generated_outputs[0].outputs[0].text  
            else:
                raise ValueError(
                    "The outputs must be in the form of token_ids or RequestOutput of vLLM"
                )        
            if not INFILL_MODE:
                # TODO 
                pass
            if postprocess:
                code_gens[sample].append(
                    task.postprocess_generation(gen_code, int(sample) + limit_start)
                )
            else:
                warnings.warn(
                    "model output is not postprocessed, this might lower evaluation scores"
                )
                code_gens[sample].append(gen_code)
    return code_gens


def write_jsonl(mesurements: Iterable[Dict], filename='test_gpu.jsonl'):
    fp = '/workdir/energy-code-eval/results/monitoring/' + filename
    with open(fp, 'wb') as f:
        for mesurement in mesurements:
            f.write((json.dumps(mesurement) + '\n').encode('utf-8'))
            
        print(f'Measurement saved to {fp}')
