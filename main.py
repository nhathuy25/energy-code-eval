import os
import fnmatch
import json
import warnings

import datasets
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.config import ModelConfig

from code_eval.arguments import EvalArguments
from code_eval.evaluator import Evaluator
from code_eval.tasks import ALL_TASKS

MODEL_NAME_TO_LOCAL_DIR = {
    "codellama7": '/workdir/models/CodeLlama-7b-hf',
    "codellama7i": '/workdir/models/CodeLlama-7b-Instruct-hf',
    "codellama34i" : '/workdir/models/CodeLlama-34b-Instruct-hf',
    "deepseek_base" : '/workdir/models/DeepSeek-Coder-V2-Lite-Base',
    "deepseek_instruct" : '/workdir/models/DeepSeek-Coder-V2-Lite-Instruct',
    "codestral" : '/workdir/models/Codestral-22B-v0.1',
}

def parse_args():
    parser = HfArgumentParser(EvalArguments)

    parser.add_argument(
        "--model",
        default="CodeLlama-7b",
        help="Model to evaluate, provide a repo name in Hugging Face hub or a local path",
    )
    parser.add_argument(
        "--peft_model",
        type=str,
        default=None,
        help="Adapter to the PEFT base model. Can be utilized for loading PEFT adapters such as a LoRA trained model. The --model parameter needs to be the base model.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Model revision to use",
    )
    parser.add_argument(
        "--use_auth_token",
        action="store_true",
        help="Use the token generated when running `huggingface-cli login` (necessary for private model).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Use a model with custom code, this requires executing code by the author of the model.",
    )
    parser.add_argument(
        '--enforce_eager',
        action='store_true',
        help='Always use eager-mode PyTorch. If False, '
        'will use eager mode and CUDA graph in hybrid '
        'for maximal performance and flexibility.'
    )
    parser.add_argument(
        '--max_model_len',
        type=int,
        default=EngineArgs.max_model_len,
        help='Model context length. If unspecified, will '
        'be automatically derived from the model config.'
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help=f"Evaluation tasks from humaneval or mbpp",
    )
    parser.add_argument(
        "--instruction_tokens",
        default=None,
        help="A series of instruction tokens used for instruction-tuning benchamrks separated by comma e.g. <user_message>,<end_user_message>,<assistant_message>",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation on each worker, can be larger for HumanEval",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="The number of GPUs to use for distributed execution with tensor parallelism",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=EngineArgs.dtype,
        choices=[
            'auto', 'half', 'float16', 'bfloat16', 'float', 'float32'
        ],
        help='Data type for model weights and activations.\n\n'
        '* "auto" will use FP16 precision for FP32 and FP16 models, and '
        'BF16 precision for BF16 models.\n'
        '* "half" for FP16. Recommended for AWQ quantization.\n'
        '* "float16" is the same as "half".\n'
        '* "bfloat16" for a balance between precision and range.\n'
        '* "float" is shorthand for FP32 precision.\n'
        '* "float32" for FP32 precision.'
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8bit",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4bit",
    )
    parser.add_argument(
        "--left_padding",
        action="store_true",
        help="Force left padding, needed for models like chatglm3-6b",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of samples to solve and evaluate from the benchmark",
    )
    parser.add_argument(
        "--limit_start",
        type=int,
        default=0,
        help="Optional offset to start from when limiting the number of samples",
    )
    parser.add_argument(
        "--save_every_k_tasks",
        type=int,
        default=-1,
        help="Optional saving after every k tasks",
    )
    parser.add_argument(
        "--postprocess",
        action="store_false",
        help="Postprocess model outputs before execution, always on except during generation tests",
    )
    parser.add_argument(
        "--allow_code_execution",
        action="store_true",
        help="Allow code evaluation to execute external/untrusted Python code on your machine",
    )
    parser.add_argument(
        "--generation_only",
        action="store_true",
        help="Do code generation but no evaluation",
    )
    parser.add_argument(
        "--load_generations_path",
        type=str,
        default=None,
        help="Path of file with previously generated solutions, if provided generation is skipped and only evaluation is done",
    )
    parser.add_argument(
        "--load_data_path",
        type=str,
        default=None,
        help="Path of additional data to load for the tasks",
    )
    parser.add_argument(
        "--metric_output_path",
        type=str,
        default="evaluation_results.json",
        help="Path to save the results",
    )
    parser.add_argument(
        "--save_generations",
        action="store_true",
        help="Whether to save code generations",
    )
    parser.add_argument(
        "--load_generations_intermediate_paths",
        type=str,
        nargs="*",
        help="List of paths for saving the intermediate code generations",
    )
    parser.add_argument(
        "--save_generations_path",
        type=str,
        default="generations.json",
        help="Path for saving the code generations",
    )
    parser.add_argument(
        "--save_references",
        action="store_true",
        help="Whether to save reference solutions/tests",
    )
    parser.add_argument(
        "--save_references_path",
        type=str,
        default="references.json",
        help="Path for saving the references solutions/tests",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="prompt",
        help="Prompt type to use for generation in HumanEvalPack tasks",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=str,
        default=None,
        help="Max memroy to allocate per gpu, you can also use 'auto'",
    )
    parser.add_argument(
        "--check_references",
        action="store_true",
        help="Don't run generation but benchmark groundtruth (useful for debugging)",
    )
    return parser.parse_args()


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

    # April 2025: Use the V0 version of vLLM since V1 is still experimental
    os.environ['VLLM_USE_V1'] = '0'

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
        }
        
        if args.model in MODEL_NAME_TO_LOCAL_DIR:
            args.model = MODEL_NAME_TO_LOCAL_DIR[args.model]
    
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
            set_vllm_hqq_backend(backend=VLLM_HQQ_BACKEND.GEMLITE)

            from hqq.utils.vllm import set_vllm_onthefly_hqq_quant
            set_vllm_onthefly_hqq_quant(weight_bits=8, group_size=None, quant_mode='dynamic', skip_modules=['lm_head']) #dynamic A8W8
            
            model_kwargs['dtype'] = torch.float16
            model = LLM(args.model, enforce_eager=args.enforce_eager, **model_kwargs)
            
        elif args.load_in_4bit:
            print("Loading model in 4bit - Using HQQ 4W16A")
            # Source https://github.com/mobiusml/hqq?tab=readme-ov-file#vllm

            from hqq.utils.vllm import set_vllm_hqq_backend, VLLM_HQQ_BACKEND
            set_vllm_hqq_backend(backend=VLLM_HQQ_BACKEND.GEMLITE)

            from hqq.utils.vllm import set_vllm_onthefly_hqq_quant
            set_vllm_onthefly_hqq_quant(weight_bits=4, group_size=64, quant_mode='static', skip_modules=['lm_head']) #A16W4 

            model_kwargs['dtype'] = torch.float16
            model = LLM(args.model, enforce_eager=args.enforce_eager, **model_kwargs)

        else:
            print(f"Loading model in {args.dtype}")

            model = LLM(
                args.model,
                enforce_eager = args.enforce_eager,
                **model_kwargs
                )
        
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

    # Save all args to config
    results["config"] = vars(args)
    if not args.generation_only:
        dumped = json.dumps(results, indent=2)
        print(dumped)

        with open(args.metric_output_path, "w") as f:
            f.write(dumped)

if __name__ == "__main__":
    main()
