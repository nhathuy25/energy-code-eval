from dataclasses import dataclass, field
import fnmatch
from typing import Optional
from transformers import HfArgumentParser
import argparse

from code_eval.tasks import ALL_TASKS
from vllm.engine.arg_utils import EngineArgs


@dataclass
class GenerationArguments:
    """
    Configuration for running the evaluation.
    """
    prefix: Optional[str] = field(
        default="",
        metadata={
            "help": "Prefix to add to the prompt. For example InCoder needs prefix='<| file ext=.py |>\n'"
        },
    )
    temperature: Optional[float] = field(
        default=0.0, metadata={"help": "Sampling temperature used for generation."}
    )
    top_k: Optional[int] = field(
        default=-1, metadata={"help": "Top-k parameter used for generation."}
    )
    top_p: Optional[float] = field(
        default=1, metadata={"help": "Top-p parameter used for nucleus sampling."}
    )
    max_tokens: Optional[int] = field(
        default=256, metadata={"help": "Maximum length of generated sequence."}
    )
    eos: Optional[str] = field(
        default="<|endoftext|>", metadata={"help": "end of sentence token."}
    )
    seed: Optional[int] = field(
        default=None, metadata={"help": "Random seed used for evaluation."}
    )
    n_samples: Optional[int] = field(
        default=1,
        metadata={"help": "Number of completions to generate for each sample."}
    )
    batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "Batch size for evaluation on each worker, can be larger for HumanEval"}
    )
    no_stop: bool = field(
        default=False,
        metadata={
            "help": "Not use stop words for interrupting generation"
        }
    )


@dataclass
class SchedulerArguments(argparse.ArgumentParser):
    """
    Configuration for vLLM's scheduler.
    """
    max_num_seqs: Optional[int] = field(
        default=None,
        metadata={
            "help": "Maximum number of sequences to generate in parallel."
        }
    )
    enable_chunked_prefill: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Enable chunk prefill."
        }
    )
    num_scheduler_steps: Optional[int] = field(
        default=1,
        metadata={"help": "Enable chunk prefill."}
    )

@dataclass
class MonitorArguments(argparse.ArgumentParser):
    """
    Configuration for energy monitoring.
    """
    gpu_indices: Optional[str] = field(
        default=None,
        metadata={
            "help": "GPU indices to monitor. If not specified, all GPUs will be monitored."
        }
    )
    cpu_indices: Optional[str] = field(
        default=None,
        metadata={
            "help": "CPU indices to monitor. If not specified, all CPUs will be monitored."
        }
    )
    no_monitor: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Disable energy & power monitoring."
        }
    )
    update_period: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "Update period for power monitoring."
        }
    )


@dataclass
class CombinedArguments(GenerationArguments, SchedulerArguments, MonitorArguments):
    """Combined arguments for both evaluation and scheduler."""
    pass
    

class MultiChoice:
    def __init__(self, choices):
        self.choices = choices

    # Simple wildcard support (linux filename patterns)
    def __contains__(self, values):
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                return False

        return True

    def __iter__(self):
        for choice in self.choices:
            yield choice



def parse_args():
    parser = HfArgumentParser(CombinedArguments)

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
        choices=MultiChoice(ALL_TASKS),
        help=f"Evaluation tasks from {ALL_TASKS}",
    )
    parser.add_argument(
        "--instruction_tokens",
        default=None,
        help="A series of instruction tokens used for instruction-tuning benchamrks separated by comma e.g. <user_message>,<end_user_message>,<assistant_message>",
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
        choices=[
            'instruct', 'deepseek', 'codellama', 'codestral'
        ],
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
