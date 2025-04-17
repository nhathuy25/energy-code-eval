from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvalArguments:
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
    n_samples: Optional[int] = field(
        default=1,
        metadata={"help": "Number of completions to generate for each sample."},
    )
    eos: Optional[str] = field(
        default="<|endoftext|>", metadata={"help": "end of sentence token."}
    )
    seed: Optional[int] = field(
        default=None, metadata={"help": "Random seed used for evaluation."}
    )
    
