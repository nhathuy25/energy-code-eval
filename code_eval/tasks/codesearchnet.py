"""CodeSearchNet Challenge: Evaluating the State of Semantic Code Search
http://arxiv.org/abs/1909.09436


CodeSearchNet corpus is a dataset of 2 milllion (comment, code) pairs from opensource libraries hosted on GitHub. 
It contains code and documentation for several programming languages.

CodeSearchNet corpus was gathered to support the CodeSearchNet challenge, to explore the problem of code retrieval using natural language.

Homepage: https://github.com/github/CodeSearchNet
"""

from code_eval.base import Task
from .utils import remove_comments_and_docstrings
import torch

import os 

__CITATION = """
@article{Husain_Wu_Gazit_Allamanis_Brockschmidt_2020, 
        title={CodeSearchNet Challenge: Evaluating the State of Semantic Code Search}, 
        url={http://arxiv.org/abs/1909.09436}, 
        DOI={10.48550/arXiv.1909.09436}, 
        note={arXiv:1909.09436 [cs]}, number={arXiv:1909.09436}, publisher={arXiv}, 
        author={Husain, Hamel and Wu, Ho-Hsiang and Gazit, Tiferet and Allamanis, Miltiadis and Brockschmidt, Marc}, 
        year={2020}, month=jun 
}
"""

LANGUAGES = ["python", "java", "javascript"]

LANGUAGE_TO_NAME = {
    "python": "Python",
    "javascript": "JavaScript",
    "java": "Java",
}

LANGUAGE_TO_EXTENSION = {
    "python": "py",
    "cpp": "cpp",
    "javasript": "js",
    "java": "java",
    "go": "go",
    "rust": "rs",
}

# Taken from https://huggingface.co/datasets/nuprl/MultiPL-E/ & https://github.com/THUDM/CodeGeeX
LANGUAGE_TO_STOP_WORDS = {
    # https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L164
    "python": ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\nassert"],
    # https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L185
    "cpp": [],
    # https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L188
    "javascript": [],
    # https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L177
    "go": ["\n//", "\nfunc main(", "struct", "\nfunc"],
    # https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L169
    "java": [],
    "rust": [],
}

LANGUAGE_TO_TIMEOUT = {
    "python": 10,
    "cpp": 60,
    "javascript": 10,
    "java": 10,
    "go": 20,
    "rust": 300, # Necessary for first-time compilation of cargo
}

# Java sometimes fails with more workers; For JS it's twice as fast with 4 workers
LANGUAGE_TO_NUM_WORKERS = {
    "python": 4,
    "cpp": 4,
    "javascript": 4,
    "java": 1,
    "go": 4,
    "rust": 1,
}

# https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L6
IMPORT_HELPER = {
    "python": [
        "import math",
        "import re",
        "import sys",
        "import copy",
        "import datetime",
        "import itertools",
        "import collections",
        "import heapq",
        "import statistics",
        "import functools",
        "import hashlib",
        "import numpy",
        "import numpy as np",
        "import string",
        "from typing import *",
        "from collections import *",
    ],
}

def mean_sentenceBERT(tensor: torch.Tensor):
    diagonal = torch.diagonal(tensor)
    mean = torch.mean(diagonal)
    return mean.item()

def create_all_tasks():
    codesearchnet = {f"codesearchnet-{language}": create_task(language) for language in LANGUAGES}
    return {**codesearchnet}

def create_task(language):
    class CodeSearchNet(GeneralCodeSearchNet):
        def __init__(self, language=language, prompt="instruct"):
            super().__init__(language=language, prompt=prompt, with_docs=True)

    return CodeSearchNet

class GeneralCodeSearchNet(Task):
    # We use the cleaned version of CodeSearchNet by microsoft/CodeXGLUE
    # https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-text
    DATASET_PATH= "google/code_x_glue_ct_code_to_text"
    DATASET_NAME = None

    def __init__(self, language="python", prompt="instruct", with_docs=True):

        self.DATASET_NAME = language
        self.prompt = prompt # Attention, 'prompt' here refers to the type of prompt "instruct"/"codellama"/"deepseek"/... refers to different LM families
        stop_words = LANGUAGE_TO_STOP_WORDS[language]
        stop_words.append("<|endoftext|>")
        self.with_docs = with_docs        
        super().__init__(stop_words=stop_words, requires_execution=True)

    def get_dataset(self):
        """ 
        Returns dataset for the task or an iterable of any object, that get_prompt can handle.
        """
        dataset = self.dataset["test"]
        dataset = dataset.select(range(300))
        return dataset   
    
    def get_prompt_base(self, doc):
        if self.with_docs: 
            # Process the prompt
            prompt_base = remove_comments_and_docstrings(doc['original_string'], self.DATASET_NAME)
            return prompt_base
        else:
            return doc['code']

    def get_prompt(self, doc):
        """
        Builds the prompt for the LM to generate from, including:
            - base prompt (original prompt from the dataset),
            - instruction - to instruct the model complete the task
        """
        prompt_base = self.get_prompt_base(doc)
        description = f"Provide a concise natural language description of the code using at most {len(doc['docstring'])} characters."

        # If we doesn't specify the prompt for the LM family:
        if self.prompt == "instruct":
            instruction = prompt_base + '\n' + description
            return instruction.strip()
        else: instruction = description + '\n' + prompt_base

        if self.prompt == "codellama" or self.prompt == "codestral":
            # https://hf.co/codellama             
            prompt = f"[INST] {instruction.strip()} [/INST]\n"
        elif  self.prompt == "deepseek":
            prompt = f"You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. \
            For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer\n \
            ### Instruction:\n{instruction.strip()}\n### Response:\n"      
        else:
            raise ValueError(f"The --prompt argument {self.prompt} wasn't provided or isn't supported")
        return prompt.strip()
    
    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset).
        :param doc: dict[str: str]
        """
        from mosestokenizer import MosesDetokenizer
        # deactivate tokenizer parallelism when calling MosesDetokenizer TODO: do it for all refs once
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # docstring_tokens are preprocessed and don't have extra context like variable defs
        docstring = " ".join(doc["docstring_tokens"]).replace("\n", "")
        # some docstrings started with r""" before tokenization but r was kept
        if docstring[0] == "r":
            docstring = docstring[1:]
        with MosesDetokenizer("en") as detokenize:
            docstring = detokenize(docstring.strip().split())
        return docstring

    def postprocess_generation(self, generation, idx):
        doc = self.get_dataset()[idx]
        prompt = self.get_prompt(doc)
        return generation[len(prompt):]
    
    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences (not needed for APPS Task)
        """
        from sentence_transformers import SentenceTransformer, SimilarityFunction
        from bert_score import score as BERTScore

        sbert = SentenceTransformer("nomic-ai/CodeRankEmbed", trust_remote_code=True)
        
        candidates = [generation[0] for generation in generations]

        embed_refs = sbert.encode(references)
        embed_gens = sbert.encode(candidates)
                
        sbert.similarity_fn_name = SimilarityFunction.COSINE
        cosine_score = mean_sentenceBERT(sbert.similarity(embed_refs, embed_gens))

        sbert.similarity_fn_name = SimilarityFunction.EUCLIDEAN
        euclidean_score = mean_sentenceBERT(sbert.similarity(embed_refs, embed_gens))

        # BERTScore metric
        _, _, f1 = BERTScore(references, candidates, lang = 'en')
        f1_score = (torch.mean(f1)).item()

        results = {
            'cosine' : cosine_score,
            'euclidean' : euclidean_score,
            'BERTScore' : f1_score,
        }

        return results



