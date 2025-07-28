import json
import time
from vllm import LLM, SamplingParams, RequestOutput
#from human_eval.data import read_problems

import os

os.environ["VLLM_USE_V1"]="0"
#os.environ["VLLM_ATTENTION_BACKEND"]="XFORMERS"

NUM_SAMPLES=20
MAX_NUM_SEQS=32

# Load prompts
from code_eval import tasks
task = tasks.get_task('humaneval')
dataset = task.get_dataset()
prompts = []
for i in range(len(dataset)):
    prompt = task.get_prompt(dataset[i])
    for _ in range(NUM_SAMPLES):
        prompts.append(prompt)
"""
prompts=[]
problems = read_problems()
for task_id, task in problems.items():
    prompt = task["prompt"]
    for _ in range(NUM_SAMPLES):
       prompts.append(prompt)
"""
#model="/datasets/models--TheBloke--deepseek-coder-6.7B-instruct-AWQ/snapshots/502ae3e19e57ae78dc30a791ba33c565da72dc62"
#model="/datasets/models--deepseek-ai--DeepSeek-Coder-6.7B-Instruct/snapshots/e5d64addd26a6a1db0f9b863abf6ee3141936807"
#model="/datasets/models--TheBloke--deepseek-coder-6.7B-instruct-GPTQ/snapshots/13ccea6e3a43dcfdcb655d92097610018b431a17"
model = '/datasets/deepseek-coder-6.7B-instruct-GPTQ'

llm=LLM(model=model,trust_remote_code=True,max_num_seqs=MAX_NUM_SEQS,max_model_len=16384,enforce_eager=True,enable_chunked_prefill=False)

sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=512, ignore_eos=True)

start_time = time.perf_counter()
outputs = llm.generate(prompts, sampling_params)
end_time = time.perf_counter()

output_tokens=0
inputs_tokens=0
for request_output in outputs:
    output_tokens+=len(request_output.outputs[0].token_ids)
    inputs_tokens+=len(request_output.prompt_token_ids)
   
print(f'Nb. of input tokens: {inputs_tokens}')
print(f'Nb. of output tokens: {output_tokens}')
print(f"TPS: {output_tokens/(end_time-start_time):.2f} token/s")
