#!/bin/bash
#SBATCH --job-name="corr_hee"                 # Job name
#SBATCH --cpus-per-gpu=5
#SBATCH --gpus-per-node=GA100:1
#SBATCH --constraint=gpu_mem_80
#SBATCH --mem=10GB                      # Memory per node
#SBATCH --time=8:00:00                # Time limit set to 8hrs
#SBATCH --output=slurm-%A_%a.out

### This script is used to run the correctness evaluation of code summerization task with HumanEvalExplain
echo "START TIME: $(date)"
nvidia-smi

### Configuration
CONTAINER=dockerproxy.repos.tech.orange_vllm_vllm-openai_v0.8.4
# Mount the host directory to the container
# !Note: /datasets used for saved models and it is ReadOnly 
CONTAINER_MOUNTS=/opt/marcel-c3/workdir/shvm6927/workdir/:/workdir,\
/opt/marcel-c3/dataset/shvm6927:/datasets
CONTAINER_WORKDIR=/workdir
CONTAINER_DATASETS=/datasets

# Language [python, java, js]
# tasks = [humanevalexplaindescribe-python,humanevalexplaindescribe-java,humanevalexplaindescribe-javascript]
# Update 30/06: we only execute code summarization task in python language
LANGUAGE=python

# Sampling temperature for summary generation
MODEL_TEMP=$1
MODEL_TOP_P=$2 
MODEL_MAXTOKENS=2000
DATASET_NUM_SAMPLE=20

# Launching jobs in array=1-<nb. of models> - we can execute multiple elemental jobs (each model is a sub-job)
# sharing the same configurations
# Models name extracted from models.txt to execute jobs in array
MODEL=$CONTAINER_DATASETS/$(sed -n "${SLURM_ARRAY_TASK_ID}p" models.txt)
# Else: pass as arguments to the script (uncomment this line and comment the above line)
# MODEL=$3

# Experiment results path
RESULT_DIR="$CONTAINER_WORKDIR/energy-code-eval/results/correctness_hee"
if [ "$MODEL_TEMP" = "0" ]; then
    RESULT_PATH="$RESULT_DIR/greedy"
elif [ "$MODEL_TEMP" = "0.8" ]; then
    RESULT_PATH="$RESULT_DIR/nucleus"
else
    RESULT_PATH="$RESULT_DIR/mix"
fi

echo "Saving generations and evaluate at ${RESULT_PATH}"

SRUN_ARGS="\
--container=$CONTAINER \
--container-mounts=$CONTAINER_MOUNTS \
--container-workdir=$CONTAINER_WORKDIR \
"

DESCRIBE="python3 $CONTAINER_WORKDIR/energy-code-eval/main.py \
	--model $MODEL \
	--tasks humanevalexplaindescribe-${LANGUAGE} \
	--n_samples $DATASET_NUM_SAMPLE \
	--temperature $MODEL_TEMP \
	--top_p $MODEL_TOP_P \
	--max_tokens $MODEL_MAXTOKENS \
	--max_num_seqs 128 \
	--allow_code_execution \
	--trust_remote_code \
	--enforce_eager \
	--max_model_len 16384 \
	--num_scheduler_steps 1 \
	--enable_chunked_prefill False \
	--save_generations \
	--save_generations_path $RESULT_PATH/generations/${MODEL_NAME}.json \
	--save_monitoring_folder $RESULT_PATH \
	--generation_only"

# To generate code from synthesized summary, using greedy search for 6 different evaluate models
SYNTHESIZE="python3 $CONTAINER_WORKDIR/energy-code-eval/main.py \
	--model $CONTAINER_DATASETS/Codestral-22B-v0.1 \
	--tasks humanevalexplainsynthesize-${LANGUAGE} \
	--n_samples 2 \
	--temperature 0 \
	--top_p 1 \
	--max_tokens 512 \
	--allow_code_execution \
	--enforce_eager \
	--max_model_len 16384 \
	--num_scheduler_steps 1 \
	--enable_chunked_prefill False \
	--max_num_seqs 128 \
	--load_data_path $RESULT_PATH/generations/${MODEL_NAME}_humanevalexplaindescribe-${LANGUAGE}.json \
	--trust_remote_code \
	--no_monitor \
	--save_monitoring_folder $RESULT_PATH"

SYNTHESIZE2="python3 $CONTAINER_WORKDIR/energy-code-eval/main.py \
	--model $CONTAINER_DATASETS/deepseek-coder-33b-instruct \
	--tasks humanevalexplainsynthesize-${LANGUAGE} \
	--n_samples 2 \
	--temperature 0 \
	--top_p 1 \
	--max_tokens 512 \
	--allow_code_execution \
	--enforce_eager \
	--max_model_len 16384 \
	--num_scheduler_steps 1 \
	--enable_chunked_prefill False \
	--max_num_seqs 128 \
	--load_data_path $RESULT_PATH/generations/${MODEL_NAME}_humanevalexplaindescribe-${LANGUAGE}.json \
	--trust_remote_code \
	--no_monitor \
	--save_monitoring_folder $RESULT_PATH"

SYNTHESIZE3="python3 $CONTAINER_WORKDIR/energy-code-eval/main.py  \
	--model $CONTAINER_DATASETS/deepseek-coder-6.7b-instruct \
	--tasks humanevalexplainsynthesize-${LANGUAGE} \
	--n_samples 2 \
	--temperature 0 \
	--top_p 1 \
	--max_tokens 512 \
	--allow_code_execution \
	--enforce_eager \
	--max_model_len 16384 \
	--num_scheduler_steps 1 \
	--enable_chunked_prefill False \
	--max_num_seqs 128 \
	--load_data_path $RESULT_PATH/generations/${MODEL_NAME}_humanevalexplaindescribe-${LANGUAGE}.json \
	--trust_remote_code \
	--no_monitor \
	--save_monitoring_folder $RESULT_PATH"

SYNTHESIZE4="python3 $CONTAINER_WORKDIR/energy-code-eval/main.py  \
	--model $CONTAINER_DATASETS/DeepSeek-Coder-V2-Lite-Instruct \
	--tasks humanevalexplainsynthesize-${LANGUAGE} \
	--n_samples 2 \
	--temperature 0 \
	--top_p 1 \
	--max_tokens 512 \
	--allow_code_execution \
	--enforce_eager \
	--max_model_len 16384 \
	--num_scheduler_steps 1 \
	--enable_chunked_prefill False \
	--max_num_seqs 128 \
	--load_data_path $RESULT_PATH/generations/${MODEL_NAME}_humanevalexplaindescribe-${LANGUAGE}.json \
	--trust_remote_code \
	--no_monitor \
	--save_monitoring_folder $RESULT_PATH"

SYNTHESIZE5="python3 $CONTAINER_WORKDIR/energy-code-eval/main.py  \
	--model $CONTAINER_DATASETS/CodeLlama-34b-Instruct-hf \
	--tasks humanevalexplainsynthesize-${LANGUAGE} \
	--n_samples 1 \
	--temperature 0 \
	--top_p 1 \
	--max_tokens 512 \
	--allow_code_execution \
	--enforce_eager \
	--max_model_len 16384 \
	--num_scheduler_steps 1 \
	--enable_chunked_prefill False \
	--max_num_seqs 128 \
	--load_data_path $RESULT_PATH/generations/${MODEL_NAME}_humanevalexplaindescribe-${LANGUAGE}.json \
	--trust_remote_code \
	--no_monitor \
	--save_monitoring_folder $RESULT_PATH"

SYNTHESIZE6="python3 $CONTAINER_WORKDIR/energy-code-eval/main.py  \
	--model $CONTAINER_DATASETS/CodeLlama-7b-Instruct-hf \
	--tasks humanevalexplainsynthesize-${LANGUAGE} \
	--n_samples 1 \
	--temperature 0 \
	--top_p 1 \
	--max_tokens 512 \
	--allow_code_execution \
	--enforce_eager \
	--max_model_len 16384 \
	--num_scheduler_steps 1 \
	--enable_chunked_prefill False \
	--max_num_seqs 128 \
	--load_data_path $RESULT_PATH/generations/${MODEL_NAME}_humanevalexplaindescribe-${LANGUAGE}.json \
	--trust_remote_code \
	--no_monitor \
	--save_monitoring_folder $RESULT_PATH"
	
echo $DESCRIBE
echo $SYNTHESIZE

# Run generation on datasets
srun  \
  $SRUN_ARGS \
  /bin/bash -c "pip install -e energy-code-eval; \
  $DESCRIBE; $SYNTHESIZE; $SYNTHESIZE2; $SYNTHESIZE3; $SYNTHESIZE4; $SYNTHESIZE5; $SYNTHESIZE6"

echo "END TIME: $(date)"