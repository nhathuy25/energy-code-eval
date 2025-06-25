#!/bin/bash
#SBATCH --job-name="gpu"                 # Job name
#SBATCH --cpus-per-gpu=2
#SBATCH --gpus-per-node=GA100:1
#SBATCH --constraint=gpu_mem_80
#SBATCH --mem=5GB                      # Memory per node
#SBATCH --time=15:00:00                # Time limit set to 15hrs
#SBATCH --output=slurm-%j.out

echo "START TIME: $(date)"

### Configuration
CONTAINER=vllm-openai_latest  
# Mount the host directory to the container
# !Note: /datasets used for saved models and it is ReadOnly 
CONTAINER_MOUNTS=$(dirname "$PWD"):/workdir,/opt/marcel-c3/dataset/shvm6927:/datasets
CONTAINER_WORKDIR=/workdir
DATASET_DIR=$CONTAINER_WORKDIR/energy-code-eval

# Experiment variable - change here for each experiment
# MODEL_NAME = [codellama7i, codellama34i, codestral, deepseek_base, deepseek_instruct]
# PROMPT = [instruct, codellama, deepseek, codestral]
MODEL_NAME=$1
PROMPT=$2

# Tasks
TASKS=humaneval,mbpp,codesearchnet-python,codesearchnet-java,codesearchnet-javascript

# Sampling temperature
MODEL_TEMP=0.0
MODEL_TOP_P=1
MODEL_MAXTOKENS=512
DATASET_NUM_SAMPLE=20
GENERATIONS_PATH="$DATASET_DIR/results/slurm_04-07"

echo "Saving generations and evaluate at ${GENERATIONS_PATH}"

SRUN_ARGS="\
--container=$CONTAINER \
--container-workdir=$CONTAINER_WORKDIR \
"

CMD="python3 $DATASET_DIR/main.py \
	--model $MODEL_NAME \
	--prompt $PROMPT \
	--tasks $TASKS \
	--n_samples 1 \
	--temperature $MODEL_TEMP \
	--top_p $MODEL_TOP_P \
	--max_tokens $MODEL_MAXTOKENS \
	--allow_code_execution \
	--save_generations \
	--save_generations_path $GENERATIONS_PATH/${MODEL_NAME}_k${MODEL_TEMP}_top-p${MODEL_TOP_P}.json \
	--save_references \
	--save_references_path references_${MODEL_NAME}.json \
	--trust_remote_code \
  	--metric_output_path $GENERATIONS_PATH/metric_${MODEL_NAME}.json
"

echo $CMD

# Run generation on datasets
srun  \
  $SRUN_ARGS \
  /bin/bash -c "pip install -e energy-code-eval; \
  $CMD"

echo "END TIME: $(date)"