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
CONTAINER_IMAGE=vllm/vllm-openai:latest
CONTAINER_NAME=vllm                   
CONTAINER_MOUNTS=$(dirname "$PWD"):/workdir
CONTAINER_WORKDIR=/workdir
DATASET_DIR=$CONTAINER_WORKDIR/energy-code-eval

# Experiment variable - change here for each experiment
# MODEL_NAME = [codellama7i, codellama34i, codestral, deepseek_base, deepseek_instruct]
# PROMPT = [instruct, codellama, deepseek]
MODEL_NAME=$1
PROMPT=$2

# Language [python, java, js]
LANGUAGE=$3

# Sampling temperature for summary generation
MODEL_TEMP=0.8
MODEL_TOP_P=0.95 # Nucleus sampling
MODEL_MAXTOKENS=1000
DATASET_NUM_SAMPLE=5
GENERATIONS_PATH="$DATASET_DIR/results/slurm_04-07"

echo "Saving generations and evaluate at ${GENERATIONS_PATH}"

SRUN_ARGS="\
--container-name=$CONTAINER_NAME \
--container-mounts=$CONTAINER_MOUNTS \
--container-workdir=$CONTAINER_WORKDIR \
"

DESCRIBE="python3 $DATASET_DIR/main.py \
	--model $MODEL_NAME \
	--prompt $PROMPT\
	--tasks humanevalexplaindescribe-$LANGUAGE\
	--n_samples $DATASET_NUM_SAMPLE \
	--temperature $MODEL_TEMP \
	--top_p $MODEL_TOP_P \
	--max_tokens $MODEL_MAXTOKENS \
	--allow_code_execution \
	--save_generations \
	--save_generations_path $GENERATIONS_PATH/${MODEL_NAME}.json \
	--trust_remote_code \
	--generation_only"

# To generate code from synthesized summary, using greedy search
SYNTHESIZE="python3 $DATASET_DIR/main.py \
	--model $MODEL_NAME \
	--prompt $PROMPT\
	--tasks humanevalexplainsynthesize-$LANGUAGE \
	--n_samples 1 \
	--temperature 0.0 \
	--top_p 1 \
	--max_tokens 386 \
	--allow_code_execution \
	--load_data_path $GENERATIONS_PATH/${MODEL_NAME}_humanevalexplaindescribe-$LANGUAGE.json \
	--save_generations \
	--save_generations_path $GENERATIONS_PATH/${MODEL_NAME}.json \
	--trust_remote_code \
	--metric_output_path $GENERATIONS_PATH/metric_${MODEL_NAME}_humanevalexplain-$LANGUAGE.json"
	
echo $DESCRIBE
echo $SYNTHESIZE

# Run generation on datasets
srun  \
  $SRUN_ARGS \
  /bin/bash -c "pip install -e energy-code-eval; \
  $DESCRIBE; $SYNTHESIZE"

echo "END TIME: $(date)"