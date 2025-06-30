#!/bin/bash
#SBATCH --job-name="corr_hee"                 # Job name
#SBATCH --cpus-per-gpu=1
#SBATCH --gpus-per-node=GA100:1
#SBATCH --constraint=gpu_mem_80
#SBATCH --mem=5GB                      # Memory per node
#SBATCH --time=15:00:00                # Time limit set to 15hrs
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

# Launching jobs in array=1-<nb. of models> - we can execute multiple elemental jobs (each model is a sub-job)
# sharing the same configurations
MODEL_NAME=$(sed -n "${SLURM_ARRAY_TASK_ID}p" models.txt)

# Language [python, java, js]
# tasks = [humanevalexplaindescribe-python,humanevalexplaindescribe-java,humanevalexplaindescribe-javascript]
# Update 30/06: we only execute code summarization task in python language
LANGUAGE=python

# Sampling temperature for summary generation
MODEL_TEMP=$1
MODEL_TOP_P=$2 
MODEL_MAXTOKENS=2000
DATASET_NUM_SAMPLE=20

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
	--model $CONTAINER_DATASETS/$MODEL_NAME \
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

# To generate code from synthesized summary, using greedy search
SYNTHESIZE="python3 $CONTAINER_WORKDIR/energy-code-eval/main.py  \
	--model $CONTAINER_DATASETS/$MODEL_NAME \
	--tasks humanevalexplainsynthesize-${LANGUAGE} \
	--n_samples 5 \
	--temperature 0 \
	--top_p 1 \
	--max_tokens 512 \
	--allow_code_execution \
	--enforce_eager \
	--max_model_len 16384 \
	--num_scheduler_steps 1 \
	--enable_chunked_prefill False \
	--max_num_seqs 128 \
	--load_data_path $RESULT_PATH/generations/${MODEL_NAME}_humanevalexplaindescribe-$LANGUAGE.json \
	--trust_remote_code \
	--no_monitor \
	--save_monitoring_folder $RESULT_PATH"
	
echo $DESCRIBE
echo $SYNTHESIZE

# Run generation on datasets
srun  \
  $SRUN_ARGS \
  /bin/bash -c "pip install -e energy-code-eval; \
  $DESCRIBE; $SYNTHESIZE"

echo "END TIME: $(date)"