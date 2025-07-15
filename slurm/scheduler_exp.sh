#!/bin/bash
#SBATCH --job-name="scheduler"                 # Job name
#SBATCH --cpus-per-gpu=1
#SBATCH --gpus-per-node=GA100:1
#SBATCH --constraint=gpu_mem_80
#SBATCH --mem=5GB                      # Memory per node
#SBATCH --time=15:00:00                # Time limit set to 15hrs
#SBATCH --output=slurm-%A_%a.out

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

# Experiment variable - change here for each experiment
# MODEL_NAME = [codellama7i, codellama34i, codestral, deepseek_base, deepseek_instruct]
MODEL_NAME=$(sed -n "${SLURM_ARRAY_TASK_ID}p" models.txt)
TASKS=$1
NUM_STEP=$2
BOOL_CHUNKED_PREFILL=$3

# Experiment results path
RESULT_DIR="$CONTAINER_WORKDIR/energy-code-eval/results/scheduler"
if [ "$BOOL_CHUNKED_PREFILL" = "True" ]; then
    RESULT_PATH="$RESULT_DIR/chunked_prefill"
elif [ "$NUM_STEP" -eq 1 ]; then
    RESULT_PATH="$RESULT_DIR/single_step"
elif [ "$NUM_STEP" -gt 1 ]; then
    RESULT_PATH="$RESULT_DIR/multi_step"
else
    echo "Error: Invalid NUM_STEP value"
    exit 1
fi

# Sampling temperature
MODEL_TEMP=0
MODEL_TOP_P=1
MODEL_MAXTOKENS=512
DATASET_NUM_SAMPLE=20

echo "Saving generations and evaluate at ${RESULT_PATH}"

SRUN_ARGS="\
--container=$CONTAINER \
--container-mounts=$CONTAINER_MOUNTS \
--container-workdir=$CONTAINER_WORKDIR \
"

CMD="python3 $CONTAINER_WORKDIR/energy-code-eval/main.py \
	--model $CONTAINER_DATASETS/$MODEL_NAME \
	--tasks $TASKS \
	--n_samples $DATASET_NUM_SAMPLE \
	--temperature $MODEL_TEMP \
	--top_p $MODEL_TOP_P \
	--max_tokens $MODEL_MAXTOKENS \
	--generation_only \
	--trust_remote_code \
	--enforce_eager \
	--max_model_len 16384 \
	--max_num_seqs 128 \
	--num_scheduler_steps $NUM_STEP \
	--enable_chunked_prefill $BOOL_CHUNKED_PREFILL \
	--save_monitoring_folder $RESULT_PATH \
"

echo $CMD

# Run generation on datasets
srun  \
  $SRUN_ARGS \
  /bin/bash -c "pip install -e energy-code-eval; \
  $CMD"

echo "END TIME: $(date)"