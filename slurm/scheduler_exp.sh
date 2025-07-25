#!/bin/bash
#SBATCH --job-name="scheduler"                 # Job name
#SBATCH --cpus-per-gpu=1
#SBATCH --gpus-per-node=GA100:1
#SBATCH --constraint=gpu_mem_80
#SBATCH --mem=5GB                      # Memory per node
#SBATCH --time=5:00:00                # Time limit set to 5hrs
#SBATCH --output=slurm-%A_%a.out

echo "START TIME: $(date)"
nvidia-smi

### Configuration
# Use container including vLLM version 0.8.4
CONTAINER=dockerproxy.repos.tech.orange_vllm_vllm-openai_v0.8.4
# Mount the host directory to the container
# !Note: /datasets used for saved models and it is ReadOnly 
CONTAINER_MOUNTS=/opt/marcel-c3/workdir/shvm6927/workdir/:/workdir,\
/opt/marcel-c3/dataset/shvm6927:/datasets
CONTAINER_WORKDIR=/workdir
# Datasets directory for models
CONTAINER_DATASETS=/datasets

# Experiment variable - change here for each experiment
NUM_STEP=$1
BOOL_CHUNKED_PREFILL=$2
# Batch size
MNS=$3
# Number of queries (duplicated from the same 1st question of HumanEval)
N_QUERIES=$4
# Maximum tokens for generation
MAX_TOKENS=$5

# Models name extracted from models.txt to execute jobs in array
MODEL=$CONTAINER_DATASETS/$(sed -n "${SLURM_ARRAY_TASK_ID}p" models.txt)
# Else: pass as arguments to the script (uncomment this line and comment the above line)
# MODEL=$4

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

echo "Saving generations and evaluate at ${RESULT_PATH}"

SRUN_ARGS="\
--container=$CONTAINER \
--container-mounts=$CONTAINER_MOUNTS \
--container-workdir=$CONTAINER_WORKDIR \
"

CMD="python3 $CONTAINER_WORKDIR/energy-code-eval/main.py \
	--model $MODEL \
	--tasks humaneval \
	--limit 1 \
	--all_n_samples $N_QUERIES \
	--all_max_tokens $MAX_TOKENS \
	--all_max_num_seqs $MNS \
	--temperature $MODEL_TEMP \
	--top_p $MODEL_TOP_P \
	--no_stop \
	--generation_only \
	--trust_remote_code \
	--enforce_eager \
	--max_model_len 16384 \
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

EXIT_CODE=$?
echo "[$SLURM_JOB_ID] Finished '$(date)' with exit($EXIT_CODE)"

# If job fails, requeue only once
if [[ $EXIT_CODE -ne 0 ]] && [[ -z "$SLURM_RESTART_COUNT" ]] ; then
    echo "[$SLURM_JOB_ID] Requeuing '$SLURM_JOB_ID' because Job exited with '$EXIT_CODE'"
    exit 42
fi

echo "END TIME: $(date)"