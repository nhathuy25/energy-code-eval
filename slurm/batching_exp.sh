#!/bin/bash
#SBATCH --job-name="batching"                 # Job name
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

## Batching experiment variable - change here for each experiment
# Batch size
MNS=$1
# Number of queries (duplicated from the same 1st question of HumanEval)
N_QUERIES=$2
# Maximum tokens for generation
MAX_TOKENS=$3

# Models name extracted from models.txt to execute jobs in array
MODEL=$CONTAINER_DATASETS/$(sed -n "${SLURM_ARRAY_TASK_ID}p" models.txt)
# Else: pass as arguments to the script (uncomment this line and comment the above line)
# MODEL=$4

# Experiment results path
RESULT_PATH="$CONTAINER_WORKDIR/energy-code-eval/results/batching/n${N_QUERIES}"

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
	--num_scheduler_steps 1 \
	--enable_chunked_prefill False \
	--save_monitoring_folder $RESULT_PATH 
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