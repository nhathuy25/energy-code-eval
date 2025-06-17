#!/bin/bash
#SBATCH --job-name="batching"                 # Job name
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
MNS=$2

# Experiment results path
RESULT_PATH="$CONTAINER_WORKDIR/energy-code-eval/results/batching/mns${MNS}"


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
	--max_num_seqs $MNS \
	--num_scheduler_steps 1 \
	--enable_chunked_prefill False \
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
