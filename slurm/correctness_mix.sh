#!/bin/bash
#SBATCH --job-name="corr_mix"                 # Job name
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

# Launching jobs in array=1-<nb. of models> - we can execute multiple elemental jobs (each model is a sub-job)
# sharing the same configurations
MODEL_NAME=$(sed -n "${SLURM_ARRAY_TASK_ID}p" models.txt)

# Execute multiple task for the same model with same generation's configuration within a container
TASKS=humaneval,mbpp,codesearchnet-python,humanevalplus,mbppplus

# Sampling temperature
MODEL_TEMP=0.5
MODEL_TOP_P=0.95
MODEL_MAXTOKENS=512
DATASET_NUM_SAMPLE=20

# Experiment results path
RESULT_PATH="$CONTAINER_WORKDIR/energy-code-eval/results/correctness/mix"

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
	--max_num_seqs 128 \
	--allow_code_execution \
	--trust_remote_code \
	--enforce_eager \
	--save_monitoring_folder $RESULT_PATH \
"

echo $CMD

# Run generation on datasets
srun  \
  $SRUN_ARGS \
  /bin/bash -c "pip install -e energy-code-eval; \
  $CMD"

echo "END TIME: $(date)"