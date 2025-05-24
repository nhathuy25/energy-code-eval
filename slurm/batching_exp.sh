#!/bin/bash
#SBATCH --job-name="gpu"                 # Job name
#SBATCH --cpus-per-gpu=1
#SBATCH --gpus-per-node=1
###SBATCH --constraint=gpu_mem_80
#SBATCH --mem=5GB                      # Memory per node
#SBATCH --time=15:00:00                # Time limit set to 15hrs
#SBATCH --output=slurm-%j.out

echo "START TIME: $(date)"

### Configuration
CONTAINER=vllm-openai_latest  
# Mount the host directory to the container
# !Note: /datasets used for saved models and it is ReadOnly 
CONTAINER_MOUNTS=/opt/marcel-c3/workdir/shvm6927/workdir/:/workdir,\
/opt/marcel-c3/dataset/shvm6927:/datasets
CONTAINER_WORKDIR=/workdir
CONTAINER_DATASETS=/datasets

# Experiment variable - change here for each experiment
# MODEL_NAME = [codellama7i, codellama34i, codestral, deepseek_base, deepseek_instruct]
MODEL_NAME=$1
MNS=$2

# Experiment results path
RESULT_PATH="$CONTAINER_WORKDIR/energy-code-eval/results/batching/mns${MNS}"
mkdir -p "$RESULT_PATH"

# Tasks
TASKS=humaneval,mbpp,codesearchnet-python,codesearchnet-java,codesearchnet-javascript

# Sampling temperature
MODEL_TEMP=0
MODEL_TOP_P=1
MODEL_MAXTOKENS=1000
DATASET_NUM_SAMPLE=20

echo "Saving generations and evaluate at ${GENERATIONS_PATH}"

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
	--allow_code_execution \
	--trust_remote_code \
	--enforce_eager \
	--max_num_seqs $MNS \
	--save_monitoring_folder $RESULT_PATH \
"

echo $CMD

# Run generation on datasets
srun  \
  $SRUN_ARGS \
  /bin/bash -c "pip install -e energy-code-eval; \
  $CMD"

echo "END TIME: $(date)"