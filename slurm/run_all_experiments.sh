#!/bin/bash
# run_all_experiments.sh
# Script to launch all experiments with a single command and capture job IDs

# Set the array range
ARRAY_RANGE="1-14"

# Create a markdown file for the job summary
SUMMARY_FILE="experiment_summary.md"

# Initialize the markdown file with headers
cat > $SUMMARY_FILE << EOF
# Experiment Job IDs

## Correctness Experiments

| Experiment Type | Sampling Method | Temperature | Top-p | Job ID |
|-----------------|----------------|------------|-------|--------|
EOF

# Correctness experiments
echo "Launching correctness experiments..."
JOB_ID=$(sbatch --array=$ARRAY_RANGE correctness_greedy.sh | awk '{print $4}')
echo "| correctness     | greedy         | 0          | 1     | $JOB_ID |" >> $SUMMARY_FILE

JOB_ID=$(sbatch --array=$ARRAY_RANGE correctness_nucleus.sh | awk '{print $4}')
echo "| correctness     | nucleus        | 0.8        | 0.95  | $JOB_ID |" >> $SUMMARY_FILE

JOB_ID=$(sbatch --array=$ARRAY_RANGE correctness_mix.sh | awk '{print $4}')
echo "| correctness     | mix            | 0.5        | 0.95  | $JOB_ID |" >> $SUMMARY_FILE

# Correctness HEE experiments
echo "Launching correctness_hee experiments..."
JOB_ID=$(sbatch --array=$ARRAY_RANGE correctness_hee.sh 0 1 | awk '{print $4}')
echo "| correctness_hee | greedy         | 0          | 1     | $JOB_ID |" >> $SUMMARY_FILE

JOB_ID=$(sbatch --array=$ARRAY_RANGE correctness_hee.sh 0.8 0.95 | awk '{print $4}')
echo "| correctness_hee | nucleus        | 0.8        | 0.95  | $JOB_ID |" >> $SUMMARY_FILE

JOB_ID=$(sbatch --array=$ARRAY_RANGE correctness_hee.sh 0.5 0.95 | awk '{print $4}')
echo "| correctness_hee | mix            | 0.5        | 0.95  | $JOB_ID |" >> $SUMMARY_FILE

# Add batching experiments header
cat >> $SUMMARY_FILE << EOF

## Batching Experiments

| Experiment Type | Task                  | Batch Size | Job ID |
|-----------------|----------------------|------------|--------|
EOF

# Batching experiments
echo "Launching batching experiments..."
JOB_ID=$(sbatch --array=$ARRAY_RANGE batching_exp.sh humaneval 32 | awk '{print $4}')
echo "| batching_exp    | humaneval            | 32         | $JOB_ID |" >> $SUMMARY_FILE

JOB_ID=$(sbatch --array=$ARRAY_RANGE batching_exp.sh codesearchnet-python 32 | awk '{print $4}')
echo "| batching_exp    | codesearchnet-python | 32         | $JOB_ID |" >> $SUMMARY_FILE

JOB_ID=$(sbatch --array=$ARRAY_RANGE batching_exp.sh codesearchnet-python 128 | awk '{print $4}')
echo "| batching_exp    | codesearchnet-python | 128        | $JOB_ID |" >> $SUMMARY_FILE

JOB_ID=$(sbatch --array=$ARRAY_RANGE batching_exp.sh codesearchnet-python 512 | awk '{print $4}')
echo "| batching_exp    | codesearchnet-python | 512        | $JOB_ID |" >> $SUMMARY_FILE

JOB_ID=$(sbatch --array=$ARRAY_RANGE batching_exp.sh humaneval 512 | awk '{print $4}')
echo "| batching_exp    | humaneval            | 512        | $JOB_ID |" >> $SUMMARY_FILE

JOB_ID=$(sbatch --array=$ARRAY_RANGE batching_exp.sh humaneval 128 | awk '{print $4}')
echo "| batching_exp    | humaneval            | 128        | $JOB_ID |" >> $SUMMARY_FILE

# Add scheduler experiments header
cat >> $SUMMARY_FILE << EOF

## Scheduler Experiments

| Experiment Type | Task                  | Steps | Chunked Prefill | Job ID |
|-----------------|----------------------|-------|----------------|--------|
EOF

# Scheduler experiments
echo "Launching scheduler experiments..."
JOB_ID=$(sbatch --array=$ARRAY_RANGE scheduler_exp.sh humaneval 1 False | awk '{print $4}')
echo "| scheduler_exp   | humaneval            | 1     | False          | $JOB_ID |" >> $SUMMARY_FILE

JOB_ID=$(sbatch --array=$ARRAY_RANGE scheduler_exp.sh humaneval 8 False | awk '{print $4}')
echo "| scheduler_exp   | humaneval            | 8     | False          | $JOB_ID |" >> $SUMMARY_FILE

JOB_ID=$(sbatch --array=$ARRAY_RANGE scheduler_exp.sh humaneval 1 True | awk '{print $4}')
echo "| scheduler_exp   | humaneval            | 1     | True           | $JOB_ID |" >> $SUMMARY_FILE

JOB_ID=$(sbatch --array=$ARRAY_RANGE scheduler_exp.sh codesearchnet-python 1 False | awk '{print $4}')
echo "| scheduler_exp   | codesearchnet-python | 1     | False          | $JOB_ID |" >> $SUMMARY_FILE

JOB_ID=$(sbatch --array=$ARRAY_RANGE scheduler_exp.sh codesearchnet-python 8 False | awk '{print $4}')
echo "| scheduler_exp   | codesearchnet-python | 8     | False          | $JOB_ID |" >> $SUMMARY_FILE

JOB_ID=$(sbatch --array=$ARRAY_RANGE scheduler_exp.sh codesearchnet-python 1 True | awk '{print $4}')
echo "| scheduler_exp   | codesearchnet-python | 1     | True           | $JOB_ID |" >> $SUMMARY_FILE

echo "All experiments have been submitted!"
echo "Job summary has been saved to $SUMMARY_FILE"
