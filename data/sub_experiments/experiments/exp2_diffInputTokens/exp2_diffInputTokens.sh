MODEL=$1
N_SAMPLES_LIST=(1 5 20)

for N_SAMPLES in "${N_SAMPLES_LIST[@]}"; do
    python3 ./profile_throughput.py \
        --model "$MODEL" \
        --max_tokens 512 \
        --max_num_seqs 128 \
        --n_samples "$N_SAMPLES" \
        --temperature 0 \
        --tasks humaneval \
        --enforce_eager \
        --no_stop
done