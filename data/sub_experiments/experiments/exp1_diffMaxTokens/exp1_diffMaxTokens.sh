MODEL=$1
MAX_TOKENS_LIST=(256 512 1024)

for MAX_TOKENS in "${MAX_TOKENS_LIST[@]}"; do
    python3 ./profile_throughput.py \
        --model "$MODEL" \
        --max_tokens "$MAX_TOKENS" \
        --max_num_seqs 128 \
        --n_samples 1 \
        --temperature 0 \
        --tasks humaneval \
        --enforce_eager \
        --no_stop
done