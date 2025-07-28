DATASETS=/datasets

CMD="python3 profile_throughput.py \
	--model $DATASETS/$MODEL_NAME \
	--tasks humaneval \
	--n_samples 1 \
	--temperature $MODEL_TEMP \
	--top_p $MODEL_TOP_P \
	--max_tokens $MAX_TOKENS \
	--no_stop \
	--generation_only \
	--trust_remote_code \
	--enforce_eager \
	--max_model_len 16384 \
	--max_num_seqs $MNS \
	--num_scheduler_steps 1 \
	--enable_chunked_prefill False \
	--save_monitoring_folder $RESULT_PATH 
"

echo $CMD