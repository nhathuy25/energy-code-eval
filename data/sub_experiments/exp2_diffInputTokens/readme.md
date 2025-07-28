Experiment on different input values to:
	1. Profile the distribution of HumanEval input sequences
	2. Study the effect of different input seqs length in prefilling
		- To understand whether the prefill time is independent of varation in nb. input sequence's length.
		- Check if prefill time is only depended on total nb. of tokens

Data: 
- `mns32_n1_CodeLlama-7B-Instruct-GPTQ.csv`: 164 questions of HumanEval, batch size 32. We observe same prefill time (ttft) every 32 sequences (batch)
- `mns32_n164_CodeLlama-7b-Instruct-hf.csv`: 1 single question of HumanEval (133 tokens) duplicated 164 times, batch size 32
  - Same prefill time every 32 sequences.
  - Last batch (contains 4 sequences) has time-to-first-token (TTFT) ~32/4 times lesser than normal batch 32.


Found:
1. HumanEval dataset contains various size (nb. token) of sequences
2. Prefilling 
   - Prefill time depends only on the total number of tokens of input batch (multiple sequences), and it's not depends on the difference in size of each sequence inside the same batch
