# Experiment Job IDs

## Correctness Experiments

| Experiment Type | Sampling Method | Temperature | Top-p | Job ID |
|-----------------|----------------|------------|-------|--------|
| correctness     | greedy         | 0          | 1     | 4205375 |
| correctness     | nucleus        | 0.8        | 0.95  | 4205376 |
| correctness     | mix            | 0.5        | 0.95  | 4205377 |
| correctness_hee | greedy         | 0          | 1     | 4205378 |
| correctness_hee | nucleus        | 0.8        | 0.95  | 4205379 |
| correctness_hee | mix            | 0.5        | 0.95  | 4205380 |

## Batching Experiments

| Experiment Type | Task                  | Batch Size | Job ID |
|-----------------|----------------------|------------|--------|
| batching_exp    | humaneval            | 32         | 4205381 |
| batching_exp    | codesearchnet-python | 32         | 4205382 |
| batching_exp    | codesearchnet-python | 128        | 4205383 |
| batching_exp    | codesearchnet-python | 512        | 4205384 |
| batching_exp    | humaneval            | 512        | 4205385 |
| batching_exp    | humaneval            | 128        | 4205386 |

## Scheduler Experiments

| Experiment Type | Task                  | Steps | Chunked Prefill | Job ID |
|-----------------|----------------------|-------|----------------|--------|
| scheduler_exp   | humaneval            | 1     | False          | 4205387 |
| scheduler_exp   | humaneval            | 8     | False          | 4205388 |
| scheduler_exp   | humaneval            | 1     | True           | 4205389 |
| scheduler_exp   | codesearchnet-python | 1     | False          | 4205390 |
| scheduler_exp   | codesearchnet-python | 8     | False          | 4205391 |
| scheduler_exp   | codesearchnet-python | 1     | True           | 4205392 |
