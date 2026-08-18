# LFOM-GQA/MQA cache frontier run

Device CUDA available: False

## Summary

| method   |   cache_fraction | repair   |      ce |   ce_std |     ppl |   acc |   acc_std |   throughput_toks_s |   trainable_params |
|:---------|-----------------:|:---------|--------:|---------:|--------:|------:|----------:|--------------------:|-------------------:|
| gqa      |              0.5 | none     | 5.56004 |      nan | 259.834 |     0 |       nan |             49.9979 |               7216 |
| gqa_lfom |              0.5 | lfom     | 5.54069 |      nan | 254.853 |     0 |       nan |             54.0525 |               8800 |
| mha      |              1   | none     | 5.54615 |      nan | 256.248 |     0 |       nan |             52.2946 |               7472 |

## Paired wins

| method   | baseline   |   n |   ce_wins |   acc_wins |   mean_ce_delta |   mean_acc_delta |
|:---------|:-----------|----:|----------:|-----------:|----------------:|-----------------:|
| gqa      | mha        |   1 |         0 |          0 |      0.0138969  |                0 |
| gqa      | gqa        |   1 |         0 |          0 |      0          |                0 |
| gqa_lfom | mha        |   1 |         1 |          0 |     -0.00545883 |                0 |
| gqa_lfom | gqa        |   1 |         1 |          0 |     -0.0193558  |                0 |