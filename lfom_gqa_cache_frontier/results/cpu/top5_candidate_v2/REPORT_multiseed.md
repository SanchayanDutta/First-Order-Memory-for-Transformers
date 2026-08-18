# LFOM-GQA/MQA multiseed distillation repair

A small causal byte-level MHA LM is trained on real text for each seed. Its K/V heads are compressed to GQA or MQA. The compressed backbone is frozen. We then train either a feedforward residual repair or a causal recurrent LFOM repair. The metric is held-out next-byte CE and accuracy.

## Headline

| method      |   kv |   cache |   ce_mean |    ce_std |   acc_mean |    acc_std |   trainable |
|:------------|-----:|--------:|----------:|----------:|-----------:|-----------:|------------:|
| GQA_LFOM    |    2 |    0.5  |   2.00436 | 0.0143809 |   0.419375 | 0.00422381 |       12480 |
| MQA_LFOM    |    1 |    0.25 |   2.00982 | 0.032942  |   0.417852 | 0.0197917  |       12480 |
| MHA_teacher |    4 |    1    |   2.09552 | 0.0142579 |   0.394635 | 0.00781253 |       86720 |
| GQA_MLP     |    2 |    0.5  |   2.12481 | 0.0162656 |   0.384557 | 0.00482075 |        8320 |
| MQA_MLP     |    1 |    0.25 |   2.20142 | 0.0230766 |   0.36931  | 0.0034854  |        8320 |
| GQA         |    2 |    0.5  |   2.2076  | 0.0606002 |   0.372409 | 0.0116931  |           0 |
| MQA         |    1 |    0.25 |   2.34615 | 0.0246454 |   0.34112  | 0.0199862  |           0 |

## Paired comparisons

| lfom     | baseline    |   ce_wins |   acc_wins |   mean_ce_delta |   mean_acc_delta |
|:---------|:------------|----------:|-----------:|----------------:|-----------------:|
| GQA_LFOM | GQA         |         3 |          3 |      -0.203231  |        0.0469661 |
| GQA_LFOM | GQA_MLP     |         3 |          3 |      -0.120448  |        0.0348177 |
| GQA_LFOM | MHA_teacher |         3 |          3 |      -0.0911593 |        0.0247396 |
| GQA_LFOM | MQA         |         3 |          3 |      -0.341789  |        0.0782552 |
| GQA_LFOM | MQA_MLP     |         3 |          3 |      -0.197058  |        0.0500651 |
| MQA_LFOM | GQA         |         3 |          3 |      -0.197775  |        0.0454427 |
| MQA_LFOM | GQA_MLP     |         3 |          3 |      -0.114992  |        0.0332943 |
| MQA_LFOM | MHA_teacher |         3 |          3 |      -0.0857033 |        0.0232161 |
| MQA_LFOM | MQA         |         3 |          3 |      -0.336333  |        0.0767318 |
| MQA_LFOM | MQA_MLP     |         3 |          3 |      -0.191602  |        0.0485417 |

## Interpretation

This is an end-to-end language-model objective on real text, but still CPU-scale. The important evidence is whether GQA/MQA with LFOM shifts the cache-quality frontier compared with compressed baselines and nonrecurrent MLP repair.
