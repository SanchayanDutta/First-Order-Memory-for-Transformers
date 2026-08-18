# LFOM-GQA/MQA multiseed distillation repair

A small causal byte-level MHA LM is trained on real text for each seed. Its K/V heads are compressed to GQA or MQA. The compressed backbone is frozen. We then train either a feedforward residual repair or a causal recurrent LFOM repair. The metric is held-out next-byte CE and accuracy.

## Headline

| method      |   kv |   cache |   ce_mean |      ce_std |   acc_mean |    acc_std |   trainable |
|:------------|-----:|--------:|----------:|------------:|-----------:|-----------:|------------:|
| MHA_LFOM    |    4 |    1    |   2.01776 | 0.025261    |   0.416943 | 0.0100127  |       12480 |
| MQA_LFOM    |    1 |    0.25 |   2.07762 | 0.0184972   |   0.402466 | 0.00210613 |       12480 |
| MHA_MLP     |    4 |    1    |   2.09415 | 0.021763    |   0.393726 | 0.0104616  |        8320 |
| GQA_LFOM    |    2 |    0.5  |   2.10643 | 0.00593939  |   0.393481 | 0.00369436 |       12480 |
| MHA_teacher |    4 |    1    |   2.13382 | 0.0241836   |   0.385986 | 0.01167    |       86720 |
| GQA_MLP     |    2 |    0.5  |   2.13782 | 0.00313771  |   0.382593 | 0.00307288 |        8320 |
| MQA_MLP     |    1 |    0.25 |   2.20375 | 0.00998918  |   0.374243 | 0.00127749 |        8320 |
| GQA         |    2 |    0.5  |   2.2146  | 0.000371432 |   0.375806 | 0.00189897 |           0 |
| MQA         |    1 |    0.25 |   2.35707 | 0.0268941   |   0.361646 | 0.00479921 |           0 |

## Paired comparisons

| lfom     | baseline    |   ce_wins |   acc_wins |   mean_ce_delta |   mean_acc_delta |
|:---------|:------------|----------:|-----------:|----------------:|-----------------:|
| GQA_LFOM | GQA         |         2 |          2 |      -0.108166  |      0.0176758   |
| GQA_LFOM | GQA_MLP     |         2 |          2 |      -0.0313854 |      0.0108887   |
| GQA_LFOM | MHA_LFOM    |         0 |          0 |       0.0886682 |     -0.0234619   |
| GQA_LFOM | MHA_MLP     |         1 |          1 |       0.0122851 |     -0.000244141 |
| GQA_LFOM | MHA_teacher |         2 |          2 |      -0.027392  |      0.00749512  |
| GQA_LFOM | MQA         |         2 |          2 |      -0.250641  |      0.0318359   |
| GQA_LFOM | MQA_MLP     |         2 |          2 |      -0.0973192 |      0.0192383   |
| MQA_LFOM | GQA         |         2 |          2 |      -0.136978  |      0.0266602   |
| MQA_LFOM | GQA_MLP     |         2 |          2 |      -0.0601976 |      0.019873    |
| MQA_LFOM | MHA_LFOM    |         0 |          0 |       0.059856  |     -0.0144775   |
| MQA_LFOM | MHA_MLP     |         2 |          1 |      -0.0165271 |      0.00874023  |
| MQA_LFOM | MHA_teacher |         2 |          2 |      -0.0562042 |      0.0164795   |
| MQA_LFOM | MQA         |         2 |          2 |      -0.279453  |      0.0408203   |
| MQA_LFOM | MQA_MLP     |         2 |          2 |      -0.126131  |      0.0282227   |

## Interpretation

This is an end-to-end language-model objective on real text, but still CPU-scale. The important evidence is whether GQA/MQA with LFOM shifts the cache-quality frontier compared with compressed baselines and nonrecurrent MLP repair.
