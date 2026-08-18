# LFOM-GQA/MQA uptraining frontier, fast 5-seed run

A full MHA byte-level LM is first trained on real text. Its K/V heads are then compressed to GQA or MQA by averaging K/V heads. The compressed models are uptrained for the same number of updates. LFOM variants add a small causal recurrent first-order memory. MLP variants add a nonrecurrent feedforward repair control.

## Final results

| method           |     kv |   cache | repair   |   ce_mean |   ce_std |   acc_mean |   acc_std |     params |   repair_params |
|:-----------------|-------:|--------:|:---------|----------:|---------:|-----------:|----------:|-----------:|----------------:|
| GQA_half_LFOM    | 2.0000 |  0.5000 | lfom     |    1.9809 |   0.0173 |     0.4300 |    0.0087 | 43313.0000 |       3233.0000 |
| MQA_quarter_MLP  | 1.0000 |  0.2500 | mlp      |    1.9816 |   0.0120 |     0.4328 |    0.0077 | 45184.0000 |       6256.0000 |
| GQA_half         | 2.0000 |  0.5000 | none     |    1.9981 |   0.0412 |     0.4279 |    0.0101 | 40080.0000 |          0.0000 |
| GQA_half_MLP     | 2.0000 |  0.5000 | mlp      |    2.0145 |   0.0588 |     0.4208 |    0.0125 | 46336.0000 |       6256.0000 |
| MQA_quarter_LFOM | 1.0000 |  0.2500 | lfom     |    2.0181 |   0.0428 |     0.4219 |    0.0156 | 42161.0000 |       3233.0000 |
| MHA_teacher      | 4.0000 |  1.0000 | none     |    2.0252 |   0.0204 |     0.4239 |    0.0098 | 42384.0000 |          0.0000 |
| MQA_quarter      | 1.0000 |  0.2500 | none     |    2.0275 |   0.0373 |     0.4231 |    0.0105 | 38928.0000 |          0.0000 |

## Paired LFOM wins

| lfom             | baseline        |   ce_wins |   acc_wins |   mean_ce_delta |   mean_acc_delta |
|:-----------------|:----------------|----------:|-----------:|----------------:|-----------------:|
| GQA_half_LFOM    | GQA_half        |         4 |          3 |         -0.0172 |           0.0021 |
| GQA_half_LFOM    | GQA_half_MLP    |         3 |          3 |         -0.0336 |           0.0092 |
| GQA_half_LFOM    | MHA_teacher     |         5 |          3 |         -0.0443 |           0.0061 |
| MQA_quarter_LFOM | MHA_teacher     |         4 |          4 |         -0.0071 |          -0.0019 |
| MQA_quarter_LFOM | MQA_quarter     |         4 |          3 |         -0.0095 |          -0.0012 |
| MQA_quarter_LFOM | MQA_quarter_MLP |         1 |          1 |          0.0365 |          -0.0109 |

## Gap recovery

| compression   |   base_ce |   mha_ce |   mlp_ce |   lfom_ce |   mlp_recovery |   lfom_recovery |   lfom_beats_mha_rate |
|:--------------|----------:|---------:|---------:|----------:|---------------:|----------------:|----------------------:|
| GQA_half      |    1.9981 |   2.0252 |   2.0145 |    1.9809 |         0.9553 |          1.2895 |                1.0000 |
| MQA_quarter   |    2.0275 |   2.0252 |   1.9816 |    2.0181 |         1.2084 |         -4.7316 |                0.8000 |

## Readout

This is closer to the practical GQA path than isolated layer reconstruction: train MHA, compress K/V heads, uptrain compressed models, and evaluate held-out next-token loss. It remains a small CPU-scale model. The claim is a cache-quality frontier diagnostic, not a completed pretrained-LM result.
