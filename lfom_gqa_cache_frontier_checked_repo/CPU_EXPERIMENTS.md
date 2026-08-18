# CPU experiments included in this repo

The CPU results are included to document the present evidence ladder. They are not a substitute for the GPU-scale validation.

## 1. `extra_final_fast`

Main CPU-scale candidate result used in the paper. It evaluates end-to-end language modeling at longer context length `T=80` after K/V-head compression and repair.

Key result:

```text
MQA + LFOM uses 25% K/V cache and beats MHA on CE and accuracy in 6/6 seeds.
GQA + LFOM uses 50% K/V cache and beats MHA on CE and accuracy in 6/6 seeds.
Both LFOM variants beat their nonrecurrent MLP repair controls in 6/6 seeds.
```

## 2. `top5_candidate_v2`

Earlier multiseed small-LM cache-quality run and full-cache controls. Useful for checking whether the LFOM effect is just a full-cache recurrent-memory effect. The result supports the mechanism but is not the final claim.

## 3. `uptraining_fast5`

Uptraining-style run. A full MHA model is trained first, K/V heads are compressed, and the compressed model is continued with repair modules. This is closer to the practical GQA path. It gives a clean half-cache GQA+LFOM signal, with a weaker MQA result.

## 4. `fair_frontier_ultra8`

Stricter 8-seed frontier check with training and longer-context evaluation. It gives a useful boundary condition: LFOM helps very compressed MQA at longer context, but plain GQA can remain strong. This is why the draft phrases the main result as a candidate claim awaiting GPU validation.

## Re-running CPU scripts

The scripts in `experiments/cpu/` are the original reproduction scripts for these folders. They are CPU-oriented and small. They are included for transparency, not as the recommended route to the final result.

For new experiments, use `experiments/cache_frontier.py`, which is cleaner and GPU-ready.
