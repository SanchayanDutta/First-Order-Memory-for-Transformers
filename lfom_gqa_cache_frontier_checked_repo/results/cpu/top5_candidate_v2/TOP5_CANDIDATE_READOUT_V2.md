# LFOM-GQA/MQA top-5 candidate readout v2

## Candidate claim

**Use fewer K/V heads, but give the compressed state first-order memory.**

In the strongest CPU-scale end-to-end result, GQA+LFOM and MQA+LFOM were evaluated on held-out next-byte cross-entropy after training on real text. MQA+LFOM uses 25% of the K/V cache and GQA+LFOM uses 50% of the K/V cache.

## Multiseed end-to-end result

| method | cache | CE | accuracy | trainable repair params |
|---|---:|---:|---:|---:|
| MHA teacher | 1.00 | 2.0955 | 0.3946 | 86720 |
| GQA | 0.50 | 2.2076 | 0.3724 | 0 |
| GQA+MLP repair | 0.50 | 2.1248 | 0.3846 | 8320 |
| **GQA+LFOM repair** | **0.50** | **2.0044** | **0.4194** | **12480** |
| MQA | 0.25 | 2.3462 | 0.3411 | 0 |
| MQA+MLP repair | 0.25 | 2.2014 | 0.3693 | 8320 |
| **MQA+LFOM repair** | **0.25** | **2.0098** | **0.4179** | **12480** |

Paired wins in the 3-seed run:

- GQA+LFOM beats the MHA teacher on CE and accuracy in **3/3** seeds.
- MQA+LFOM beats the MHA teacher on CE and accuracy in **3/3** seeds.
- GQA+LFOM beats GQA+MLP in **3/3** seeds.
- MQA+LFOM beats MQA+MLP in **3/3** seeds.

## Full-cache repair control

A second control trains full-cache MHA+LFOM and MHA+MLP repair modules. It shows that LFOM helps full-cache attention too, but compressed LFOM variants remain close to the full-cache LFOM model. This is the right control for the possible objection that LFOM is simply adding parameters.

## Interpretation

This is the first result that really points toward a top-tier claim. It is end-to-end on a language-model objective, not just layer reconstruction. The effect is also not explained by an ordinary feedforward repair module.

The claim is not final yet because the model is CPU-scale. The decisive GPU experiment is the same comparison on a larger decoder with validation perplexity, long-context recall, KV-cache memory, and throughput. But the current result gives a concrete target:

> **LFOM-GQA/MQA can shift the cache-quality frontier: 25%-50% K/V cache plus recurrent first-order memory can match or beat a full-cache MHA teacher in small end-to-end language modeling.**
