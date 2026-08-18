#!/usr/bin/env bash
set -euo pipefail
mkdir -p results/gpu_cache_frontier
python experiments/cache_frontier.py \
  --out_dir results/gpu_cache_frontier \
  --seeds 0 1 2 \
  --hf_dataset Salesforce/wikitext \
  --hf_config wikitext-103-raw-v1 \
  --max_chars 12000000 \
  --n_layer 8 \
  --n_head 8 \
  --n_embd 512 \
  --block_size 512 \
  --long_block_size 512 \
  --long_eval \
  --batch_size 32 \
  --teacher_steps 5000 \
  --uptrain_steps 1200 \
  --eval_iters 100 \
  --throughput_iters 50 \
  --lr 3e-4 \
  --repair_lr 1e-3 \
  --repair_scale 0.2 \
  --freeze_backbone
