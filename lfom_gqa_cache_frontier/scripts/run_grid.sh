#!/usr/bin/env bash
set -euo pipefail
DATA=${DATA:-data/train.txt}
VAL=${VAL:-data/valid.txt}
OUT=${OUT:-runs}
SEEDS=${SEEDS:-"0 1 2"}
for s in $SEEDS; do
  python train.py --data_path $DATA --val_path $VAL --out_dir $OUT/mha_s$s --attn mha --n_layer 8 --n_head 8 --n_embd 512 --block_size 512 --batch_size 32 --steps 50000 --seed $s --device cuda
  CKPT=$OUT/mha_s$s/model.pt
  for attn in gqa mqa; do
    for repair in none mlp lfom; do
      kv=4; [ "$attn" = "mqa" ] && kv=1
      python train.py --data_path $DATA --val_path $VAL --out_dir $OUT/${attn}_${repair}_s$s --init_from $CKPT --attn $attn --repair $repair --n_layer 8 --n_head 8 --n_kv_head $kv --n_embd 512 --block_size 512 --batch_size 32 --steps 10000 --seed $s --device cuda
      python eval_lm.py --ckpt $OUT/${attn}_${repair}_s$s/model.pt --val_path $VAL --device cuda > $OUT/${attn}_${repair}_s$s/eval.json
    done
  done
done
