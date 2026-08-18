# Google Colab / GPU validation

This is the validation path for the cache-quality frontier claim.

## Setup

```bash
unzip -q lfom_gqa_cache_frontier_checked_repo.zip
cd lfom_gqa_cache_frontier_checked_repo
pip install -r requirements.txt
```

Check CUDA:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')
```

## Smoke test

```bash
python experiments/cache_frontier.py --smoke_test --out_dir results/smoke --seeds 0
```

## Small T4/L4 run

```bash
python experiments/cache_frontier.py \
  --out_dir results/colab_small \
  --seeds 0 \
  --hf_dataset Salesforce/wikitext \
  --hf_config wikitext-103-raw-v1 \
  --max_chars 3000000 \
  --n_layer 4 --n_head 4 --n_embd 256 \
  --block_size 256 --long_block_size 256 --long_eval \
  --batch_size 16 \
  --teacher_steps 1000 \
  --uptrain_steps 300 \
  --eval_iters 30 \
  --throughput_iters 10 \
  --freeze_backbone
```

## Full Colab Pro / A100-style run

```bash
bash scripts/run_gpu_cache_frontier.sh
```

## Outputs

The main output directory contains:

```text
raw.csv
summary.csv
paired_wins.csv
README_REPORT.md
```

The paper-level result should be reported only if the win condition in `configs/target_win_condition.md` is met.
