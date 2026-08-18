# LFOM-GQA/MQA cache frontier

Main candidate claim:

> Use fewer K/V heads, but give the compressed state first-order memory.

The present evidence is CPU-scale. The strongest CPU run shows that MQA+LFOM uses 25% of the K/V cache and beats a full-cache MHA baseline in a small end-to-end byte-level LM across 6/6 seeds. The repo also contains the exact GPU validation path that should be run on Google Colab or another CUDA machine before making a large-model claim.

## Repository layout

```text
paper/                  Two-column draft, figures, and compiled PDF
src/                    Tiny GPT, GQA/MQA conversion, MLP and LFOM repair modules
experiments/gpu/        GPU-ready cache-frontier experiment
experiments/cpu/        CPU-scale reproduction scripts from the current evidence ladder
results/cpu/            Reported CPU-scale result folders
scripts/                Smoke, grid, and Colab/GPU launch scripts
configs/                Win condition and recommended validation grid
tests/                  Fast smoke test
```

## Quick check

```bash
pip install -r requirements.txt
python tests/smoke_test.py
```

## Rebuild the paper

```bash
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The compiled draft is `paper/main.pdf`.

## CPU-scale experiments

The reported CPU-scale experiments are under `results/cpu/`.

The main current result is `results/cpu/extra_final_fast/`, which reports the 6-seed longer-context result used in the paper table:

```text
MHA:        1.00x cache, CE 2.889, accuracy 0.239
GQA+LFOM:   0.50x cache, CE 2.728, accuracy 0.292
MQA+LFOM:   0.25x cache, CE 2.710, accuracy 0.291
```

Other CPU folders are included as boundary and robustness checks:

```text
top5_candidate_v2/      multiseed distillation and full-cache controls
uptraining_fast5/       MHA-to-GQA/MQA uptraining-style run
fair_frontier_ultra8/   stricter 8-seed cache-frontier check
```

## GPU / Colab validation

Run the smoke test first:

```bash
python experiments/cache_frontier.py --smoke_test --out_dir results/smoke --seeds 0
```

Then on Colab or another CUDA machine:

```bash
bash scripts/run_gpu_cache_frontier.sh
```

The strict win condition is in `configs/target_win_condition.md`.

The large-model claim should only be made if LFOM-GQA or LFOM-MQA:

1. matches or beats MHA on validation CE and long-context CE,
2. uses 25%-50% of the K/V cache,
3. beats the nonrecurrent MLP repair control on paired seeds,
4. improves the quality-per-cache or quality-per-throughput frontier.

## Notes

The repo intentionally contains no pretrained model weights. The GPU script can load WikiText through `datasets` if internet is available. If not, pass `--text_path` to a local text corpus.
