# Quality checks performed

Date: 2026-08-17

## Paper

- `paper/main.tex` compiles with `pdflatex` twice without LaTeX errors.
- `paper/main.pdf` renders to 4 pages.
- Rendered pages were visually inspected as PNGs. No clipping, missing figures, or obvious broken glyphs were observed.

## Code

- `python tests/smoke_test.py` passed on CPU.
- The smoke test exercises the MHA path, GQA conversion, LFOM repair path, metric writing, and paired-win table generation.

## Repository cleanup

- Python cache files are excluded by `.gitignore`.
- The repo contains no pretrained weights.
- CPU result artifacts are included under `results/cpu/`.
- The to-be-run GPU experiment is implemented in `experiments/cache_frontier.py` and launched by `scripts/run_gpu_cache_frontier.sh`.

## Claim status

The draft states the cache-quality claim as a candidate top-tier claim. The current CPU result is strong enough to motivate the GPU run, but the large-model claim should not be made until the Colab/GPU validation table is positive.
