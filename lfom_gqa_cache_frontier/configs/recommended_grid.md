# Recommended GPU grid

Run three to five seeds. Start small, then scale.

## Small
- n_layer 8
- n_head 8
- n_embd 512
- block_size 512
- MHA pretrain 50k steps
- compressed uptrain 10k steps

## Medium
- n_layer 12
- n_head 12
- n_embd 768
- block_size 1024
- MHA pretrain 100k steps
- compressed uptrain 20k steps

Report validation CE, PPL, accuracy, long-context CE at 2x block size if using RoPE/relative positions, KV-cache bytes/token, and decoding throughput.
