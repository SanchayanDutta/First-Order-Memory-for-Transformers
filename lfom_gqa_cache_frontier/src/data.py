from __future__ import annotations
import os, glob, random
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch

VOCAB_SIZE = 256

@dataclass
class ByteDataset:
    train: torch.Tensor
    val: torch.Tensor
    block_size: int

    def get_batch(self, split: str, batch_size: int, device: torch.device, block_size: Optional[int] = None):
        data = self.train if split == 'train' else self.val
        T = block_size or self.block_size
        if len(data) <= T + 1:
            raise ValueError(f"dataset split too short for block_size={T}: {len(data)} tokens")
        ix = torch.randint(0, len(data) - T - 1, (batch_size,))
        x = torch.stack([data[i:i+T] for i in ix]).long().to(device)
        y = torch.stack([data[i+1:i+T+1] for i in ix]).long().to(device)
        return x, y

def _read_local_text(path_or_glob: Optional[str]) -> str:
    if path_or_glob and os.path.exists(path_or_glob):
        if os.path.isdir(path_or_glob):
            paths = glob.glob(os.path.join(path_or_glob, '**/*'), recursive=True)
            texts = []
            for p in paths:
                if os.path.isfile(p) and os.path.getsize(p) < 5_000_000:
                    try:
                        texts.append(open(p, 'r', encoding='utf-8', errors='ignore').read())
                    except Exception:
                        pass
            return '\n'.join(texts)
        return open(path_or_glob, 'r', encoding='utf-8', errors='ignore').read()
    if path_or_glob:
        texts = []
        for p in glob.glob(path_or_glob):
            try:
                texts.append(open(p, 'r', encoding='utf-8', errors='ignore').read())
            except Exception:
                pass
        if texts:
            return '\n'.join(texts)
    # Fallback for smoke tests.
    base = (
        "First-order memory can repair compressed attention state. "
        "A small recurrent state can remember context while using fewer key-value heads. "
        "The router, adapter, and attention memory are small states inside a frozen model.\n"
    )
    return base * 2000

def load_byte_dataset(block_size: int, val_frac: float = 0.08, text_path: Optional[str] = None,
                      hf_dataset: Optional[str] = None, hf_config: Optional[str] = None,
                      hf_split: str = 'train', max_chars: int = 5_000_000, seed: int = 1234) -> ByteDataset:
    text = None
    if hf_dataset:
        try:
            from datasets import load_dataset
            ds = load_dataset(hf_dataset, hf_config, split=hf_split) if hf_config else load_dataset(hf_dataset, split=hf_split)
            parts = []
            for row in ds:
                if 'text' in row and row['text']:
                    parts.append(row['text'])
                if sum(len(x) for x in parts) > max_chars:
                    break
            text = '\n'.join(parts)
        except Exception as e:
            print(f"[data] HF load failed ({e}); falling back to local text")
    if text is None:
        text = _read_local_text(text_path)
    text = text[:max_chars]
    b = np.frombuffer(text.encode('utf-8', errors='ignore'), dtype=np.uint8).astype(np.int64)
    if len(b) < block_size * 20:
        b = np.tile(b, int(np.ceil((block_size * 20) / max(1, len(b)))) + 1)
    rng = np.random.default_rng(seed)
    # Keep sequential order but choose a split point near the end.
    n = len(b)
    n_val = max(block_size * 10, int(n * val_frac))
    train = torch.from_numpy(b[:n-n_val].copy())
    val = torch.from_numpy(b[n-n_val:].copy())
    return ByteDataset(train=train, val=val, block_size=block_size)
