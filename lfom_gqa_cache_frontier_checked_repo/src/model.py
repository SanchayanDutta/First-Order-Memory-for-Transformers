from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class GPTConfig:
    vocab_size: int = 256
    block_size: int = 256
    n_layer: int = 8
    n_head: int = 8
    n_kv_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    repair: str = 'none'  # none, mlp, lfom
    repair_scale: float = 0.1

class MLPRepair(nn.Module):
    def __init__(self, d: int, hidden_mult: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2*d, hidden_mult*d), nn.GELU(), nn.Linear(hidden_mult*d, d)
        )
    def forward(self, x, attn):
        return self.net(torch.cat([x, attn], dim=-1))

class LFOMRepair(nn.Module):
    """Causal recurrent first-order memory repair.

    The update has the LFOM shape
        m_t = gate_t * m_{t-1} + (1-gate_t) * correction_t
        out_t = W m_t.
    It is deliberately small. The point is to add recurrent state, not a large feedforward adapter.
    """
    def __init__(self, d: int):
        super().__init__()
        self.inp = nn.Linear(2*d, d)
        self.mem = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(2*d, d)
        self.out = nn.Linear(d, d)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
    def forward(self, x, attn):
        B, T, C = x.shape
        cat = torch.cat([x, attn], dim=-1)
        m = torch.zeros(B, C, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(T):
            ct = cat[:, t, :]
            g = torch.sigmoid(self.gate(ct))
            cand = torch.tanh(self.inp(ct) + self.mem(m))
            m = g * m + (1.0 - g) * cand
            outs.append(self.out(m))
        return torch.stack(outs, dim=1)

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        assert cfg.n_head % cfg.n_kv_head == 0
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.group = cfg.n_head // cfg.n_kv_head
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        if cfg.repair == 'mlp':
            self.repair = MLPRepair(cfg.n_embd)
        elif cfg.repair == 'lfom':
            self.repair = LFOMRepair(cfg.n_embd)
        else:
            self.repair = None
    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1,2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1,2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1,2)
        if self.n_kv_head != self.n_head:
            k = k.repeat_interleave(self.group, dim=1)
            v = v.repeat_interleave(self.group, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.cfg.dropout if self.training else 0.0)
        y = y.transpose(1,2).contiguous().view(B, T, C)
        y = self.o_proj(y)
        if self.repair is not None:
            y = y + self.cfg.repair_scale * self.repair(x, y)
        return self.dropout(y)

class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(nn.Linear(cfg.n_embd, 4*cfg.n_embd), nn.GELU(), nn.Linear(4*cfg.n_embd, cfg.n_embd), nn.Dropout(cfg.dropout))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos)[None, :, :])
        for block in self.blocks: x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

def count_params(model: nn.Module, trainable_only: bool = False):
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))

def freeze_backbone_except_repair(model: GPT):
    for p in model.parameters():
        p.requires_grad = False
    for block in model.blocks:
        if block.attn.repair is not None:
            for p in block.attn.repair.parameters():
                p.requires_grad = True
    return model

def convert_mha_to_gqa_state(mha_state: Dict[str, torch.Tensor], cfg_src: GPTConfig, cfg_dst: GPTConfig) -> Dict[str, torch.Tensor]:
    """Map an MHA checkpoint to a GQA/MQA checkpoint by averaging K/V projection heads."""
    state = {}
    hd = cfg_src.n_embd // cfg_src.n_head
    group = cfg_src.n_head // cfg_dst.n_kv_head
    for k, v in mha_state.items():
        if k.endswith('attn.k_proj.weight') or k.endswith('attn.v_proj.weight'):
            # src shape [n_head*hd, d]
            w = v.view(cfg_src.n_head, hd, cfg_src.n_embd)
            w = w.view(cfg_dst.n_kv_head, group, hd, cfg_src.n_embd).mean(dim=1)
            state[k] = w.reshape(cfg_dst.n_kv_head * hd, cfg_src.n_embd).contiguous()
        else:
            state[k] = v
    return state
