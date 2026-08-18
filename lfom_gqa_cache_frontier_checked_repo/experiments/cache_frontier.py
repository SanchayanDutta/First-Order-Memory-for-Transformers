from __future__ import annotations
import os, sys, time, json, math, argparse, random
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from tqdm import tqdm

ROOT = Path(__file__).resolve()
while ROOT != ROOT.parent and not (ROOT / 'src').exists():
    ROOT = ROOT.parent
sys.path.append(str(ROOT))
from src.data import load_byte_dataset, VOCAB_SIZE
from src.model import GPT, GPTConfig, count_params, convert_mha_to_gqa_state, freeze_backbone_except_repair


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def eval_model(model, ds, device, batch_size, eval_iters, block_size):
    model.eval(); losses=[]; accs=[]
    with torch.no_grad():
        for _ in range(eval_iters):
            x,y = ds.get_batch('val', batch_size, device, block_size=block_size)
            logits, loss = model(x,y)
            pred = logits.argmax(dim=-1)
            losses.append(float(loss.item()))
            accs.append(float((pred == y).float().mean().item()))
    return {'ce': float(np.mean(losses)), 'ppl': float(np.exp(np.mean(losses))), 'acc': float(np.mean(accs))}

def train_model(model, ds, device, steps, batch_size, block_size, lr, weight_decay, eval_every=0, eval_iters=20, desc='train'):
    model.train()
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    t0=time.time()
    for it in tqdm(range(steps), desc=desc, leave=False):
        x,y = ds.get_batch('train', batch_size, device, block_size=block_size)
        opt.zero_grad(set_to_none=True)
        _, loss = model(x,y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
    return time.time()-t0

def throughput(model, ds, device, batch_size, block_size, iters=20):
    model.eval()
    # warmup
    with torch.no_grad():
        for _ in range(3):
            x,_ = ds.get_batch('val', batch_size, device, block_size=block_size)
            model(x)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0=time.time()
        toks=0
        for _ in range(iters):
            x,_ = ds.get_batch('val', batch_size, device, block_size=block_size)
            model(x)
            toks += x.numel()
        if torch.cuda.is_available(): torch.cuda.synchronize()
    return toks/(time.time()-t0)

def kv_cache_fraction(cfg: GPTConfig):
    return cfg.n_kv_head / cfg.n_head

def run_one(seed, args):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    if args.smoke_test:
        args.n_layer=1; args.n_head=2; args.n_embd=16; args.block_size=8; args.long_block_size=8
        args.teacher_steps=1; args.uptrain_steps=1; args.compressed_steps=0; args.batch_size=1; args.eval_iters=1; args.throughput_iters=1
    # The model's position table must cover the longest sequence used at evaluation.
    model_block_size = max(args.block_size, args.long_block_size) if args.long_eval else args.block_size
    ds = load_byte_dataset(block_size=model_block_size, text_path=args.text_path, hf_dataset=args.hf_dataset,
                           hf_config=args.hf_config, max_chars=args.max_chars, seed=seed)
    cfg_mha = GPTConfig(vocab_size=VOCAB_SIZE, block_size=model_block_size, n_layer=args.n_layer, n_head=args.n_head,
                        n_kv_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout, repair='none')
    teacher = GPT(cfg_mha).to(device)
    train_model(teacher, ds, device, args.teacher_steps, args.batch_size, args.block_size, args.lr, args.weight_decay, desc=f'seed{seed}:mha')
    teacher_metrics = eval_model(teacher, ds, device, args.batch_size, args.eval_iters, args.block_size)
    teacher_long = eval_model(teacher, ds, device, max(1,args.batch_size//2), args.eval_iters, min(args.long_block_size, cfg_mha.block_size)) if args.long_eval else {}
    teacher_thr = throughput(teacher, ds, device, args.batch_size, args.block_size, iters=max(3,args.throughput_iters))
    rows=[]
    rows.append(dict(seed=seed, method='mha', kv_heads=args.n_head, cache_fraction=1.0, repair='none', trainable_params=count_params(teacher, True), throughput_toks_s=teacher_thr, **teacher_metrics, **{f'long_{k}':v for k,v in teacher_long.items()}))
    base_state = {k:v.detach().cpu() for k,v in teacher.state_dict().items()}
    variants=[]
    for name, kvh in [('gqa', max(1,args.n_head//2)), ('mqa', 1)]:
        for repair in ['none','mlp','lfom']:
            variants.append((name if repair=='none' else f'{name}_{repair}', kvh, repair))
    if args.smoke_test:
        variants = [('gqa', max(1,args.n_head//2), 'none'), ('gqa_lfom', max(1,args.n_head//2), 'lfom')]
    for method, kvh, repair in variants:
        cfg = GPTConfig(vocab_size=VOCAB_SIZE, block_size=model_block_size, n_layer=args.n_layer, n_head=args.n_head,
                        n_kv_head=kvh, n_embd=args.n_embd, dropout=args.dropout, repair=repair, repair_scale=args.repair_scale)
        model = GPT(cfg).to(device)
        converted = convert_mha_to_gqa_state(base_state, cfg_mha, cfg)
        missing, unexpected = model.load_state_dict(converted, strict=False)
        if args.freeze_backbone and repair != 'none':
            freeze_backbone_except_repair(model)
        # plain compressed can optionally get short continued training to be fair
        if repair == 'none':
            if args.compressed_steps > 0:
                train_model(model, ds, device, args.compressed_steps, args.batch_size, args.block_size, args.lr*0.5, args.weight_decay, desc=f'seed{seed}:{method}')
        else:
            train_model(model, ds, device, args.uptrain_steps, args.batch_size, args.block_size, args.repair_lr, args.weight_decay, desc=f'seed{seed}:{method}')
        met = eval_model(model, ds, device, args.batch_size, args.eval_iters, args.block_size)
        long_met = eval_model(model, ds, device, max(1,args.batch_size//2), args.eval_iters, min(args.long_block_size, cfg.block_size)) if args.long_eval else {}
        thr = throughput(model, ds, device, args.batch_size, args.block_size, iters=max(3,args.throughput_iters))
        rows.append(dict(seed=seed, method=method, kv_heads=kvh, cache_fraction=kv_cache_fraction(cfg), repair=repair,
                         trainable_params=count_params(model, True), throughput_toks_s=thr, **met, **{f'long_{k}':v for k,v in long_met.items()}))
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out_dir', type=str, default='results/gpu_cache_frontier')
    ap.add_argument('--seeds', nargs='+', default=['0','1','2'], help='Seeds as comma-separated or space-separated values, e.g. --seeds 0,1,2 or --seeds 0 1 2')
    ap.add_argument('--text_path', type=str, default=None)
    ap.add_argument('--hf_dataset', type=str, default=None)
    ap.add_argument('--hf_config', type=str, default=None)
    ap.add_argument('--max_chars', type=int, default=8_000_000)
    ap.add_argument('--n_layer', type=int, default=8)
    ap.add_argument('--n_head', type=int, default=8)
    ap.add_argument('--n_embd', type=int, default=512)
    ap.add_argument('--block_size', type=int, default=256)
    ap.add_argument('--long_block_size', type=int, default=512)
    ap.add_argument('--long_eval', action='store_true')
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--teacher_steps', type=int, default=3000)
    ap.add_argument('--compressed_steps', type=int, default=0)
    ap.add_argument('--uptrain_steps', type=int, default=800)
    ap.add_argument('--eval_iters', type=int, default=80)
    ap.add_argument('--throughput_iters', type=int, default=30)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--repair_lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=0.01)
    ap.add_argument('--dropout', type=float, default=0.0)
    ap.add_argument('--repair_scale', type=float, default=0.2)
    ap.add_argument('--freeze_backbone', action='store_true')
    ap.add_argument('--smoke_test', action='store_true')
    ap.add_argument('--cpu', action='store_true')
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    all_rows=[]
    seed_items=[]
    for tok in args.seeds:
        seed_items.extend([x for x in tok.split(',') if x.strip()!=''])
    for s in [int(x) for x in seed_items]:
        rows=run_one(s,args); all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(out/'raw.csv', index=False)
    df=pd.DataFrame(all_rows)
    agg=df.groupby(['method','cache_fraction','repair'], as_index=False).agg(
        ce=('ce','mean'), ce_std=('ce','std'), ppl=('ppl','mean'), acc=('acc','mean'), acc_std=('acc','std'),
        throughput_toks_s=('throughput_toks_s','mean'), trainable_params=('trainable_params','mean')
    )
    if 'long_ce' in df.columns:
        long_agg=df.groupby(['method'], as_index=False).agg(long_ce=('long_ce','mean'), long_acc=('long_acc','mean'))
        agg=agg.merge(long_agg,on='method',how='left')
    agg.to_csv(out/'summary.csv', index=False)
    # paired wins against MHA and MLP controls
    pairs=[]
    for method in sorted(df.method.unique()):
        if method=='mha': continue
        for baseline in ['mha','gqa_mlp','mqa_mlp','gqa','mqa']:
            if baseline not in df.method.unique(): continue
            merged=df[df.method==method][['seed','ce','acc']].merge(df[df.method==baseline][['seed','ce','acc']], on='seed', suffixes=('_method','_base'))
            if len(merged):
                pairs.append(dict(method=method, baseline=baseline, n=len(merged), ce_wins=int((merged.ce_method < merged.ce_base).sum()), acc_wins=int((merged.acc_method > merged.acc_base).sum()), mean_ce_delta=float((merged.ce_method-merged.ce_base).mean()), mean_acc_delta=float((merged.acc_method-merged.acc_base).mean())))
    pd.DataFrame(pairs).to_csv(out/'paired_wins.csv', index=False)
    report = ["# LFOM-GQA/MQA cache frontier run", "", f"Device CUDA available: {torch.cuda.is_available()}", "", "## Summary", "", agg.to_markdown(index=False), "", "## Paired wins", "", pd.DataFrame(pairs).to_markdown(index=False)]
    (out/'README_REPORT.md').write_text('\n'.join(report))
    print('\n'.join(report[:20]))

if __name__=='__main__': main()
