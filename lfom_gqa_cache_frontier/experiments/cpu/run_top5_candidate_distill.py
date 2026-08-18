import os, math, time, random, zipfile
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
OUT=Path('/mnt/data/lfom_gqa_distill_3seed_fast'); OUT.mkdir(exist_ok=True)
torch.set_num_threads(4)
# corpus
texts=[]
for p in list(Path('/mnt/data').rglob('*.md'))[:250]:
    try:
        s=p.read_text(errors='ignore')
        if len(s)>1000: texts.append(s[:5000])
    except: pass
for base in ['/mnt/data/wikitext_unzipped','/mnt/data/wikitext_unzip']:
    if Path(base).exists():
        for p in Path(base).rglob('*.parquet'):
            if 'validation' in p.name or 'test' in p.name:
                try:
                    df=pd.read_parquet(p, columns=['text']); texts += [str(x) for x in df['text'].dropna().tolist()[:2000]]
                except Exception: pass
text='\n'.join(texts)[:900000]
if len(text)<100000: text=(text+'\n')*(100000//max(len(text),1)+1)
arr=np.frombuffer(text.encode('utf-8','ignore'),dtype=np.uint8).astype(np.int64)
train=arr[:int(.9*len(arr))]; val=arr[int(.9*len(arr)):]
print('bytes',len(arr),flush=True)
vocab=256; seq_len=64; batch=16; d_model=64; n_heads=4; head_dim=16

def get_batch(split,rng):
    a=train if split=='train' else val
    ix=rng.integers(0,len(a)-seq_len-1,size=batch)
    x=np.stack([a[i:i+seq_len] for i in ix]); y=np.stack([a[i+1:i+seq_len+1] for i in ix])
    return torch.tensor(x,dtype=torch.long), torch.tensor(y,dtype=torch.long)
class LFOMMem(nn.Module):
    def __init__(self,d,mem=64):
        super().__init__(); self.i=nn.Linear(d,mem); self.g=nn.Linear(d,mem); self.o=nn.Linear(mem,d); nn.init.zeros_(self.o.weight); nn.init.zeros_(self.o.bias)
    def forward(self,x):
        B,T,D=x.shape; m=x.new_zeros(B,self.i.out_features); outs=[]
        for t in range(T):
            xt=x[:,t]; gate=torch.sigmoid(self.g(xt)); m=gate*m+(1-gate)*torch.tanh(self.i(xt)); outs.append(self.o(m))
        return torch.stack(outs,1)
class MLPRes(nn.Module):
    def __init__(self,d,mem=64):
        super().__init__(); self.net=nn.Sequential(nn.Linear(d,mem),nn.GELU(),nn.Linear(mem,d)); nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self,x): return self.net(x)
class Attn(nn.Module):
    def __init__(self,kv=4,res='none'):
        super().__init__(); self.kv=kv; self.q=nn.Linear(d_model,d_model,bias=False); self.k=nn.Linear(d_model,kv*head_dim,bias=False); self.v=nn.Linear(d_model,kv*head_dim,bias=False); self.o=nn.Linear(d_model,d_model,bias=False)
        self.mem=LFOMMem(d_model,64) if res=='lfom' else MLPRes(d_model,64) if res=='mlp' else None
    def forward(self,x):
        B,T,D=x.shape; q=self.q(x).view(B,T,n_heads,head_dim).transpose(1,2); k=self.k(x).view(B,T,self.kv,head_dim).transpose(1,2); v=self.v(x).view(B,T,self.kv,head_dim).transpose(1,2)
        if self.kv!=n_heads:
            rep=n_heads//self.kv; k=k.repeat_interleave(rep,1); v=v.repeat_interleave(rep,1)
        a=q@k.transpose(-2,-1)/math.sqrt(head_dim); mask=torch.triu(torch.ones(T,T,dtype=torch.bool),1); a=a.masked_fill(mask,-1e4); y=(F.softmax(a,-1)@v).transpose(1,2).reshape(B,T,D); y=self.o(y)
        if self.mem is not None: y=y+self.mem(x)
        return y
class LM(nn.Module):
    def __init__(self,kv=4,res='none'):
        super().__init__(); self.tok=nn.Embedding(vocab,d_model); self.pos=nn.Parameter(torch.zeros(1,seq_len,d_model)); self.ln1=nn.LayerNorm(d_model); self.att=Attn(kv,res); self.ln2=nn.LayerNorm(d_model); self.mlp=nn.Sequential(nn.Linear(d_model,4*d_model),nn.GELU(),nn.Linear(4*d_model,d_model)); self.ln=nn.LayerNorm(d_model); self.head=nn.Linear(d_model,vocab,bias=False)
    def forward(self,x):
        h=self.tok(x)+self.pos[:,:x.shape[1]]; h=h+self.att(self.ln1(h)); h=h+self.mlp(self.ln2(h)); return self.head(self.ln(h))
def evalm(m,rng,nb=30):
    m.eval(); loss=0; acc=0; n=0
    with torch.no_grad():
        for _ in range(nb):
            x,y=get_batch('val',rng); z=m(x); loss+=F.cross_entropy(z.reshape(-1,vocab),y.reshape(-1),reduction='sum').item(); acc+=(z.argmax(-1)==y).sum().item(); n+=y.numel()
    return loss/n, acc/n
def compress(t,kv,res='none'):
    s=LM(kv,res); s.tok.weight.data.copy_(t.tok.weight); s.pos.data.copy_(t.pos); s.ln1.load_state_dict(t.ln1.state_dict()); s.ln2.load_state_dict(t.ln2.state_dict()); s.ln.load_state_dict(t.ln.state_dict()); s.mlp.load_state_dict(t.mlp.state_dict()); s.head.weight.data.copy_(t.head.weight); s.att.q.weight.data.copy_(t.att.q.weight); s.att.o.weight.data.copy_(t.att.o.weight)
    for nm in ['k','v']:
        tw=getattr(t.att,nm).weight.data.view(n_heads,head_dim,d_model); rep=n_heads//kv; cw=tw.view(kv,rep,head_dim,d_model).mean(1).reshape(kv*head_dim,d_model); getattr(s.att,nm).weight.data.copy_(cw)
    return s
def train_teacher(seed,rng,steps=180):
    torch.manual_seed(seed); m=LM(4); opt=torch.optim.AdamW(m.parameters(),lr=3e-3)
    for i in range(steps):
        x,y=get_batch('train',rng); z=m(x); loss=F.cross_entropy(z.reshape(-1,vocab),y.reshape(-1)); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1); opt.step()
    return m
def train_res(m,t,rng,steps=120,res='lfom'):
    for p in m.parameters(): p.requires_grad=False
    for p in m.att.mem.parameters(): p.requires_grad=True
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=4e-3 if res=='lfom' else 3e-3)
    t.eval()
    for i in range(steps):
        x,y=get_batch('train',rng); z=m(x)
        with torch.no_grad(): tz=t(x)
        ce=F.cross_entropy(z.reshape(-1,vocab),y.reshape(-1)); kl=F.kl_div(F.log_softmax(z/2,-1),F.softmax(tz/2,-1),reduction='batchmean')*4/seq_len
        loss=.5*ce+.5*kl; opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1); opt.step()
    return m
def count_train(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
rows=[]
for seed in range(3):
    random.seed(seed); np.random.seed(seed); rng=np.random.default_rng(1234+seed); torch.manual_seed(seed)
    t0=time.time(); teacher=train_teacher(seed,rng,140); ce,ac=evalm(teacher,rng,25); rows.append(dict(seed=seed,method='MHA_teacher',kv=4,cache=1.0,ce=ce,acc=ac,trainable=sum(p.numel() for p in teacher.parameters()),time=time.time()-t0)); print(seed,'teacher',ce,ac,flush=True)
    for kv,prefix in [(2,'GQA'),(1,'MQA')]:
        b=compress(teacher,kv,'none'); ce,ac=evalm(b,rng,25); rows.append(dict(seed=seed,method=prefix,kv=kv,cache=kv/n_heads,ce=ce,acc=ac,trainable=0,time=0)); print(seed,prefix,ce,ac,flush=True)
        for res in ['mlp','lfom']:
            t0=time.time(); m=compress(teacher,kv,res); train_res(m,teacher,rng,90,res); ce,ac=evalm(m,rng,25); rows.append(dict(seed=seed,method=prefix+'_'+res.upper(),kv=kv,cache=kv/n_heads,ce=ce,acc=ac,trainable=count_train(m),time=time.time()-t0)); print(seed,prefix,res,ce,ac,count_train(m),flush=True)
raw=pd.DataFrame(rows); raw.to_csv(OUT/'raw.csv',index=False)
summary=raw.groupby(['method','kv','cache']).agg(ce_mean=('ce','mean'),ce_std=('ce','std'),acc_mean=('acc','mean'),acc_std=('acc','std'),trainable=('trainable','mean')).reset_index().sort_values('ce_mean')
summary.to_csv(OUT/'summary.csv',index=False)
# paired comparisons
pairs=[]
for seed,g in raw.groupby('seed'):
    d={r.method:r for _,r in g.iterrows()}
    for lf in ['GQA_LFOM','MQA_LFOM']:
        for base in ['MHA_teacher','GQA','GQA_MLP','MQA','MQA_MLP']:
            pairs.append(dict(seed=seed,lfom=lf,baseline=base,ce_delta=d[lf].ce-d[base].ce,acc_delta=d[lf].acc-d[base].acc,ce_win=d[lf].ce<d[base].ce,acc_win=d[lf].acc>d[base].acc))
pd.DataFrame(pairs).to_csv(OUT/'paired.csv',index=False)
pair_summary=pd.DataFrame(pairs).groupby(['lfom','baseline']).agg(ce_wins=('ce_win','sum'),acc_wins=('acc_win','sum'),mean_ce_delta=('ce_delta','mean'),mean_acc_delta=('acc_delta','mean')).reset_index()
pair_summary.to_csv(OUT/'paired_summary.csv',index=False)
# headline
headline=summary[summary.method.isin(['MHA_teacher','GQA','GQA_LFOM','MQA','MQA_LFOM','GQA_MLP','MQA_MLP'])]
headline.to_csv(OUT/'headline.csv',index=False)
# figures
fig,ax=plt.subplots(figsize=(7,3.5)); h=headline.sort_values(['cache','method']); labels=[f"{m}\n{c:.2f}x" for m,c in zip(h.method,h.cache)]; ax.bar(range(len(h)),h.ce_mean,yerr=h.ce_std,capsize=2); ax.set_xticks(range(len(h))); ax.set_xticklabels(labels,rotation=35,ha='right',fontsize=8); ax.set_ylabel('held-out CE'); ax.set_title('LFOM repair after KV-head compression, 5 seeds'); fig.tight_layout(); fig.savefig(OUT/'ce.png',dpi=180)
report=f"""# LFOM-GQA/MQA multiseed distillation repair

A small causal byte-level MHA LM is trained on real text for each seed. Its K/V heads are compressed to GQA or MQA. The compressed backbone is frozen. We then train either a feedforward residual repair or a causal recurrent LFOM repair. The metric is held-out next-byte CE and accuracy.

## Headline

{headline.to_markdown(index=False)}

## Paired comparisons

{pair_summary.to_markdown(index=False)}

## Interpretation

This is an end-to-end language-model objective on real text, but still CPU-scale. The important evidence is whether GQA/MQA with LFOM shifts the cache-quality frontier compared with compressed baselines and nonrecurrent MLP repair.
"""
(OUT/'REPORT.md').write_text(report)
with zipfile.ZipFile('/mnt/data/lfom_gqa_distill_3seed_fast_package.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob('*'): z.write(p,p.relative_to(OUT.parent))
    z.write('/mnt/data/run_lfom_gqa_distill_multiseed.py','run_lfom_gqa_distill_multiseed.py')
print('DONE',OUT,flush=True)
