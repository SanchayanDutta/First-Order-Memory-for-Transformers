import re, math, time, random, zipfile
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt

torch.set_num_threads(4)
OUT=Path('/mnt/data/lfom_gqa_uptraining_fast5_quick'); OUT.mkdir(parents=True,exist_ok=True); (OUT/'figures').mkdir(exist_ok=True)
# ASCII real-text corpus. Fast but real text.
def get_corpus(max_chars=650000):
    chunks=[]
    # Prefer WikiText parquet, then source/reports.
    for base in ['/mnt/data/wikitext_unzipped','/mnt/data/wikitext']:
        if Path(base).exists():
            for p in Path(base).rglob('*.parquet'):
                try:
                    df=pd.read_parquet(p, columns=['text']); chunks += [str(x) for x in df['text'].dropna().tolist()[:2500]]
                    if len('\n'.join(chunks))>max_chars: break
                except Exception: pass
            if len('\n'.join(chunks))>max_chars: break
    if len('\n'.join(chunks))<100000:
        for p in Path('/mnt/data').rglob('*.md'):
            try:
                s=p.read_text(errors='ignore')
                if len(s)>500: chunks.append(s[:6000])
            except: pass
            if len('\n'.join(chunks))>max_chars: break
    text='\n'.join(chunks)[:max_chars]
    arr=np.array([ord(c) if (c=='\n' or c=='\t' or 32<=ord(c)<127) else 32 for c in text],dtype=np.int64)
    return arr
class Data:
    def __init__(self,N=360000,V=70000):
        arr=get_corpus(N+V+1000); assert len(arr)>N+V+10, len(arr)
        self.train=torch.tensor(arr[:N]); self.val=torch.tensor(arr[N:N+V]); self.vocab=128
    def batch(self,split,B,T):
        a=self.train if split=='train' else self.val
        idx=torch.randint(0,len(a)-T-1,(B,))
        return torch.stack([a[i:i+T] for i in idx]), torch.stack([a[i+1:i+T+1] for i in idx])
class LFOM(nn.Module):
    def __init__(self,d,h,rank=8):
        super().__init__(); self.h=h; self.rank=rank; self.hd=d//h
        self.write=nn.Linear(d,h*rank); self.gate=nn.Linear(d,h*rank); self.read=nn.Linear(rank,self.hd,bias=False); self.scale=nn.Parameter(torch.tensor(0.08)); nn.init.zeros_(self.read.weight)
    def forward(self,x):
        B,T,D=x.shape; H=self.h; R=self.rank
        w=torch.tanh(self.write(x)).view(B,T,H,R); g=torch.sigmoid(self.gate(x)).view(B,T,H,R)
        s=x.new_zeros(B,H,R); outs=[]
        for t in range(T):
            s=g[:,t]*s + (1-g[:,t])*w[:,t]
            outs.append(self.read(s).reshape(B,D))
        return self.scale*torch.stack(outs,1)
class MLP(nn.Module):
    def __init__(self,d):
        super().__init__(); self.net=nn.Sequential(nn.Linear(d,64),nn.GELU(),nn.Linear(64,d)); nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self,x): return self.net(x)
class Attn(nn.Module):
    def __init__(self,d=48,h=4,kv=4,repair='none'):
        super().__init__(); self.d=d; self.h=h; self.kv=kv; self.hd=d//h; self.repair_kind=repair
        self.q=nn.Linear(d,d,bias=False); self.k=nn.Linear(d,kv*self.hd,bias=False); self.v=nn.Linear(d,kv*self.hd,bias=False); self.o=nn.Linear(d,d,bias=False)
        self.repair=LFOM(d,h,8) if repair=='lfom' else MLP(d) if repair=='mlp' else None
    def forward(self,x):
        B,T,D=x.shape; H=self.h; kv=self.kv; hd=self.hd
        q=self.q(x).view(B,T,H,hd).transpose(1,2); k=self.k(x).view(B,T,kv,hd).transpose(1,2); v=self.v(x).view(B,T,kv,hd).transpose(1,2)
        if kv!=H:
            k=k.repeat_interleave(H//kv,1); v=v.repeat_interleave(H//kv,1)
        a=q@k.transpose(-2,-1)/math.sqrt(hd); mask=torch.triu(torch.ones(T,T,dtype=torch.bool,device=x.device),1); a=a.masked_fill(mask[None,None],-1e4).softmax(-1)
        y=(a@v).transpose(1,2).reshape(B,T,D); y=self.o(y)
        if self.repair is not None: y=y+self.repair(x)
        return y
class LM(nn.Module):
    def __init__(self,vocab=128,d=48,h=4,kv=4,repair='none',T=40):
        super().__init__(); self.vocab=vocab; self.d=d; self.h=h; self.kv=kv; self.T=T
        self.emb=nn.Embedding(vocab,d); self.pos=nn.Parameter(torch.randn(T,d)*.01); self.ln1=nn.LayerNorm(d); self.att=Attn(d,h,kv,repair); self.ln2=nn.LayerNorm(d); self.ff=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d)); self.ln=nn.LayerNorm(d); self.head=nn.Linear(d,vocab,bias=False)
    def forward(self,x):
        z=self.emb(x)+self.pos[:x.shape[1]]; z=z+self.att(self.ln1(z)); z=z+self.ff(self.ln2(z)); return self.head(self.ln(z))
def evalm(m,data,n=35,B=32,T=40):
    m.eval(); L=A=N=0
    with torch.no_grad():
        for _ in range(n):
            x,y=data.batch('val',B,T); logits=m(x); L+=F.cross_entropy(logits.reshape(-1,data.vocab),y.reshape(-1),reduction='sum').item(); A+=(logits.argmax(-1)==y).sum().item(); N+=y.numel()
    return L/N,A/N
def train(m,data,steps,lr=0.003,B=32,T=40):
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=.01); t=time.time()
    for s in range(steps):
        x,y=data.batch('train',B,T); logits=m(x); loss=F.cross_entropy(logits.reshape(-1,data.vocab),y.reshape(-1)); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1); opt.step()
    return time.time()-t
def compress(src,kv,repair):
    h=src.h; hd=src.d//src.h; d=src.d; m=LM(src.vocab,src.d,src.h,kv,repair,src.T); sd=src.state_dict(); nd=m.state_dict()
    for k in nd:
        if k in sd and nd[k].shape==sd[k].shape and 'att.k.weight' not in k and 'att.v.weight' not in k: nd[k]=sd[k].clone()
    for name in ['k','v']:
        W=sd[f'att.{name}.weight'].view(h,hd,d); group=h//kv; nd[f'att.{name}.weight']=W.view(kv,group,hd,d).mean(1).reshape(kv*hd,d).clone()
    m.load_state_dict(nd,strict=False); return m
def main():
    data=Data(); rows=[]
    for seed in range(5):
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        teacher=LM(data.vocab,kv=4,repair='none'); t=train(teacher,data,steps=100,lr=.004); ce,acc=evalm(teacher,data,8); rows.append(dict(seed=seed,method='MHA_teacher',kv=4,cache=1.0,repair='none',ce=ce,acc=acc,params=sum(p.numel() for p in teacher.parameters()),repair_params=0,seconds=t))
        for kv,tag in [(2,'GQA_half'),(1,'MQA_quarter')]:
            for repair in ['none','mlp','lfom']:
                torch.manual_seed(seed*100+kv*10+len(repair)); m=compress(teacher,kv,repair); init_ce,init_acc=evalm(m,data,4); t=train(m,data,steps=35,lr=.0025); ce,acc=evalm(m,data,8); rparams=sum(p.numel() for n,p in m.named_parameters() if 'repair' in n)
                rows.append(dict(seed=seed,method=tag+('' if repair=='none' else '_'+repair.upper()),kv=kv,cache=kv/4,repair=repair,init_ce=init_ce,init_acc=init_acc,ce=ce,acc=acc,params=sum(p.numel() for p in m.parameters()),repair_params=rparams,seconds=t))
        pd.DataFrame(rows).to_csv(OUT/'raw_partial.csv',index=False); print('seed',seed,'done',flush=True)
    df=pd.DataFrame(rows); df.to_csv(OUT/'raw.csv',index=False)
    summ=df.groupby('method').agg(kv=('kv','mean'),cache=('cache','mean'),repair=('repair','first'),ce_mean=('ce','mean'),ce_std=('ce','std'),acc_mean=('acc','mean'),acc_std=('acc','std'),params=('params','mean'),repair_params=('repair_params','mean')).reset_index().sort_values('ce_mean'); summ.to_csv(OUT/'summary.csv',index=False)
    pairs=[]; rec=[]
    for seed,g in df.groupby('seed'):
        D={r.method:r for r in g.itertuples()}; mha=D['MHA_teacher']
        for base,mlp,lfom in [('GQA_half','GQA_half_MLP','GQA_half_LFOM'),('MQA_quarter','MQA_quarter_MLP','MQA_quarter_LFOM')]:
            b,m,l=D[base],D[mlp],D[lfom]
            for other in [base,mlp,'MHA_teacher']:
                o=D[other]; pairs.append(dict(seed=seed,lfom=lfom,baseline=other,ce_win=l.ce<o.ce,acc_win=l.acc>o.acc,ce_delta=l.ce-o.ce,acc_delta=l.acc-o.acc))
            gap=b.ce-mha.ce; rec.append(dict(seed=seed,compression=base,base_ce=b.ce,mlp_ce=m.ce,lfom_ce=l.ce,mha_ce=mha.ce,lfom_recovery=(b.ce-l.ce)/gap if abs(gap)>1e-9 else np.nan,mlp_recovery=(b.ce-m.ce)/gap if abs(gap)>1e-9 else np.nan,lfom_beats_mha=l.ce<mha.ce))
    pw=pd.DataFrame(pairs); pw.to_csv(OUT/'paired_raw.csv',index=False); pws=pw.groupby(['lfom','baseline']).agg(ce_wins=('ce_win','sum'),acc_wins=('acc_win','sum'),mean_ce_delta=('ce_delta','mean'),mean_acc_delta=('acc_delta','mean')).reset_index(); pws.to_csv(OUT/'paired_wins.csv',index=False)
    rdf=pd.DataFrame(rec); rdf.to_csv(OUT/'gap_recovery_raw.csv',index=False); rsum=rdf.groupby('compression').agg(base_ce=('base_ce','mean'),mha_ce=('mha_ce','mean'),mlp_ce=('mlp_ce','mean'),lfom_ce=('lfom_ce','mean'),mlp_recovery=('mlp_recovery','mean'),lfom_recovery=('lfom_recovery','mean'),lfom_beats_mha_rate=('lfom_beats_mha','mean')).reset_index(); rsum.to_csv(OUT/'gap_recovery_summary.csv',index=False)
    order=['MHA_teacher','GQA_half','GQA_half_MLP','GQA_half_LFOM','MQA_quarter','MQA_quarter_MLP','MQA_quarter_LFOM']; s=summ.set_index('method').loc[order].reset_index(); fig,ax=plt.subplots(figsize=(8,3.4)); ax.bar(range(len(s)),s.ce_mean,yerr=s.ce_std,capsize=3); ax.set_xticks(range(len(s))); ax.set_xticklabels(order,rotation=30,ha='right',fontsize=8); ax.set_ylabel('Held-out CE'); ax.set_title('MHA-to-GQA/MQA uptraining with LFOM memory'); fig.tight_layout(); fig.savefig(OUT/'figures/ce.png',dpi=180)
    fig,ax=plt.subplots(figsize=(5,3)); x=np.arange(len(rsum)); w=.35; ax.bar(x-w/2,rsum.mlp_recovery,w,label='MLP'); ax.bar(x+w/2,rsum.lfom_recovery,w,label='LFOM'); ax.axhline(1,ls='--',lw=1); ax.set_xticks(x); ax.set_xticklabels(rsum.compression); ax.set_ylabel('CE gap recovery'); ax.legend(); fig.tight_layout(); fig.savefig(OUT/'figures/gap.png',dpi=180)
    report=f"""# LFOM-GQA/MQA uptraining frontier, fast 5-seed run

A full MHA byte-level LM is first trained on real text. Its K/V heads are then compressed to GQA or MQA by averaging K/V heads. The compressed models are uptrained for the same number of updates. LFOM variants add a small causal recurrent first-order memory. MLP variants add a nonrecurrent feedforward repair control.

## Final results

{summ.to_markdown(index=False,floatfmt='.4f')}

## Paired LFOM wins

{pws.to_markdown(index=False,floatfmt='.4f')}

## Gap recovery

{rsum.to_markdown(index=False,floatfmt='.4f')}

## Readout

This is closer to the practical GQA path than isolated layer reconstruction: train MHA, compress K/V heads, uptrain compressed models, and evaluate held-out next-token loss. It remains a small CPU-scale model. The claim is a cache-quality frontier diagnostic, not a completed pretrained-LM result.
"""
    (OUT/'REPORT.md').write_text(report)
    with zipfile.ZipFile('/mnt/data/lfom_gqa_uptraining_fast5_package.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'): z.write(p,p.relative_to(OUT.parent))
        z.write('/mnt/data/run_lfom_gqa_uptraining_fast5.py','run_lfom_gqa_uptraining_fast5.py')
    print(summ.to_string(index=False)); print(pws.to_string(index=False)); print(rsum.to_string(index=False))
if __name__=='__main__': main()
