import os, re, math, json, zipfile, time
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt

torch.set_num_threads(4); OUT=Path('/mnt/data/lfom_gqa_fair_frontier_ultra'); OUT.mkdir(exist_ok=True); (OUT/'figures').mkdir(exist_ok=True)

def collect():
    chunks=[]; total=0
    for p in Path('/mnt/data').rglob('*'):
        if p.is_file() and p.suffix.lower() in {'.md','.py','.tex','.txt'} and p.stat().st_size<300000:
            try:s=p.read_text(errors='ignore')
            except: continue
            if len(s)>500:
                chunks.append(s[:12000]); total+=min(len(s),12000)
                if total>350000: break
    arr=np.array([ord(c) if c in '\n\t' or 32<=ord(c)<=126 else 32 for c in '\n'.join(chunks)],dtype=np.int64)
    return torch.tensor(arr[:250000]), torch.tensor(arr[250000:320000]), {'chars':len(arr),'chunks':len(chunks)}
class Data:
    def __init__(self):
        self.train,self.val,self.meta=collect(); self.vocab=128
        if len(self.val)<20000: self.val=self.train[-50000:]; self.train=self.train[:-50000]
    def batch(self,split,B,T):
        a=self.train if split=='train' else self.val; st=torch.randint(0,len(a)-T-1,(B,)); return torch.stack([a[s:s+T] for s in st]), torch.stack([a[s+1:s+T+1] for s in st])
class Att(nn.Module):
    def __init__(self,d=36,h=6,kv=6,repair='none',rank=6):
        super().__init__(); self.h=h; self.kv=kv; self.hd=d//h; self.d=d; self.repair=repair; self.rank=rank
        self.q=nn.Linear(d,d,bias=False); self.k=nn.Linear(d,kv*self.hd,bias=False); self.v=nn.Linear(d,kv*self.hd,bias=False); self.o=nn.Linear(d,d,bias=False)
        if repair=='lfom': self.u=nn.Linear(d,h*rank); self.g=nn.Linear(d,h*rank); self.r=nn.Linear(rank,self.hd,bias=False); self.mix=nn.Parameter(torch.tensor(.2))
        if repair=='mlp': self.mlp=nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,d)); self.mix=nn.Parameter(torch.tensor(.2))
    def forward(self,x):
        B,T,D=x.shape; H=self.h; kv=self.kv; hd=self.hd
        q=self.q(x).view(B,T,H,hd).transpose(1,2); k=self.k(x).view(B,T,kv,hd).transpose(1,2); v=self.v(x).view(B,T,kv,hd).transpose(1,2)
        if kv!=H: k=k.repeat_interleave(H//kv,1); v=v.repeat_interleave(H//kv,1)
        a=(q@k.transpose(-2,-1))/math.sqrt(hd); mask=torch.triu(torch.ones(T,T,dtype=torch.bool),1)
        a=a.masked_fill(mask[None,None],-1e4).softmax(-1); out=(a@v).transpose(1,2).contiguous().view(B,T,D); out=self.o(out)
        if self.repair=='lfom':
            u=torch.tanh(self.u(x)).view(B,T,H,self.rank); g=torch.sigmoid(self.g(x)).view(B,T,H,self.rank); st=torch.zeros(B,H,self.rank); add=[]
            for t in range(T): st=g[:,t]*st+(1-g[:,t])*u[:,t]; add.append(self.r(st).reshape(B,D))
            out=out+self.mix*torch.stack(add,1)
        elif self.repair=='mlp': out=out+self.mix*self.mlp(x)
        return out
class Model(nn.Module):
    def __init__(self,vocab=128,kv=6,repair='none',Tmax=96,d=36,h=6):
        super().__init__(); self.d=d; self.h=h; self.kv=kv; self.repair=repair; self.Tmax=Tmax
        self.emb=nn.Embedding(vocab,d); self.pos=nn.Parameter(torch.zeros(Tmax,d)); self.ln1=nn.LayerNorm(d); self.att=Att(d,h,kv,repair); self.ln2=nn.LayerNorm(d); self.ff=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d)); self.ln=nn.LayerNorm(d); self.head=nn.Linear(d,vocab,bias=False)
    def forward(self,x): z=self.emb(x)+self.pos[:x.shape[1]]; z=z+self.att(self.ln1(z)); z=z+self.ff(self.ln2(z)); return self.head(self.ln(z))

def copy_model(src,kv,repair):
    dst=Model(src.emb.num_embeddings,kv=kv,repair=repair); sd=src.state_dict(); dd=dst.state_dict(); H=src.h; hd=src.d//src.h
    for k in dd:
        if k.startswith('att.k') or k.startswith('att.v') or 'att.u' in k or 'att.g' in k or 'att.r' in k or 'att.mlp' in k or k=='att.mix': continue
        if k in sd and sd[k].shape==dd[k].shape: dd[k].copy_(sd[k])
    for name in ['k','v']:
        W=sd[f'att.{name}.weight'].view(H,hd,src.d); grp=H//kv; Wc=W if kv==H else W.view(kv,grp,hd,src.d).mean(1); dd[f'att.{name}.weight'].copy_(Wc.reshape(kv*hd,src.d))
    dst.load_state_dict(dd); return dst

def train(m,data,steps,T=40,B=8,lr=3e-3):
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=0.01)
    for _ in range(steps):
        x,y=data.batch('train',B,T); log=m(x); loss=F.cross_entropy(log.reshape(-1,data.vocab),y.reshape(-1)); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1); opt.step()

def ev(m,data,T=40):
    m.eval(); L=[]; A=[]
    with torch.no_grad():
      for _ in range(6):
        x,y=data.batch('val',8,T); log=m(x); L.append(F.cross_entropy(log.reshape(-1,data.vocab),y.reshape(-1)).item()); A.append((log.argmax(-1)==y).float().mean().item())
    return np.mean(L),np.mean(A)

def main():
    data=Data(); (OUT/'meta.json').write_text(json.dumps(data.meta)); rows=[]
    for seed in range(2):
      torch.manual_seed(seed); np.random.seed(seed); base=Model(data.vocab); train(base,data,60,T=40,B=8,lr=3e-3)
      variants=[('MHA_pretrain',base,1.0,0),('MHA_continue',copy_model(base,6,'none'),1.0,25),('GQA_half',copy_model(base,3,'none'),.5,25),('GQA_half_MLP',copy_model(base,3,'mlp'),.5,25),('GQA_half_LFOM',copy_model(base,3,'lfom'),.5,25),('MQA_sixth',copy_model(base,1,'none'),1/6,25),('MQA_sixth_LFOM',copy_model(base,1,'lfom'),1/6,25)]
      for name,m,cache,up in variants:
        if up: train(m,data,up,T=40,B=8,lr=2e-3)
        for T in [40,80]:
          ce,acc=ev(m,data,T); rows.append({'seed':seed,'method':name,'cache':cache,'T':T,'ce':ce,'acc':acc})
        print(seed,name,flush=True)
      pd.DataFrame(rows).to_csv(OUT/'raw_partial.csv',index=False)
    df=pd.DataFrame(rows); df.to_csv(OUT/'raw.csv',index=False); summ=df.groupby(['T','method']).agg(ce=('ce','mean'),acc=('acc','mean'),cache=('cache','mean')).reset_index().sort_values(['T','ce']); summ.to_csv(OUT/'summary.csv',index=False)
    paired=[]
    for (seed,T),g in df.groupby(['seed','T']):
      d={r.method:r for r in g.itertuples()}
      for b,l in [('GQA_half','GQA_half_LFOM'),('MQA_sixth','MQA_sixth_LFOM')]: paired.append({'seed':seed,'T':T,'base':b,'lfom':l,'base_ce':d[b].ce,'lfom_ce':d[l].ce,'mha_continue_ce':d['MHA_continue'].ce,'lfom_beats_mha_continue':d[l].ce<d['MHA_continue'].ce,'lfom_beats_base':d[l].ce<d[b].ce})
    pair=pd.DataFrame(paired); pair.to_csv(OUT/'paired.csv',index=False)
    report='# Ultra fair LFOM-GQA frontier check\n\n'+summ.to_markdown(index=False,floatfmt='.4f')+'\n\n## Paired\n\n'+pair.groupby(['T','base']).agg(lfom_ce=('lfom_ce','mean'),base_ce=('base_ce','mean'),mha_continue_ce=('mha_continue_ce','mean'),lfom_beats_mha_continue=('lfom_beats_mha_continue','sum'),lfom_beats_base=('lfom_beats_base','sum')).reset_index().to_markdown(index=False,floatfmt='.4f')+'\n'
    (OUT/'REPORT.md').write_text(report)
    plt.figure(figsize=(7,3)); sub=summ[summ.T==80]; plt.bar(sub.method,sub.ce); plt.xticks(rotation=45,ha='right'); plt.ylabel('CE T=80'); plt.tight_layout(); plt.savefig(OUT/'figures/ce_T80.png',dpi=150)
    with zipfile.ZipFile('/mnt/data/lfom_gqa_fair_frontier_ultra_package.zip','w',zipfile.ZIP_DEFLATED) as z:
      for p in OUT.rglob('*'): z.write(p,p.relative_to(OUT.parent)); z.write('/mnt/data/run_lfom_gqa_fair_frontier_ultra.py','run_lfom_gqa_fair_frontier_ultra.py')
    print(summ)
if __name__=='__main__': main()
