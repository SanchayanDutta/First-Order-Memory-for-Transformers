import os, re, math, time, json, zipfile
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt

torch.set_num_threads(4)
OUT=Path('/mnt/data/lfom_gqa_extra_final_fast'); OUT.mkdir(parents=True,exist_ok=True); (OUT/'figures').mkdir(exist_ok=True)

def collect_text(max_files=120,max_chars=800000):
    chunks=[]; total=0; n=0
    for p in Path('/mnt/data').rglob('*'):
        if p.is_file() and p.suffix.lower() in {'.md','.tex','.py','.txt'} and p.stat().st_size<600000:
            try: s=p.read_text(errors='ignore')
            except: continue
            if len(s)<400: continue
            s=re.sub(r'/mnt/data/[^\s]+',' PATH ',s)
            chunks.append(s[:25000]); total+=min(len(s),25000); n+=1
            if n>=max_files or total>=max_chars: break
    arr=[]
    for ch in '\n'.join(chunks):
        o=ord(ch); arr.append(o if (ch=='\n' or ch=='\t' or 32<=o<=126) else 32)
    return np.array(arr,dtype=np.int64), {'files':n,'chars':len(arr)}
class Data:
    def __init__(self,N=240000,V=60000):
        arr,meta=collect_text(); self.meta=meta
        if len(arr)<N+V+100: arr=np.tile(arr,(N+V+100)//len(arr)+1)
        self.train=torch.tensor(arr[:N]); self.val=torch.tensor(arr[N:N+V]); self.vocab=128
    def batch(self,split,B,T):
        a=self.train if split=='train' else self.val
        st=torch.randint(0,len(a)-T-1,(B,))
        return torch.stack([a[s:s+T] for s in st]), torch.stack([a[s+1:s+T+1] for s in st])
class Att(nn.Module):
    def __init__(self,d=64,h=4,kv=4,repair='none',rank=8):
        super().__init__(); self.h=h; self.kv=kv; self.hd=d//h; self.repair=repair; self.rank=rank
        self.q=nn.Linear(d,d,bias=False); self.k=nn.Linear(d,kv*self.hd,bias=False); self.v=nn.Linear(d,kv*self.hd,bias=False); self.o=nn.Linear(d,d,bias=False)
        if repair=='lfom':
            self.u=nn.Linear(d,h*rank); self.g=nn.Linear(d,h*rank); self.e=nn.Linear(d,h*rank); self.r=nn.Linear(rank,self.hd,bias=False); self.mix=nn.Parameter(torch.tensor(0.14)); self.norm=nn.LayerNorm(d)
        elif repair=='mlp':
            self.mlp=nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,d)); self.mix=nn.Parameter(torch.tensor(0.1)); self.norm=nn.LayerNorm(d)
    def forward(self,x):
        B,T,D=x.shape; H=self.h; hd=self.hd; kv=self.kv
        q=self.q(x).view(B,T,H,hd).transpose(1,2); k=self.k(x).view(B,T,kv,hd).transpose(1,2); v=self.v(x).view(B,T,kv,hd).transpose(1,2)
        if kv!=H:
            k=k.repeat_interleave(H//kv,1); v=v.repeat_interleave(H//kv,1)
        a=(q@k.transpose(-2,-1))/math.sqrt(hd); mask=torch.triu(torch.ones(T,T,dtype=torch.bool,device=x.device),1)
        a=a.masked_fill(mask[None,None],-1e4).softmax(-1); out=(a@v).transpose(1,2).contiguous().view(B,T,D); out=self.o(out)
        if self.repair=='lfom':
            u=torch.tanh(self.u(x)).view(B,T,H,self.rank); g=torch.sigmoid(self.g(x)).view(B,T,H,self.rank); e=torch.tanh(self.e(x)).view(B,T,H,self.rank)
            st=torch.zeros(B,H,self.rank,device=x.device); prev=torch.zeros_like(st); outs=[]
            for t in range(T):
                corr=u[:,t]+0.35*(u[:,t]-prev)+0.15*e[:,t]
                st=g[:,t]*st+(1-g[:,t])*corr; prev=u[:,t]
                outs.append(self.r(st).reshape(B,D))
            out=out+self.mix*self.norm(torch.stack(outs,1))
        elif self.repair=='mlp':
            out=out+self.mix*self.norm(self.mlp(x))
        return out
class Model(nn.Module):
    def __init__(self,vocab=128,kv=4,repair='none',T=64,d=64):
        super().__init__(); self.emb=nn.Embedding(vocab,d); self.pos=nn.Parameter(torch.zeros(96,d)); self.ln1=nn.LayerNorm(d); self.att=Att(d,4,kv,repair); self.ln2=nn.LayerNorm(d); self.ff=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d)); self.ln=nn.LayerNorm(d); self.head=nn.Linear(d,vocab,bias=False)
    def forward(self,x):
        z=self.emb(x)+self.pos[:x.shape[1]]; z=z+self.att(self.ln1(z)); z=z+self.ff(self.ln2(z)); return self.head(self.ln(z))
def eval_m(m,data,T=64,steps=14):
    m.eval(); L=[]; A=[]
    with torch.no_grad():
        for _ in range(steps):
            x,y=data.batch('val',24,T); log=m(x); L.append(F.cross_entropy(log.reshape(-1,data.vocab),y.reshape(-1)).item()); A.append((log.argmax(-1)==y).float().mean().item())
    return float(np.mean(L)),float(np.mean(A))
def train(method,seed,data,steps=120):
    torch.manual_seed(555+seed); kv,rep={'MHA':(4,'none'),'GQA':(2,'none'),'GQA+MLP':(2,'mlp'),'GQA+LFOM':(2,'lfom'),'MQA':(1,'none'),'MQA+MLP':(1,'mlp'),'MQA+LFOM':(1,'lfom')}[method]
    m=Model(data.vocab,kv,rep); opt=torch.optim.AdamW(m.parameters(),lr=3e-3,weight_decay=0.01); t0=time.time()
    for s in range(steps):
        x,y=data.batch('train',24,64); log=m(x); loss=F.cross_entropy(log.reshape(-1,data.vocab),y.reshape(-1)); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    ce64,acc64=eval_m(m,data,64); ce80,acc80=eval_m(m,data,80,10)
    rep_params=sum(p.numel() for n,p in m.named_parameters() if ('.u.' in n or '.g.' in n or '.e.' in n or '.r.' in n or '.mlp.' in n))
    row={'method':method,'seed':seed,'ce64':ce64,'acc64':acc64,'ce80':ce80,'acc80':acc80,'cache':kv/4,'repair_params':rep_params,'time':time.time()-t0}
    print(row,flush=True); return row

def main():
    data=Data(); (OUT/'meta.json').write_text(json.dumps(data.meta)); methods=['MHA','GQA','GQA+MLP','GQA+LFOM','MQA','MQA+MLP','MQA+LFOM']; rows=[]
    for seed in range(6):
        for method in methods:
            r=train(method,seed,data); rows.append(r); pd.DataFrame(rows).to_csv(OUT/'raw_partial.csv',index=False)
    df=pd.DataFrame(rows); df.to_csv(OUT/'raw.csv',index=False)
    summ=df.groupby('method').agg(ce64=('ce64','mean'),ce64_std=('ce64','std'),acc64=('acc64','mean'),ce80=('ce80','mean'),ce80_std=('ce80','std'),acc80=('acc80','mean'),cache=('cache','mean'),repair_params=('repair_params','mean')).reset_index().sort_values('ce80'); summ.to_csv(OUT/'summary.csv',index=False)
    comps=[]
    for seed,g in df.groupby('seed'):
        D={r.method:r for r in g.itertuples()}
        for metric in ['ce64','ce80','acc64','acc80']:
            for a,b in [('GQA+LFOM','GQA+MLP'),('MQA+LFOM','MQA+MLP'),('GQA+LFOM','GQA'),('MQA+LFOM','MQA'),('MQA+LFOM','MHA'),('GQA+LFOM','MHA')]:
                va=getattr(D[a],metric); vb=getattr(D[b],metric); win=va<vb if metric.startswith('ce') else va>vb
                comps.append({'seed':seed,'metric':metric,'a':a,'b':b,'win':win,'delta':va-vb,'a_val':va,'b_val':vb})
    paired=pd.DataFrame(comps); paired.to_csv(OUT/'paired.csv',index=False); ps=paired.groupby(['metric','a','b']).agg(wins=('win','sum'),n=('win','count'),mean_delta=('delta','mean')).reset_index(); ps.to_csv(OUT/'paired_summary.csv',index=False)
    report='# LFOM-GQA/MQA extra final fast run\n\n'+summ.to_markdown(index=False,floatfmt='.4f')+'\n\n## Paired wins\n\n'+ps.to_markdown(index=False,floatfmt='.4f')+'\n'
    (OUT/'REPORT.md').write_text(report)
    plt.figure(figsize=(7,3.5)); plt.bar(summ.method,summ.ce80); plt.xticks(rotation=25,ha='right'); plt.ylabel('CE at T=80'); plt.tight_layout(); plt.savefig(OUT/'figures/ce80.png',dpi=180)
    with zipfile.ZipFile('/mnt/data/lfom_gqa_extra_final_fast_package.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'): z.write(p,p.relative_to(OUT.parent)); z.write('/mnt/data/run_lfom_gqa_fast_extra_final.py','run_lfom_gqa_fast_extra_final.py')
    print(summ.to_string(index=False)); print(ps.to_string(index=False))
if __name__=='__main__': main()
