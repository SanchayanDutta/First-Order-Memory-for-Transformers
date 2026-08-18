# chunked wrapper imports same definitions by executing file without main then runs missing pairs
import importlib.util, sys, pandas as pd
from pathlib import Path
spec=importlib.util.spec_from_file_location('exp','/mnt/data/run_lfom_gqa_fast_extra_final.py')
mod=importlib.util.module_from_spec(spec)
# disable main guard fine
spec.loader.exec_module(mod)
OUT=mod.OUT
raw=OUT/'raw_partial.csv'
if raw.exists(): rows=pd.read_csv(raw).to_dict('records')
else: rows=[]
done={(int(r['seed']),r['method']) for r in rows}
data=mod.Data(); methods=['MHA','GQA','GQA+MLP','GQA+LFOM','MQA','MQA+MLP','MQA+LFOM']
count=0
for seed in range(6):
    for method in methods:
        if (seed,method) in done: continue
        rows.append(mod.train(method,seed,data)); pd.DataFrame(rows).to_csv(raw,index=False)
        count+=1
        if count>=5:
            raise SystemExit
# finalize
import numpy as np, matplotlib.pyplot as plt, zipfile
OUT.mkdir(exist_ok=True)
df=pd.DataFrame(rows); df.to_csv(OUT/'raw.csv',index=False)
summ=df.groupby('method').agg(ce64=('ce64','mean'),ce64_std=('ce64','std'),acc64=('acc64','mean'),ce80=('ce80','mean'),ce80_std=('ce80','std'),acc80=('acc80','mean'),cache=('cache','mean'),repair_params=('repair_params','mean')).reset_index().sort_values('ce80'); summ.to_csv(OUT/'summary.csv',index=False)
comps=[]
for seed,g in df.groupby('seed'):
    if len(g)<len(methods): continue
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
    for p in OUT.rglob('*'): z.write(p,p.relative_to(OUT.parent)); z.write('/mnt/data/run_lfom_gqa_fast_extra_final.py','run_lfom_gqa_fast_extra_final.py'); z.write('/mnt/data/run_lfom_gqa_fast_extra_chunk.py','run_lfom_gqa_fast_extra_chunk.py')
print('FINAL'); print(summ.to_string(index=False)); print(ps.to_string(index=False))
