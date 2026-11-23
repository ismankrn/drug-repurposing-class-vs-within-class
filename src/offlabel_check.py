"""Uji sensitivitas: sebagian negatif ternyata tercatat sebagai off-label use.

Sampler negatif menolak indikasi dan kontraindikasi, tetapi TIDAK menolak
'off-label use'. Jadi ada negatif yang obatnya memang dipakai untuk penyakit
itu, hanya belum disetujui. Secara definisi tugas (disetujui vs tidak) mereka
negatif yang benar, tapi besarnya pengaruh harus diukur, bukan diasumsikan
kecil. Keluaran: results/offlabel_check.csv
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from pgcopy import read_copy

_,om=read_copy('data/interim/drugcentral/omop_relationship.sql')
off={(str(r['struct_id']),r['umls_cui']) for r in om
     if r['relationship_name']=='off-label use' and r['umls_cui']}
W=pd.read_csv('data/processed/within_class_pairs.tsv',sep='\t').drop_duplicates(
    ['drugbank_id','umls_cui'])
GK={(r.drugbank_id,r.umls_cui):f"{r.umls_cui}|{r.shared_class}" for r in W.itertuples()}
BAD={(r.drugbank_id,r.umls_cui) for r in W.itertuples()
     if r.label==0 and (str(r.struct_id),r.umls_cui) in off}
NEG=int((W.label==0).sum())
print(f"negatif intra-kelas yang tercatat off-label: {len(BAD)} dari {NEG} "
      f"({100*len(BAD)/NEG:.1f}%)")

def auroc(y,s):
    y=np.asarray(y); s=np.asarray(s,float); n1=y.sum(); n0=len(y)-n1
    if n1==0 or n0==0: return np.nan
    r=pd.Series(s).rank().values
    return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)

def macro(f,drop):
    d=pd.DataFrame([json.loads(l) for l in open(f)])
    g=d[d.condition=='A0'].drop_duplicates(['drugbank_id','umls_cui','template'])
    if drop:
        keep=[(a,b) not in BAD for a,b in zip(g.drugbank_id,g.umls_cui)]
        g=g[keep]
    per=[]
    for t in ('qa','tf','cloze'):
        h=g[g.template==t].copy()
        h['k']=[GK[(a,b)] for a,b in zip(h.drugbank_id,h.umls_cui)]
        v=[auroc(x.label.values,x.score.values) for _,x in h.groupby('k')
           if x.label.nunique()==2]
        per.append(np.nanmean(v))
    return float(np.mean(per))

MOD=[('OLMo-2 7B','results/scores_within.jsonl'),
     ('OLMo-2 13B','results/within_OLMo-2-1124-13B-Instruct.jsonl'),
     ('OLMo-2 32B','results/within_OLMo-2-0325-32B-Instruct.jsonl'),
     ('Qwen3 8B','results/wit_Qwen3-8B.jsonl'),
     ('Qwen3 32B','results/wit_Qwen3-32B.jsonl')]
rows=[]
for nm,f in MOD:
    a=macro(f,False); b=macro(f,True)
    print(f"  {nm:12} dengan={a:.3f}  tanpa={b:.3f}  selisih={b-a:+.3f}")
    rows.append(dict(model=nm,with_offlabel=round(a,4),without=round(b,4),
                     delta=round(b-a,4)))
pd.DataFrame(rows).to_csv('results/offlabel_check.csv',index=False)
print(f"ditulis: results/offlabel_check.csv  (n_dibuang={len(BAD)})")
