"""Analisis makro untuk keluarga model apa pun (tidak terikat OLMo).

  python3 src/analyze_family.py \
      --class-files "Qwen 8B=results/cls_qwen3_8b.jsonl" "Qwen 14B=..." \
      --within-files "Qwen 8B=results/wit_qwen3_8b.jsonl" ...

Metrik identik dengan src/rebuild_outputs.py: AUROC makro dalam-kelompok,
per template lalu dirata-ratakan, CI bootstrap pada tingkat kelompok.
"""
import argparse, json, numpy as np, pandas as pd
ap=argparse.ArgumentParser()
ap.add_argument('--class-files',nargs='*',default=[])
ap.add_argument('--within-files',nargs='*',default=[])
ap.add_argument('--cond',default='A0')
ap.add_argument('--out',default='results/family_metrics.csv')
A=ap.parse_args()
B=2000; TPL=['qa','tf','cloze']; rng=np.random.default_rng(20261102)

def auroc(y,s):
    y=np.asarray(y); s=np.asarray(s,float); n1=y.sum(); n0=len(y)-n1
    if n1==0 or n0==0: return np.nan
    r=pd.Series(s).rank().values
    return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
def per_group(y,g,s):
    df=pd.DataFrame({'y':y,'g':g,'s':s}); k=[];v=[]
    for key,gg in df.groupby('g',sort=True):
        if gg.y.nunique()==2: k.append(key); v.append(auroc(gg.y.values,gg.s.values))
    return pd.Series(v,index=k)
def summarize(M):
    pt=M.mean(axis=0); n=M.shape[0]
    d=np.array([M[rng.integers(0,n,n)].mean(axis=0).mean() for _ in range(B)])
    return pt.mean(),np.percentile(d,2.5),np.percentile(d,97.5),pt,n

WIT=pd.read_csv('data/processed/within_class_pairs.tsv',sep='\t')
GK={(r.drugbank_id,r.umls_cui):f"{r.umls_cui}|{r.shared_class}" for r in WIT.itertuples()}

def load(path,within):
    d=pd.DataFrame([json.loads(l) for l in open(path)])
    g=d[d.condition==A.cond] if within else pd.concat(
        [d[d.tier=='pos'],d[d.tier=='neg_random']]).query(f"condition=='{A.cond}'")
    return g.drop_duplicates(['drugbank_id','umls_cui','template'])

rows=[]
for task,spec,within in (('tingkat-kelas',A.class_files,False),('intra-kelas',A.within_files,True)):
    if not spec: continue
    print(f"\n=== {task} (kondisi {A.cond}) ===")
    print(f"{'model':<24} {'qa':>7} {'tf':>7} {'cloze':>7} | {'rerata':>7}  CI 95%           n klp")
    for item in spec:
        name,path=item.split('=',1)
        g=load(path,within); cols={}
        for t in TPL:
            gg=g[g.template==t]
            key=[GK[(d,c)] for d,c in zip(gg.drugbank_id,gg.umls_cui)] if within else gg.drugbank_id.values
            cols[t]=per_group(gg.label.values,key,gg.score.values)
        M=pd.DataFrame(cols).dropna()
        if M.empty: print(f"{name:<24} (tidak ada kelompok valid)"); continue
        pt,lo,hi,per,n=summarize(M.values)
        print(f"{name:<24} "+" ".join(f"{v:>7.3f}" for v in per)+
              f" | {pt:>7.3f}  [{lo:.3f}-{hi:.3f}] {n:>6}")
        rows.append(dict(tugas=task,model=name,qa=round(per[0],4),tf=round(per[1],4),
                         cloze=round(per[2],4),rerata=round(pt,4),lo=round(lo,4),
                         hi=round(hi,4),n_klp=n))
if rows:
    pd.DataFrame(rows).to_csv(A.out,index=False); print(f"\nditulis: {A.out}")
