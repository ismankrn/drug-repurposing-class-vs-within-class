"""Metrik final bebas-konfound, LLM vs baseline lookup.

Dua koreksi atas analisis sebelumnya:

(1) AUROC GLOBAL PUNYA KOMPONEN ANTAR-OBAT. Mengacak label di dalam tiap obat
    tetap menyisakan AUROC 0,55-0,66 (bukan 0,50), karena obat yang berbeda
    menyumbang campuran positif/negatif yang berbeda. Lantai itu BERBEDA per
    model (1B 0,547 vs 32B 0,657), jadi membandingkan AUROC global antar model
    bukan perbandingan setara. Solusi: AUROC dihitung DI DALAM kelompok lalu
    dirata-ratakan (makro).
      - tugas tingkat-kelas : kelompok = obat
      - tugas intra-kelas   : kelompok = (penyakit, kelas EPC), persis unit
                              yang dicocokkan saat set itu dibangun
(2) TIGA TEMPLATE DINILAI TERPISAH; skor log-prob antar template tidak sebanding.
"""
import sys, os, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
rng=np.random.default_rng(20261102); B=2000; TPL=['qa','tf','cloze']

def auroc(y,s):
    y=np.asarray(y); s=np.asarray(s,float); n1=y.sum(); n0=len(y)-n1
    if n1==0 or n0==0: return np.nan
    r=pd.Series(s).rank().values
    return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)

def macro(y,g,s):
    df=pd.DataFrame({'y':y,'g':g,'s':s}); a=[]
    for _,gg in df.groupby('g',sort=False):
        if gg.y.nunique()==2: a.append(auroc(gg.y.values,gg.s.values))
    return float(np.mean(a)), len(a)

def macro_ci(y,g,s):
    """Bootstrap pada tingkat KELOMPOK: resample kelompok, rata-rata ulang."""
    df=pd.DataFrame({'y':y,'g':g,'s':s}); per={}
    for k,gg in df.groupby('g',sort=False):
        if gg.y.nunique()==2: per[k]=auroc(gg.y.values,gg.s.values)
    v=np.array(list(per.values())); o=[v[rng.integers(0,len(v),len(v))].mean() for _ in range(B)]
    return float(v.mean()), float(np.percentile(o,2.5)), float(np.percentile(o,97.5)), len(v)

E={'1B':'results/easy_OLMo-2-0425-1B.jsonl','7B':'results/scores_5cond.jsonl',
   '13B':'results/easy_OLMo-2-1124-13B-Instruct.jsonl','32B':'results/easy_OLMo-2-0325-32B-Instruct.jsonl'}
H={'1B':'results/within_OLMo-2-0425-1B.jsonl','7B':'results/scores_within.jsonl',
   '13B':'results/within_OLMo-2-1124-13B-Instruct.jsonl','32B':'results/within_OLMo-2-0325-32B-Instruct.jsonl'}
MS=['1B','7B','13B','32B']

def llm(f,within,cond='A0'):
    d=pd.DataFrame([json.loads(l) for l in open(f)])
    g=d[d.condition==cond] if within else pd.concat(
        [d[d.tier=='pos'],d[d.tier=='neg_random']]).query(f"condition=='{cond}'")
    return g.drop_duplicates(['drugbank_id','umls_cui','template'])

# kunci kelompok utk set intra-kelas
WIT=pd.read_csv('data/processed/within_class_pairs.tsv',sep='\t')
GKEY={(r.drugbank_id,r.umls_cui):f"{r.umls_cui}|{r.shared_class}" for r in WIT.itertuples()}

OUT=[]
print(f"{'='*86}\nAUROC MAKRO DALAM-KELOMPOK (bebas konfound antar-obat/antar-penyakit)\n{'='*86}")
for nama,FILES,within in (('tingkat-kelas',E,False),('intra-kelas',H,True)):
    print(f"\n--- TUGAS {nama.upper()} ---")
    print(f"{'model':<6} {'qa':>7} {'tf':>7} {'cloze':>7} | {'rerata':>7}  {'terbaik + CI 95%':<24} {'n klp':>6}")
    for m in MS:
        g=llm(FILES[m],within); per=[]; ci=None
        for t in TPL:
            gg=g[g.template==t]
            grp=[GKEY[(d,c)] for d,c in zip(gg.drugbank_id,gg.umls_cui)] if within else gg.drugbank_id.values
            a,lo,hi,n=macro_ci(gg.label.values,grp,gg.score.values)
            per.append(a)
            if a==max(per): ci=(lo,hi,n,t)
        lo,hi,n,bt=ci
        print(f"{m:<6} "+" ".join(f"{p:>7.3f}" for p in per)+
              f" | {np.mean(per):>7.3f}  {max(per):.3f} [{lo:.3f}-{hi:.3f}] ({bt:<5}) {n:>6}")
        OUT.append(dict(tugas=nama,sistem=f'OLMo-2 {m}',qa=round(per[0],4),tf=round(per[1],4),
                        cloze=round(per[2],4),rerata=round(float(np.mean(per)),4),
                        terbaik=round(max(per),4),lo=round(lo,4),hi=round(hi,4),n_klp=n))
pd.DataFrame(OUT).to_csv('results/final_metrics_llm.csv',index=False)
print("\nditulis: results/final_metrics_llm.csv")
