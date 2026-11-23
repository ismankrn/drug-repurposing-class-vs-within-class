"""Validasi (1): apa yang diukur benchmark bila negatifnya TIDAK dicocokkan derajat.

pilot_pairs_v2.tsv (set tak tercocok yang asli) sudah tidak ada, dan lagi pula
v2 memakai negatif kontraindikasi -- membandingkannya dengan v5 mengubah dua
hal sekaligus. Di sini pembandingnya dibangun ulang sehingga HANYA cara
sampling negatif yang berbeda:

  tak tercocok : obat dan penyakit diambil seragam atas entitas BERBEDA,
                 persis cara benchmark naif menyusun negatif acak
  tercocok     : set yang dipakai studi ini (pilot_pairs_v5, tier neg_random)

Positifnya identik pada keduanya. Keluaran: results/degree_check.csv
"""
import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from pgcopy import read_copy

rng=np.random.default_rng(20261103)
def auroc(y,s):
    y=np.asarray(y); s=np.asarray(s,float); n1=y.sum(); n0=len(y)-n1
    r=pd.Series(s).rank().values
    return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)

_,om=read_copy('data/interim/drugcentral/omop_relationship.sql')
ind={(r['struct_id'],r['umls_cui']) for r in om
     if r['relationship_name']=='indication' and r['umls_cui']}
# Sampler benchmark yang asli (src/build_pilot_set_v2.py) menolak indikasi DAN
# kontraindikasi. Aturan tolak di sini harus sama, kalau tidak kedua baris
# tabel berbeda pada DUA hal dan perbandingannya tidak mengisolasi pencocokan.
contra={(r['struct_id'],r['umls_cui']) for r in om
        if r['relationship_name']=='contraindication' and r['umls_cui']}
dd=collections.Counter(s for s,_ in ind); sd=collections.Counter(c for _,c in ind)
drugs=sorted(dd); dis=sorted(sd)

v5=pd.read_csv('data/processed/pilot_pairs_v5.tsv',sep='\t')
cls=v5[v5.tier.isin(['pos','neg_random'])].drop_duplicates(['drugbank_id','umls_cui'])
pos=cls[cls.label==1]; neg_m=cls[cls.label==0]
print(f"positif={len(pos)}  negatif tercocok={len(neg_m)}")

# negatif TAK tercocok: seragam atas obat berbeda x penyakit berbeda
rows=[]
while len(rows)<len(pos):
    s=drugs[rng.integers(len(drugs))]; c=dis[rng.integers(len(dis))]
    if (s,c) in ind or (s,c) in contra: continue
    rows.append({'label':0,'drug_degree':dd[s],'disease_degree':sd[c]})
neg_u=pd.DataFrame(rows)

out=[]
for nm,neg in (('tak tercocok',neg_u),('tercocok',neg_m)):
    d=pd.concat([pos[['label','drug_degree','disease_degree']],
                 neg[['label','drug_degree','disease_degree']]],ignore_index=True)
    a=auroc(d.label,d.drug_degree); b=auroc(d.label,d.disease_degree)
    print(f"  {nm:14} derajat obat {a:.3f} | derajat penyakit {b:.3f}")
    out.append(dict(set=nm,drug_degree_auroc=round(a,4),disease_degree_auroc=round(b,4),
                    n_pos=len(pos),n_neg=len(neg)))
pd.DataFrame(out).to_csv('results/degree_check.csv',index=False)
print("ditulis: results/degree_check.csv")

# ---------- tabel LaTeX ----------
t=["\\begin{table}[!tbp]","\\centering","\\small",
   "\\caption{Degree features under the two ways of sampling negative instances.}",
   "\\label{tab:degree}","\\begin{tabular}{@{}lcc@{}}","\\toprule",
   # Tanpa kepala rentang ini, "Drug degree & 0.838" terbaca sebagai NILAI
   # derajatnya, padahal derajat itu cacahan indikasi, bukan angka di [0,1].
   "& \\multicolumn{2}{c}{\\auroc{} of the feature alone (no model)} \\\\",
   "\\cmidrule(l){2-3}",
   "How the negative pairs are chosen & Drug degree & Disease degree \\\\","\\midrule"]
# Label baris menyatakan APA YANG DILAKUKAN, bukan sekadar namanya: pembaca
# tidak bisa menebak arti "unmatched" tanpa membaca Metode.
LBL={'tak tercocok':'Drug and disease drawn uniformly (naive)',
     'tercocok':'Both degrees matched to the positive (this study)'}
for r in out:
    t.append(f"{LBL[r['set']]} & {r['drug_degree_auroc']:.3f} & {r['disease_degree_auroc']:.3f} \\\\")
t+=["\\bottomrule","\\end{tabular}","\\end{table}"]
open('paper/tables/tab-degree.tex','w').write("\n".join(t)+"\n")
print("ditulis: paper/tables/tab-degree.tex")
