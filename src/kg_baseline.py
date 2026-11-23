"""Baseline graf pengetahuan: titik acuan ATAS untuk kedua tugas.

Pertanyaan yang dijawab: 0,700 pada tugas intra-kelas itu bagus atau buruk?
Tanpa metode non-LLM yang kuat pada 1.034 kelompok yang SAMA, angka itu tidak
terkalibrasi, dan reviewer pasti menanyakannya.

Graf: bipartit obat--penyakit dari indikasi DrugCentral (9.682 sisi).

KEBOCORAN. Positif di set uji ADALAH sisi di graf itu. Melatih pada graf utuh
berarti menghafal jawabannya. Jadi SELURUH sisi positif dari kedua set uji
dibuang sebelum metode apa pun dijalankan -- protokol link-prediction baku.

Satu asimetri yang tersisa dan sengaja dibiarkan karena arahnya konservatif:
membuang sisi positif menurunkan derajat obat positif sebanyak satu, sementara
obat negatif tidak kehilangan apa pun. Set uji dicocokkan pada derajat
SEBENARNYA, jadi di graf latih derajat positif justru sistematis lebih rendah.
Itu membebani metode KG, bukan menguntungkannya.

Metode (semuanya numpy/scipy, jalan di laptop dalam hitungan detik):
  AA    Adamic-Adar pada tetangga bersama obat--obat
  Katz  jumlah lintasan berbobot, beta=0,05, dipotong di panjang 5
  SVD   faktorisasi matriks insidensi terpotong, rank 64
  TGT   propagasi kemiripan target molekuler ke penyakit
  KOMBINASI  regresi logistik atas keempatnya, CV dikelompokkan per obat
"""
import sys, os, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from pgcopy import read_copy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

DC='data/interim/drugcentral/'; B=2000
rng=np.random.default_rng(20261102)

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
def macro_ci(v):
    v=np.asarray(v); n=len(v)
    d=np.array([v[rng.integers(0,n,n)].mean() for _ in range(B)])
    return v.mean(), np.percentile(d,2.5), np.percentile(d,97.5), n

# ---------- graf ----------
_,omop=read_copy(DC+'omop_relationship.sql')
edges={(r['struct_id'],r['umls_cui']) for r in omop
       if r['relationship_name']=='indication' and r['umls_cui']}
CLS=pd.read_csv('data/processed/pilot_pairs_v5.tsv',sep='\t')
CLS=CLS[CLS.tier.isin(['pos','neg_random'])].reset_index(drop=True)
WIT=pd.read_csv('data/processed/within_class_pairs.tsv',sep='\t').reset_index(drop=True)

# Dedup pasangan ganda SEKALI di hulu, sama persis dengan yang dilakukan
# src/gpu_phase2.py sebelum inferensi. Tanpa ini baseline dinilai pada 1.034
# kelompok sementara LLM pada 1.017, dan perbandingannya tidak setara.
_n0=(len(CLS),len(WIT))
CLS=CLS.drop_duplicates(['drugbank_id','umls_cui']).reset_index(drop=True)
WIT=WIT.drop_duplicates(['drugbank_id','umls_cui']).reset_index(drop=True)
print(f"dedup: tingkat-kelas {_n0[0]}->{len(CLS)}, intra-kelas {_n0[1]}->{len(WIT)}")
drop={(str(r.struct_id),r.umls_cui) for r in CLS.itertuples() if r.label==1}
drop|={(str(r.struct_id),r.umls_cui) for r in WIT.itertuples() if r.label==1}
train=edges-drop
print(f"sisi total={len(edges)}  dibuang (positif uji)={len(edges&drop)}  sisa latih={len(train)}")

D=sorted({s for s,_ in edges}|{str(x) for x in CLS.struct_id}|{str(x) for x in WIT.struct_id})
Y=sorted({c for _,c in edges}|set(CLS.umls_cui)|set(WIT.umls_cui))
di={d:i for i,d in enumerate(D)}; yi={c:i for i,c in enumerate(Y)}
A=np.zeros((len(D),len(Y)),dtype=np.float32)
for s,c in train:
    if s in di and c in yi: A[di[s],yi[c]]=1
print(f"matriks insidensi: {A.shape}, kepadatan {A.mean()*100:.3f}%")

# ---------- metode ----------
kd=A.sum(1); ky=A.sum(0)
# Adamic-Adar: obat yang berbagi penyakit langka lebih mirip daripada yg berbagi penyakit umum
wy=1.0/np.log(np.maximum(ky,2.0))
SIM=(A*wy)@A.T                      # kemiripan obat-obat
np.fill_diagonal(SIM,0)
S_AA=SIM@A

# Katz pada bipartit, dipotong di lintasan panjang 5
beta=0.05; P=A.copy(); S_KATZ=beta*A.copy()
for _ in range(2):
    P=A@(A.T@P); S_KATZ+=(beta**3)*P if _==0 else (beta**5)*P

# SVD terpotong
U,sv,Vt=np.linalg.svd(A,full_matrices=False)
k=64; S_SVD=(U[:,:k]*sv[:k])@Vt[:k]

# propagasi kemiripan target
_,act=read_copy(DC+'act_table_full.sql')
tg=collections.defaultdict(set)
for r in act:
    if r.get('accession') and r['struct_id'] in di: tg[r['struct_id']].add(r['accession'])
T=sorted({a for v in tg.values() for a in v}); ti={a:i for i,a in enumerate(T)}
M=np.zeros((len(D),len(T)),dtype=np.float32)
for s,v in tg.items():
    for a in v: M[di[s],ti[a]]=1
nrm=np.linalg.norm(M,axis=1,keepdims=True); nrm[nrm==0]=1
Mn=M/nrm; TSIM=Mn@Mn.T; np.fill_diagonal(TSIM,0)
S_TGT=TSIM@A
print(f"obat dengan target terdaftar: {(M.sum(1)>0).sum()}/{len(D)}")

METH=[('AA  Adamic-Adar',S_AA),('Katz (beta=0,05)',S_KATZ),
      ('SVD (rank 64)',S_SVD),('TGT propagasi target',S_TGT)]

def sc(S,df):
    return np.array([S[di[str(s)],yi[c]] if str(s) in di and c in yi else 0.0
                     for s,c in zip(df.struct_id,df.umls_cui)])

GK={(r.drugbank_id,r.umls_cui):f"{r.umls_cui}|{r.shared_class}" for r in WIT.itertuples()}
OUT=[]
for nama,df,within in (('tingkat-kelas',CLS,False),('intra-kelas',WIT,True)):
    key=[GK[(a,b)] for a,b in zip(df.drugbank_id,df.umls_cui)] if within else df.drugbank_id.values
    y=df.label.values
    print(f"\n{'='*66}\nTUGAS {nama.upper()}\n{'='*66}")
    F=[]
    for nm,S in METH:
        s=sc(S,df); F.append(s)
        a,lo,hi,n=macro_ci(per_group(y,key,s).values)
        print(f"  {nm:<24} {a:.3f} [{lo:.3f}-{hi:.3f}]  n={n}")
        OUT.append(dict(tugas=nama,metode=nm,auroc=round(a,4),lo=round(lo,4),hi=round(hi,4)))
    X=np.column_stack(F); oof=np.zeros(len(df))
    for tr,te in GroupKFold(5).split(X,y,df.drugbank_id.values):
        st=StandardScaler().fit(X[tr])
        oof[te]=LogisticRegression(max_iter=2000).fit(st.transform(X[tr]),y[tr]
                ).predict_proba(st.transform(X[te]))[:,1]
    a,lo,hi,n=macro_ci(per_group(y,key,oof).values)
    print(f"  {'KOMBINASI (regresi log.)':<24} {a:.3f} [{lo:.3f}-{hi:.3f}]  n={n}  <- terkuat")
    OUT.append(dict(tugas=nama,metode='KG kombinasi',auroc=round(a,4),lo=round(lo,4),hi=round(hi,4)))
    np.save(f'results/kg_oof_{"wit" if within else "cls"}.npy',oof)
pd.DataFrame(OUT).to_csv('results/kg_baseline.csv',index=False)
print("\nditulis: results/kg_baseline.csv + skor OOF")
