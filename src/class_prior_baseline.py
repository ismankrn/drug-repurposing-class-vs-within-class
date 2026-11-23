"""Baseline prior-kelas: bisakah tabel lookup sederhana menandingi LLM?

Pertanyaan reviewer yang paling berbahaya: kalau A5c (kelas farmakologis saja,
tanpa nama obat) = 0,742, apakah tabel "seberapa sering kelas X diindikasikan
untuk penyakit Y" juga mencapai ~0,74? Kalau ya, LLM tidak menyumbang apa pun
di atas prior kelas.

Semua prior dihitung LEAVE-ONE-DRUG-OUT: saat menilai pasangan (obat d,
penyakit y), obat d dikeluarkan dari pembilang DAN penyebut, dan kontribusinya
dihapus dari derajat penyakit. Tanpa itu baseline bocor dan jadi trivial.

Universe prior = himpunan indikasi DrugCentral omop_relationship -- sumber yang
sama dengan yang melahirkan positif di set uji, jadi perbandingannya adil
(bahkan menguntungkan baseline).
"""
import sys, os, collections, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from pgcopy import read_copy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

DC='data/interim/drugcentral/'
ALPHA=5.0                       # smoothing prior kelas
rng=np.random.default_rng(20261102); B=2000

# ---------- metrik ----------
def auroc(y,s):
    y=np.asarray(y); s=np.asarray(s,dtype=float); n1=y.sum(); n0=len(y)-n1
    if n1==0 or n0==0: return np.nan
    r=pd.Series(s).rank().values                 # rank rata-rata -> seri dihitung 0,5
    return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)

def boot_ci(df,score):
    g=collections.defaultdict(list)
    for i,c in enumerate(df['drugbank_id'].values): g[c].append(i)
    k=list(g); idx={a:np.array(b) for a,b in g.items()}
    y=df['label'].values; s=np.asarray(score,dtype=float); o=[]
    for _ in range(B):
        p=rng.choice(len(k),len(k),replace=True)
        sel=np.concatenate([idx[k[i]] for i in p]); a=auroc(y[sel],s[sel])
        if not np.isnan(a): o.append(a)
    o=np.array(o); return np.percentile(o,2.5), np.percentile(o,97.5)

# ---------- universe indikasi ----------
_, omop = read_copy(DC+'omop_relationship.sql')
ind=[r for r in omop if r['relationship_name']=='indication' and r['umls_cui']]
ind_set={(r['struct_id'],r['umls_cui']) for r in ind}
by_dis=collections.defaultdict(set)          # cui -> {struct_id}
for s,c in ind_set: by_dis[c].add(s)
ALL=sorted({s for s,_ in ind_set}); N_ALL=len(ALL)
dd=collections.Counter(s for s,_ in ind_set)
print(f"universe: {len(ind_set)} pasangan indikasi, {N_ALL} obat, {len(by_dis)} penyakit")

# ---------- keanggotaan kelompok ----------
_, pc = read_copy(DC+'pharma_class.sql')
epc=collections.defaultdict(set); moa=collections.defaultdict(set)
for r in pc:
    if r['type']=='EPC': epc[r['struct_id']].add(r['name'])
    elif r['type']=='MoA': moa[r['struct_id']].add(r['name'])
_, s2a = read_copy(DC+'struct2atc.sql')
atc4=collections.defaultdict(set); atc3=collections.defaultdict(set)
for r in s2a:
    a=r['atc_code']
    if a and len(a)>=5: atc4[r['struct_id']].add(a[:5]); atc3[r['struct_id']].add(a[:4])
_, act = read_copy(DC+'act_table_full.sql')
tgt=collections.defaultdict(set)
for r in act:
    if r.get('accession'): tgt[r['struct_id']].add(r['accession'])

def invert(d):
    inv=collections.defaultdict(set)
    for s,gs in d.items():
        for g in gs: inv[g].add(s)
    return inv
GROUPS={'epc':(epc,invert(epc)),'moa':(moa,invert(moa)),
        'atc4':(atc4,invert(atc4)),'atc3':(atc3,invert(atc3)),'tgt':(tgt,invert(tgt))}
print({k:f"{len(v[1])} kelompok" for k,v in GROUPS.items()})

# ---------- prior leave-one-drug-out ----------
def prior(kind, sid, cui):
    """Tingkat indikasi kelompok utk penyakit ini, obat sendiri DIKELUARKAN.
    Beberapa kelompok -> ambil maksimum (baseline versi paling kuat)."""
    memb, inv = GROUPS[kind]
    p_y = (len(by_dis.get(cui,())) - (1 if (sid,cui) in ind_set else 0)) / max(N_ALL-1,1)
    best = None
    for g in memb.get(sid,()):
        sibs = inv[g] - {sid}
        if not sibs: continue
        k = len(sibs & by_dis.get(cui,set()))
        best = max(best or 0.0, (k + ALPHA*p_y) / (len(sibs) + ALPHA))
    return (p_y if best is None else best), (best is not None)

def featurize(df):
    rows=[]
    for sid,cui in zip(df['struct_id'].astype(str), df['umls_cui']):
        f={}
        for k in GROUPS:
            f[k], f[k+'_def'] = prior(k, sid, cui)
        own = 1 if (sid,cui) in ind_set else 0
        f['dis_prior'] = (len(by_dis.get(cui,())) - own)/max(N_ALL-1,1)
        f['drug_deg']  = np.log1p(max(dd.get(sid,0)-own,0))
        rows.append(f)
    return pd.DataFrame(rows, index=df.index)

# ---------- set uji ----------
pil=pd.read_csv('data/processed/pilot_pairs_v5.tsv',sep='\t')
CLS=pil[pil.tier.isin(['pos','neg_random'])].reset_index(drop=True)
WIT=pd.read_csv('data/processed/within_class_pairs.tsv',sep='\t').reset_index(drop=True)
print(f"set tingkat-kelas: {len(CLS)} | set intra-kelas: {len(WIT)}")

OUT=[]
for nama, D in (('tingkat-kelas',CLS), ('intra-kelas',WIT)):
    X=featurize(D); y=D['label'].values
    print(f"\n{'='*74}\nTUGAS {nama.upper()}  (n={len(D)}, positif={int(y.sum())})\n{'='*74}")
    print(f"  cakupan prior terdefinisi: " +
          ", ".join(f"{k} {100*X[k+'_def'].mean():.0f}%" for k in GROUPS))
    single=[('epc','Prior kelas EPC (lookup)'),('moa','Prior kelas MoA'),
            ('atc4','Prior ATC level-4'),('atc3','Prior ATC level-3'),
            ('tgt','Prior target bersama'),('dis_prior','Prior penyakit saja'),
            ('drug_deg','Derajat obat saja')]
    for col,lab in single:
        a=auroc(y,X[col].values); lo,hi=boot_ci(D,X[col].values)
        print(f"  {lab:<28} AUROC {a:.3f}  [{lo:.3f}-{hi:.3f}]")
        OUT.append(dict(tugas=nama,baseline=lab,auroc=round(a,4),lo=round(lo,4),hi=round(hi,4)))
    # regresi logistik, CV dikelompokkan per OBAT (tak ada obat di train & test)
    feats=[k for k in GROUPS]+['dis_prior','drug_deg']
    M=X[feats].values; grp=D['drugbank_id'].values
    oof=np.zeros(len(D))
    for tr,te in GroupKFold(n_splits=5).split(M,y,grp):
        sc=StandardScaler().fit(M[tr])
        clf=LogisticRegression(max_iter=2000,C=1.0).fit(sc.transform(M[tr]),y[tr])
        oof[te]=clf.predict_proba(sc.transform(M[te]))[:,1]
    a=auroc(y,oof); lo,hi=boot_ci(D,oof)
    print(f"  {'Regresi logistik (semua)':<28} AUROC {a:.3f}  [{lo:.3f}-{hi:.3f}]   <- baseline terkuat")
    OUT.append(dict(tugas=nama,baseline='Regresi logistik (semua fitur)',auroc=round(a,4),lo=round(lo,4),hi=round(hi,4)))

pd.DataFrame(OUT).to_csv('results/class_prior_baseline.csv',index=False)
print("\nditulis: results/class_prior_baseline.csv")
