"""Baseline prior-kelas vs LLM, dengan dua perbaikan metodologis.

(1) PRIOR BEBAS KEBOCORAN. Semua prior dihitung HANYA dari obat yang tidak
    pernah dinilai di set uji mana pun (set referensi terpisah). Dengan begitu
    tidak ada obat uji yang menyumbang ke priornya sendiri, dan -- penting --
    pada tugas intra-kelas kedua saudara sekelas memakai himpunan referensi
    yang identik, sehingga fitur tingkat-kelas bernilai persis sama untuk
    keduanya. Itu jawaban yang benar secara konstruksi.

    Skema leave-one-drug-out yang lebih sederhana TIDAK boleh dipakai di sini:
    ia selalu mengeluarkan obat yang sedang dinilai, dan karena positif
    terindikasi untuk penyakit itu sementara negatif tidak, pengeluarannya
    asimetris -- label bocor lewat mekanisme koreksinya sendiri.

(2) AUROC PER TEMPLATE. Tiga template prompt (qa/tf/cloze) menghasilkan skor
    log-prob dengan skala berbeda; menggabungkannya ke satu peringkat mencampur
    skala yang tak sebanding. Di sini tiap template dinilai terpisah.
"""
import sys, os, collections, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from pgcopy import read_copy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

DC='data/interim/drugcentral/'; ALPHA=5.0
rng=np.random.default_rng(20261102); B=2000
TPL=['qa','tf','cloze']

def auroc(y,s):
    y=np.asarray(y); s=np.asarray(s,float); n1=y.sum(); n0=len(y)-n1
    if n1==0 or n0==0: return np.nan
    r=pd.Series(s).rank().values
    return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)

def boot_ci(keys,y,s):
    g=collections.defaultdict(list)
    for i,c in enumerate(keys): g[c].append(i)
    k=list(g); idx={a:np.array(b) for a,b in g.items()}
    y=np.asarray(y); s=np.asarray(s,float); o=[]
    for _ in range(B):
        p=rng.choice(len(k),len(k),replace=True)
        sel=np.concatenate([idx[k[i]] for i in p]); a=auroc(y[sel],s[sel])
        if not np.isnan(a): o.append(a)
    o=np.array(o); return np.percentile(o,2.5), np.percentile(o,97.5)

# ---------- universe indikasi ----------
_,omop=read_copy(DC+'omop_relationship.sql')
ind_set={(r['struct_id'],r['umls_cui']) for r in omop
         if r['relationship_name']=='indication' and r['umls_cui']}
by_dis=collections.defaultdict(set)
for s,c in ind_set: by_dis[c].add(s)

# ---------- set uji ----------
pil=pd.read_csv('data/processed/pilot_pairs_v5.tsv',sep='\t')
CLS=pil[pil.tier.isin(['pos','neg_random'])].reset_index(drop=True)
WIT=pd.read_csv('data/processed/within_class_pairs.tsv',sep='\t').reset_index(drop=True)
TEST={str(s) for s in CLS.struct_id} | {str(s) for s in WIT.struct_id}
REF=sorted({s for s,_ in ind_set} - TEST)
N_REF=len(REF); REFSET=set(REF)
print(f"obat universe={len({s for s,_ in ind_set})}  obat uji={len(TEST)}  obat REFERENSI={N_REF}")
ref_dis={c:(ss & REFSET) for c,ss in by_dis.items()}

# ---------- kelompok, dibatasi ke REF ----------
_,pc=read_copy(DC+'pharma_class.sql')
epc=collections.defaultdict(set); moa=collections.defaultdict(set)
for r in pc:
    if r['type']=='EPC': epc[r['struct_id']].add(r['name'])
    elif r['type']=='MoA': moa[r['struct_id']].add(r['name'])
_,s2a=read_copy(DC+'struct2atc.sql')
atc4=collections.defaultdict(set); atc3=collections.defaultdict(set)
for r in s2a:
    a=r['atc_code']
    if a and len(a)>=5: atc4[r['struct_id']].add(a[:5]); atc3[r['struct_id']].add(a[:4])
_,act=read_copy(DC+'act_table_full.sql')
tgt=collections.defaultdict(set)
for r in act:
    if r.get('accession'): tgt[r['struct_id']].add(r['accession'])

def inv_ref(d):
    inv=collections.defaultdict(set)
    for s,gs in d.items():
        if s in REFSET:
            for g in gs: inv[g].add(s)
    return inv
G={'epc':(epc,inv_ref(epc)),'moa':(moa,inv_ref(moa)),'atc4':(atc4,inv_ref(atc4)),
   'atc3':(atc3,inv_ref(atc3)),'tgt':(tgt,inv_ref(tgt))}

def prior(kind,sid,cui):
    memb,inv=G[kind]
    p_y=len(ref_dis.get(cui,()))/max(N_REF,1)
    best=None
    for g in memb.get(sid,()):
        sibs=inv.get(g)
        if not sibs: continue
        k=len(sibs & ref_dis.get(cui,set()))
        best=max(best or 0.0,(k+ALPHA*p_y)/(len(sibs)+ALPHA))
    return (p_y if best is None else best),(best is not None)

def featurize(df):
    out=[]
    for sid,cui in zip(df.struct_id.astype(str),df.umls_cui):
        f={}
        for k in G: f[k],f[k+'_def']=prior(k,sid,cui)
        f['dis_prior']=len(ref_dis.get(cui,()))/max(N_REF,1)
        f['drug_deg']=np.log1p(len([1 for c in by_dis if sid in by_dis[c]]))
        out.append(f)
    return pd.DataFrame(out,index=df.index)

# ---------- LLM per template ----------
E={'1B':'results/easy_OLMo-2-0425-1B-Instruct.jsonl','7B':'results/scores_5cond.jsonl',
   '13B':'results/easy_OLMo-2-1124-13B-Instruct.jsonl','32B':'results/easy_OLMo-2-0325-32B-Instruct.jsonl'}
H={'1B':'results/within_OLMo-2-0425-1B-Instruct.jsonl','7B':'results/scores_within.jsonl',
   '13B':'results/within_OLMo-2-1124-13B-Instruct.jsonl','32B':'results/within_OLMo-2-0325-32B-Instruct.jsonl'}
def llm(f,within,cond='A0'):
    d=pd.DataFrame([json.loads(l) for l in open(f)])
    g=d[d.condition==cond] if within else pd.concat(
        [d[d.tier=='pos'],d[d.tier=='neg_random']]).query(f"condition=='{cond}'")
    return g.drop_duplicates(['drugbank_id','umls_cui','template'])

ROWS=[]
for nama,D,FILES,within in (('tingkat-kelas',CLS,E,False),('intra-kelas',WIT,H,True)):
    X=featurize(D); y=D.label.values; keys=D.drugbank_id.values
    print(f"\n{'='*78}\nTUGAS {nama.upper()}  n={len(D)}  positif={int(y.sum())}\n{'='*78}")
    print("  cakupan prior: "+", ".join(f"{k} {100*X[k+'_def'].mean():.0f}%" for k in G))
    print(f"\n  -- BASELINE (lookup, tanpa LLM) --")
    for col,lab in [('epc','Prior kelas EPC'),('atc3','Prior ATC level-3'),
                    ('atc4','Prior ATC level-4'),('tgt','Prior target bersama'),
                    ('moa','Prior kelas MoA'),('dis_prior','Prior penyakit saja')]:
        a=auroc(y,X[col]); lo,hi=boot_ci(keys,y,X[col])
        flag=" (konstan dlm pasangan)" if nama=='intra-kelas' and abs(a-0.5)<0.02 else ""
        print(f"    {lab:<24} {a:.3f}  [{lo:.3f}-{hi:.3f}]{flag}")
        ROWS.append(dict(tugas=nama,sistem=lab,auroc=round(a,4),lo=round(lo,4),hi=round(hi,4)))
    feats=list(G)+['dis_prior','drug_deg']
    M=X[feats].values; oof=np.zeros(len(D))
    for tr,te in GroupKFold(n_splits=5).split(M,y,keys):
        sc=StandardScaler().fit(M[tr])
        oof[te]=LogisticRegression(max_iter=2000).fit(sc.transform(M[tr]),y[tr]
                 ).predict_proba(sc.transform(M[te]))[:,1]
    a=auroc(y,oof); lo,hi=boot_ci(keys,y,oof)
    print(f"    {'Regresi logistik (semua)':<24} {a:.3f}  [{lo:.3f}-{hi:.3f}]  <- baseline terkuat")
    ROWS.append(dict(tugas=nama,sistem='Regresi logistik (semua fitur)',auroc=round(a,4),lo=round(lo,4),hi=round(hi,4)))
    print(f"\n  -- LLM (per template) --")
    for m in ['1B','7B','13B','32B']:
        g=llm(FILES[m],within); per=[]
        for t in TPL:
            gg=g[g.template==t]; per.append(auroc(gg.label,gg.score))
        best=TPL[int(np.nanargmax(per))]; gb=g[g.template==best]
        lo,hi=boot_ci(gb.drugbank_id.values,gb.label.values,gb.score.values)
        print(f"    OLMo-2 {m:<4} qa={per[0]:.3f} tf={per[1]:.3f} cloze={per[2]:.3f} | "
              f"rerata={np.mean(per):.3f}  terbaik={max(per):.3f} [{lo:.3f}-{hi:.3f}] ({best})")
        ROWS.append(dict(tugas=nama,sistem=f'OLMo-2 {m} (template terbaik)',
                         auroc=round(max(per),4),lo=round(lo,4),hi=round(hi,4),
                         rerata_template=round(float(np.mean(per)),4)))
pd.DataFrame(ROWS).to_csv('results/baseline_vs_llm.csv',index=False)
print("\nditulis: results/baseline_vs_llm.csv")
