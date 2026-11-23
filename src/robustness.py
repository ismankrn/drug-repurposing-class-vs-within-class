"""Analisis ketahanan B1-B3: stratifikasi ATC, kalibrasi, stratifikasi derajat.

Terpisah dari src/rebuild_outputs.py dengan sengaja. Skrip itu memakai satu
`rng` global, dan menambahkan bootstrap di tengahnya menggeser setiap selang
kepercayaan di seluruh tabel yang sudah diperiksa. Skrip ini punya rng sendiri.

Metrik AUROC identik dengan rebuild_outputs.py: makro dalam-kelompok, dihitung
per template lalu dirata-ratakan; bootstrap pada tingkat KELOMPOK.

  python3 -I src/robustness.py
"""
import sys, os, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from pgcopy import read_copy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

B=2000; TPL=['qa','tf','cloze']
rng=np.random.default_rng(20261109)
DC='data/interim/drugcentral/'

# ---------------------------------------------------------------- data
WIT=pd.read_csv('data/processed/within_class_pairs.tsv',sep='\t')
GKEY={(r.drugbank_id,r.umls_cui):f"{r.umls_cui}|{r.shared_class}" for r in WIT.itertuples()}

E={'1B':'results/easy_OLMo-2-0425-1B-Instruct.jsonl','7B':'results/scores_5cond.jsonl',
   '13B':'results/easy_OLMo-2-1124-13B-Instruct.jsonl','32B':'results/easy_OLMo-2-0325-32B-Instruct.jsonl'}
H={'1B':'results/within_OLMo-2-0425-1B-Instruct.jsonl','7B':'results/scores_within.jsonl',
   '13B':'results/within_OLMo-2-1124-13B-Instruct.jsonl','32B':'results/within_OLMo-2-0325-32B-Instruct.jsonl'}
EQ={q:f'results/cls_Qwen3-{q}.jsonl' for q in ('1.7B','8B','14B','32B')}
HQ={q:f'results/wit_Qwen3-{q}.jsonl' for q in ('1.7B','8B','14B','32B')}
MODELS=[(f"OLMo-2 {m}",E[m],H[m]) for m in ('1B','7B','13B','32B')] + \
       [(f"Qwen3 {q}",EQ[q],HQ[q]) for q in ('1.7B','8B','14B','32B')]

def auroc(y,s):
    y=np.asarray(y); s=np.asarray(s,float); n1=y.sum(); n0=len(y)-n1
    if n1==0 or n0==0: return np.nan
    r=pd.Series(s).rank().values
    return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)

def load(f,within,cond='A0'):
    d=pd.DataFrame([json.loads(l) for l in open(f)])
    g=d[d.condition==cond] if within else pd.concat(
        [d[d.tier=='pos'],d[d.tier=='neg_random']]).query(f"condition=='{cond}'")
    g=g.drop_duplicates(['drugbank_id','umls_cui','template'])
    if within:
        ok=[(a,b) in GKEY for a,b in zip(g.drugbank_id,g.umls_cui)]
        g=g[pd.Series(ok,index=g.index).values]
    return g

def gkey(g,within):
    return (np.array([GKEY[(a,b)] for a,b in zip(g.drugbank_id,g.umls_cui)])
            if within else g.drugbank_id.values)

def mat(g,within):
    """DataFrame AUROC: baris=kelompok, kolom=template."""
    cols={}
    for t in TPL:
        gg=g[g.template==t]
        df=pd.DataFrame({'y':gg.label.values,'k':gkey(gg,within),'s':gg.score.values})
        ks=[];vs=[]
        for k,q in df.groupby('k',sort=True):
            if q.y.nunique()==2: ks.append(k); vs.append(auroc(q.y.values,q.s.values))
        cols[t]=pd.Series(vs,index=ks)
    return pd.DataFrame(cols).dropna()

def boot(M):
    """Titik estimasi + CI bootstrap pada kelompok."""
    v=M.values; n=len(v)
    if n<2: return (np.nan,np.nan,np.nan,n)
    d=np.array([v[rng.integers(0,n,n)].mean(axis=0).mean() for _ in range(B)])
    return v.mean(axis=0).mean(), np.percentile(d,2.5), np.percentile(d,97.5), n

# ---------------------------------------------------------------- strata
_,s2a=read_copy(DC+'struct2atc.sql')
ATC1=collections.defaultdict(set)
for r in s2a:
    a=r['atc_code']
    if a: ATC1[str(r['struct_id'])].add(a[0])
# Nama resmi kelompok anatomi utama ATC (WHOCC).
ATCNAME={'A':'Alimentary tract and metabolism','B':'Blood and blood forming organs',
 'C':'Cardiovascular system','D':'Dermatologicals',
 'G':'Genito-urinary system and sex hormones','H':'Systemic hormonal preparations',
 'J':'Anti-infectives for systemic use','L':'Antineoplastic and immunomodulating agents',
 'M':'Musculo-skeletal system','N':'Nervous system','P':'Antiparasitic products',
 'R':'Respiratory system','S':'Sensory organs','V':'Various'}

PIL=pd.read_csv('data/processed/pilot_pairs_v5.tsv',sep='\t')
_cls=PIL[PIL.tier.isin(['pos','neg_random'])].drop_duplicates(['drugbank_id','umls_cui'])
# Kunci kelompok tingkat-kelas adalah drugbank_id; satu obat satu struct_id.
SID={r.drugbank_id:str(r.struct_id) for r in _cls.itertuples()}
# Kunci kelompok intra-kelas adalah (penyakit|kelas); ATC-nya diambil dari obat
# yang DISETUJUI dalam kelompok itu, karena kelompok memuat beberapa obat.
SIDW={}
for r in WIT[WIT.label==1].itertuples():
    SIDW.setdefault(f"{r.umls_cui}|{r.shared_class}", str(r.struct_id))

def atc_of(key,within):
    sid=SIDW.get(key) if within else SID.get(key)
    return ATC1.get(sid,set())

# Derajat: nilai sama untuk tiap kelompok, diambil dari baris positifnya.
DEG={r.drugbank_id:(r.drug_degree,r.disease_degree) for r in _cls.itertuples()}
DEGW={}
for r in WIT[WIT.label==1].itertuples():
    DEGW.setdefault(f"{r.umls_cui}|{r.shared_class}", (r.drug_degree,r.disease_degree))
def deg_of(key,within,which):
    d=(DEGW if within else DEG).get(key)
    return np.nan if d is None else d[0 if which=='drug' else 1]

def c3(t):
    return "--" if np.isnan(t[0]) else f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"
def w(fn,txt): open(f'paper/tables/{fn}','w').write(txt+"\n")

# =================== B1: stratifikasi wilayah terapi (ATC level-1) ==========
# Satu obat dapat memiliki beberapa kode ATC, jadi obat itu masuk ke SETIAP
# wilayah yang dimilikinya. Strata karena itu tumpang tindih dan jumlahnya
# tidak menjumlah ke total; yang diuji adalah "apakah jurangnya positif di
# dalam tiap wilayah", bukan dekomposisi.
# Ambang 30 kelompok di KEDUA tugas. Matriks terskor hanya memuat kelompok
# yang berisi dua kelas, jadi 363 kelompok tingkat-kelas tersebar atas 14
# wilayah; ambang lebih tinggi menyisakan dua wilayah saja, ambang lebih
# rendah menghasilkan selang yang terlalu lebar untuk dibaca.
MINGRP=30
REF=[("OLMo-2 32B",E['32B'],H['32B']),("Qwen3 32B",EQ['32B'],HQ['32B'])]

def gap_boot(Mc,Mw):
    """Selisih makro-AUROC dua tugas. Kelompoknya berbeda jenis (obat lawan
    (penyakit,kelas)), jadi bootstrap-nya independen, bukan berpasangan."""
    a,b=Mc.values,Mw.values; na,nb=len(a),len(b)
    if na<2 or nb<2: return (np.nan,np.nan,np.nan,0)
    d=np.array([a[rng.integers(0,na,na)].mean(axis=0).mean()
               -b[rng.integers(0,nb,nb)].mean(axis=0).mean() for _ in range(B)])
    return a.mean(axis=0).mean()-b.mean(axis=0).mean(), np.percentile(d,2.5), np.percentile(d,97.5), na

print("memuat matriks referensi...")
MM={}
for nm,fc,fw in REF:
    MM[(nm,False)]=mat(load(fc,False),False)
    MM[(nm,True)] =mat(load(fw,True ),True )
    print(f"  {nm}: {len(MM[(nm,False)])} kelompok kelas, {len(MM[(nm,True)])} kelompok intra")

AREAS=[]
_m0=MM[(REF[0][0],False)]; _m1=MM[(REF[0][0],True)]
for L in sorted(ATCNAME):
    kc=[k for k in _m0.index if L in atc_of(k,False)]
    kw=[k for k in _m1.index if L in atc_of(k,True)]
    if len(kc)>=MINGRP and len(kw)>=MINGRP: AREAS.append((L,len(kc),len(kw)))
print(f"\nB1: {len(AREAS)} wilayah dengan >={MINGRP} kelompok di KEDUA tugas")

B1={}
for nm,_,_ in REF:
    Mc,Mw=MM[(nm,False)],MM[(nm,True)]
    for L,_,_ in AREAS:
        kc=[k for k in Mc.index if L in atc_of(k,False)]
        kw=[k for k in Mw.index if L in atc_of(k,True)]
        B1[(nm,L)]=(boot(Mc.loc[kc]),boot(Mw.loc[kw]),gap_boot(Mc.loc[kc],Mw.loc[kw]))
    # Baris gabungan: SELURUH kelompok, pembanding untuk tiap wilayah.
    B1[(nm,'*')]=(boot(Mc),boot(Mw),gap_boot(Mc,Mw))

print(f"\n{'wilayah':46s} {'n_kls':>6} {'n_int':>6} "+" ".join(f"{nm+' jurang':>26}" for nm,_,_ in REF))
for L,nc,nw in AREAS:
    print(f"{L} {ATCNAME[L][:43]:44s} {nc:6d} {nw:6d} "
          +" ".join(f"{c3(B1[(nm,L)][2]):>26}" for nm,_,_ in REF))
print("\nrincian OLMo-2 32B per wilayah (kelas | intra):")
for L,_,_ in AREAS:
    a,b,_=B1[("OLMo-2 32B",L)]
    print(f"  {L}: {c3(a)}  |  {c3(b)}")

# =================== B3: stratifikasi derajat =============================
# Pasangan negatif sudah DICOCOKKAN derajatnya (Section 3.1), jadi derajat
# tidak bisa menjadi jalan pintas DI DALAM satu kelompok. Yang diuji di sini
# berbeda: apakah keunggulan model terpusat pada obat yang banyak dipelajari.
# Tersil dihitung di dalam tiap tugas atas kelompok yang benar-benar terskor.
def tersil(M,within,which='drug'):
    d=pd.Series({k:deg_of(k,within,which) for k in M.index}).dropna()
    q=np.percentile(d,[100/3,200/3])
    lab=pd.Series(np.where(d<=q[0],'rendah',np.where(d<=q[1],'sedang','tinggi')),index=d.index)
    return lab,q

B3={}; TQ={}
for nm,_,_ in REF:
    Mc,Mw=MM[(nm,False)],MM[(nm,True)]
    lc,qc=tersil(Mc,False); lw,qw=tersil(Mw,True); TQ[nm]=(qc,qw)
    for s_ in ('rendah','sedang','tinggi'):
        kc=lc[lc==s_].index; kw=lw[lw==s_].index
        B3[(nm,s_)]=(boot(Mc.loc[kc]),boot(Mw.loc[kw]),gap_boot(Mc.loc[kc],Mw.loc[kw]))

print(f"\n=== B3: tersil derajat obat ===")
print(f"  batas tersil OLMo (kelas): {TQ[REF[0][0]][0]}  (intra): {TQ[REF[0][0]][1]}")
print(f"{'model':14s} {'tersil':8s} {'n':>5} {'kelas':>24} {'intra':>24} {'jurang':>24}")
for nm,_,_ in REF:
    for s_ in ('rendah','sedang','tinggi'):
        a,b,g=B3[(nm,s_)]
        print(f"{nm:14s} {s_:8s} {a[3]:5d} {c3(a):>24} {c3(b):>24} {c3(g):>24}")

# =================== B2: kalibrasi ========================================
# AUROC bersifat berbasis peringkat dan bebas ambang: ia tidak mengatakan
# apa pun tentang apakah skor model dapat dibaca sebagai peluang. Dua hal
# diukur di sini.
#   (1) Peluang IMPLISIT model sendiri, sigma(skor), yaitu yang akan dipakai
#       pembaca kalau ia memperlakukan selisih log-peluang sebagai logit.
#   (2) Peluang setelah penskalaan Platt yang dicocokkan DI LUAR LIPATAN
#       (GroupKFold atas obat), yaitu batas atas dari apa yang dapat dicapai
#       pemetaan monoton global.
# Selisih keduanya memisahkan "skornya salah kalibrasi" dari "skornya memang
# tidak membawa informasi peluang".
def sig(x): return 1.0/(1.0+np.exp(-np.clip(x,-50,50)))

def ece(y,p,bins=10):
    """Expected calibration error, 10 nampan lebar-sama, berbobot isi."""
    e=0.0; n=len(y)
    for i in range(bins):
        lo,hi=i/bins,(i+1)/bins
        m=(p>lo)&(p<=hi) if i else (p>=lo)&(p<=hi)
        if m.sum(): e+=m.sum()/n*abs(y[m].mean()-p[m].mean())
    return e

def kalibrasi(f,within):
    g=load(f,within)
    piv=g.pivot_table(index=['drugbank_id','umls_cui'],columns='template',
                      values='score',aggfunc='mean')
    lab=g.drop_duplicates(['drugbank_id','umls_cui']).set_index(['drugbank_id','umls_cui']).label
    piv=piv.dropna(); y=lab.loc[piv.index].values.astype(float)
    s=piv.mean(axis=1).values                       # rerata tiga template
    grp=np.array([i[0] for i in piv.index])         # obat, supaya tidak bocor
    p_naif=sig(s)
    p_cal=np.zeros(len(s))
    for tr,te in GroupKFold(5).split(s.reshape(-1,1),y,grp):
        lr=LogisticRegression(C=1e6,max_iter=1000).fit(s[tr].reshape(-1,1),y[tr])
        p_cal[te]=lr.predict_proba(s[te].reshape(-1,1))[:,1]
    br=lambda p:((p-y)**2).mean()
    b0=br(np.full(len(y),y.mean()))                 # hanya prevalensi
    # CI untuk BSS: bootstrap atas OBAT, yaitu satuan yang sama dengan
    # pengelompokan GroupKFold. Mengambil ulang baris satu per satu akan
    # memecah obat antar-lipatan dan meremehkan ketidakpastiannya.
    uo=np.unique(grp); idx={o:np.where(grp==o)[0] for o in uo}
    dr=[]
    for _ in range(B):
        pick=np.concatenate([idx[o] for o in rng.choice(uo,len(uo),replace=True)])
        yy,pp=y[pick],p_cal[pick]
        bb0=((yy.mean()-yy)**2).mean()
        dr.append(np.nan if bb0==0 else 1-((pp-yy)**2).mean()/bb0)
    dr=np.array(dr,float)
    return dict(n=len(y),prev=y.mean(),
                bss_lo=np.nanpercentile(dr,2.5),bss_hi=np.nanpercentile(dr,97.5),
                ece_naif=ece(y,p_naif),ece_cal=ece(y,p_cal),
                brier_naif=br(p_naif),brier_cal=br(p_cal),b0=b0,
                bss_naif=1-br(p_naif)/b0,bss_cal=1-br(p_cal)/b0)

print("\n=== B2: kalibrasi ===")
B2={}
for nm,fc,fw in MODELS:
    for tnm,f_,wi in (('kelas',fc,False),('intra',fw,True)):
        B2[(nm,tnm)]=kalibrasi(f_,wi)
print(f"{'model':14s} {'tugas':6s} {'n':>5} {'prev':>5} "
      f"{'ECE naif':>9} {'ECE cal':>8} {'Brier naif':>11} {'Brier cal':>10} "
      f"{'BSS naif':>9} {'BSS cal':>8}")
for nm,_,_ in MODELS:
    for tnm in ('kelas','intra'):
        r=B2[(nm,tnm)]
        print(f"{nm:14s} {tnm:6s} {r['n']:5d} {r['prev']:5.2f} "
              f"{r['ece_naif']:9.3f} {r['ece_cal']:8.3f} {r['brier_naif']:11.3f} "
              f"{r['brier_cal']:10.3f} {r['bss_naif']:9.3f} {r['bss_cal']:8.3f}")

# =================== tabel LaTeX ==========================================
# Nama wilayah dipendekkan dari nama resmi WHOCC: nama penuh "Antineoplastic
# and immunomodulating agents" membuat tabel meluber bersama dua kolom selang.
# Huruf ATC-nya disertakan supaya tetap dapat ditelusuri.
SHORT={'A':'Alimentary and metabolism','C':'Cardiovascular','D':'Dermatologicals',
 'J':'Anti-infectives','L':'Antineoplastic','N':'Nervous system',
 'R':'Respiratory','S':'Sensory organs'}

t=["\\begin{table}[!tbp]","\\centering","\\small","\\setlength{\\tabcolsep}{4pt}",
 "\\caption{Gap between the two tasks in the two largest checkpoints, "
 "stratified by therapeutic area.}",
 "\\label{tab:atc}","\\begin{tabular}{@{}lccc@{}}","\\toprule",
 "ATC area & Groups & OLMo-2 32B & Qwen3 32B \\\\","\\midrule"]
for L,nc,nw in AREAS:
    t.append(f"{SHORT[L]} ({L}) & {nc}/{nw} & "
             f"{c3(B1[('OLMo-2 32B',L)][2])} & {c3(B1[('Qwen3 32B',L)][2])} \\\\")
t.append("\\midrule")
t.append(f"All groups & {len(MM[('OLMo-2 32B',False)])}/{len(MM[('OLMo-2 32B',True)])} & "
         f"{c3(B1[('OLMo-2 32B','*')][2])} & {c3(B1[('Qwen3 32B','*')][2])} \\\\")
t+=["\\bottomrule","\\end{tabular}","\\end{table}"]
w('tab-atc.tex',"\n".join(t))

# Label baris dibiarkan polos. Ambang tersilnya (5 dan 10 di tingkat-kelas, 4 dan
# 10,3 di intra-kelas) dinyatakan di paragraf stratifikasi Section 2.6, dan arti
# "137/143" di kolom Groups tab-atc di kalimat kedua paragraf Section 3.7 --
# keduanya atas permintaan penulis, yang menghendaki keterangan di prosa dan
# bukan di kepala kolom. Ambangnya tetap dicetak ke log oleh blok B3 di atas.
DEGEN={'rendah':'Low','sedang':'Middle','tinggi':'High'}
t=["\\begin{table}[!tbp]","\\centering","\\small","\\setlength{\\tabcolsep}{4pt}",
 "\\caption{Macro within-group \\auroc{} in the two largest checkpoints, stratified by "
 "tertile of the drug degree.}",
 "\\label{tab:degstrat}","\\begin{tabular}{@{}lccc@{}}","\\toprule",
 "Tertile of drug degree & Class-level & Within-class & Gap \\\\","\\midrule"]
for i,(nm,_,_) in enumerate(REF):
    if i: t.append("\\addlinespace")
    t.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\textit{{{nm}}}}} \\\\")
    for s_ in ('rendah','sedang','tinggi'):
        a,b_,_g=B3[(nm,s_)]
        t.append(f"\\quad {DEGEN[s_]} & {c3(a)} & {c3(b_)} & ${a[0]-b_[0]:.3f}$ \\\\")
t+=["\\bottomrule","\\end{tabular}","\\end{table}"]
w('tab-degstrat.tex',"\n".join(t))

def _bss(r):
    body=f"{r['bss_cal']:+.3f}\;[{r['bss_lo']:+.3f},\;{r['bss_hi']:+.3f}]"
    return ("$\\mathbf{"+body+"}$") if (r['bss_lo']>0 or r['bss_hi']<0) else "$"+body+"$"
t=["\\begin{table}[!tbp]","\\centering","\\small","\\setlength{\\tabcolsep}{4pt}",
 "\\caption{Calibration of the scores.}",
 "\\label{tab:calib}","\\begin{tabular}{@{}lcccc@{}}","\\toprule",
 "& \\multicolumn{2}{c}{Class-level task} & \\multicolumn{2}{c}{Within-class task} \\\\",
 "\\cmidrule(lr){2-3}\\cmidrule(l){4-5}",
 # Arah "bagus" kedua kolom BERLAWANAN (ECE galat, Brier skill kecakapan).
 # Keterangannya ada di kalimat kedua paragraf Section 3.9, bukan di kepala
 # kolom: panah pernah dipasang di sini lalu dilepas atas permintaan penulis.
 "Model & \\textsc{ece} & Brier skill & \\textsc{ece} & Brier skill \\\\","\\midrule"]
for i,(nm,_,_) in enumerate(MODELS):
    if i==4: t.append("\\addlinespace")
    rc,rw=B2[(nm,'kelas')],B2[(nm,'intra')]
    t.append(f"{nm} & {rc['ece_naif']:.3f} & {_bss(rc)} & "
             f"{rw['ece_naif']:.3f} & {_bss(rw)} \\\\")
t+=["\\bottomrule","\\end{tabular}","\\end{table}"]
w('tab-calib.tex',"\n".join(t))
print("\n3 tabel ditulis: tab-atc, tab-degstrat, tab-calib")

# =================== diagnostik untuk Limitations =========================
# Naskah menyatakan bahwa derajat pada graf LATIH berada di bawah kebetulan
# sebagai prediktor, karena membuang positif uji menurunkan derajat tiap obat
# positif sebesar satu sementara negatifnya tidak kehilangan apa pun. Angka itu
# sebelumnya ditulis tangan (0,477) dan tidak dapat ditelusuri ke kode mana
# pun; di sini ia dihitung dari graf yang sama dengan src/kg_baseline.py.
_,_omop=read_copy(DC+'omop_relationship.sql')
_edges={(r['struct_id'],r['umls_cui']) for r in _omop
        if r['relationship_name']=='indication' and r['umls_cui']}
_CLS=PIL[PIL.tier.isin(['pos','neg_random'])].drop_duplicates(['drugbank_id','umls_cui'])
_W=WIT.drop_duplicates(['drugbank_id','umls_cui'])
_drop ={(str(r.struct_id),r.umls_cui) for r in _CLS.itertuples() if r.label==1}
_drop|={(str(r.struct_id),r.umls_cui) for r in _W.itertuples()   if r.label==1}
_deg={}
for _s,_c in (_edges-_drop): _deg[_s]=_deg.get(_s,0)+1
print("\n=== diagnostik: derajat graf-latih sebagai prediktor ===")
for _nm,_D,_kf in (('intra-kelas',_W,lambda r:f"{r.umls_cui}|{r.shared_class}"),
                   ('tingkat-kelas',_CLS,lambda r:r.drugbank_id)):
    _df=pd.DataFrame([(_kf(r),r.label,_deg.get(str(r.struct_id),0)) for r in _D.itertuples()],
                     columns=['k','y','s'])
    _v=[auroc(q.y.values,q.s.values) for _,q in _df.groupby('k') if q.y.nunique()==2]
    _v=[x for x in _v if not np.isnan(x)]
    print(f"  {_nm}: makro AUROC {np.mean(_v):.3f}  (n={len(_v)} kelompok)")
