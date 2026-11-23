"""Bangun ulang figur + tabel dengan metrik terkoreksi (lihat Temuan AI).

Metrik: AUROC MAKRO DALAM-KELOMPOK, dihitung per template lalu dirata-ratakan.
  - tugas tingkat-kelas : kelompok = obat
  - tugas intra-kelas   : kelompok = (penyakit, kelas EPC)
CI: bootstrap pada tingkat KELOMPOK (B=2.000); untuk rerata-template, tiap
resample menghitung ulang ketiga template lalu dirata-ratakan.

Angka utama memakai RERATA tiga template, bukan template terbaik: memilih
template terbaik di set uji adalah seleksi pada data uji (bias optimistis).
"""
import sys, os, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pgcopy import read_copy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

S1,S2,S3="#2a78d6","#eb6834","#1baf7a"
SURF,GRID,AXIS="#fcfcfb","#e1e0d9","#c3c2b7"
INK,SEC,MUT="#0b0b0b","#52514e","#898781"
B=2000; TPL=['qa','tf','cloze']; ALPHA=5.0
rng=np.random.default_rng(20261102)
def k3(v): return f"{v:.3f}".replace('.',',')     # tabel (Indonesia)
def ribu(n): return f"{n:,}".replace(',','.')      # pemisah ribuan Indonesia
def e3(v): return f"{v:.3f}"                       # figur (Inggris)
dot=FuncFormatter(lambda v,_: f"{v:.2f}")          # tick figur
def ci3(lo,hi): return f"[{k3(lo)}–{k3(hi)}]"
mpl.rcParams.update({
 "figure.facecolor":SURF,"axes.facecolor":SURF,"savefig.facecolor":SURF,
 "axes.edgecolor":AXIS,"axes.labelcolor":SEC,"text.color":INK,
 "xtick.color":MUT,"ytick.color":MUT,"grid.color":GRID,"grid.linewidth":0.8,
 "axes.linewidth":0.8,"font.size":10,"axes.titlesize":12,
 "axes.spines.top":False,"axes.spines.right":False})

def auroc(y,s):
    y=np.asarray(y); s=np.asarray(s,float); n1=y.sum(); n0=len(y)-n1
    if n1==0 or n0==0: return np.nan
    r=pd.Series(s).rank().values
    return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)

def per_group(y,g,s):
    """AUROC tiap kelompok -> (kunci, nilai). Kelompok tanpa dua kelas dibuang."""
    df=pd.DataFrame({'y':y,'g':g,'s':s}); ks=[]; vs=[]
    for k,gg in df.groupby('g',sort=True):
        if gg.y.nunique()==2: ks.append(k); vs.append(auroc(gg.y.values,gg.s.values))
    return np.array(ks,dtype=object), np.array(vs,dtype=float)

def summarize(mat):
    """mat: (n_kelompok, n_template) -> rerata, CI rerata, nilai per template."""
    per_t=mat.mean(axis=0); point=per_t.mean()
    n=mat.shape[0]; draws=[]
    for _ in range(B):
        ix=rng.integers(0,n,n); draws.append(mat[ix].mean(axis=0).mean())
    d=np.array(draws)
    return point, np.percentile(d,2.5), np.percentile(d,97.5), per_t, n

# ================= data LLM =================
E={'1B':'results/easy_OLMo-2-0425-1B-Instruct.jsonl','7B':'results/scores_5cond.jsonl',
   '13B':'results/easy_OLMo-2-1124-13B-Instruct.jsonl','32B':'results/easy_OLMo-2-0325-32B-Instruct.jsonl'}
H={'1B':'results/within_OLMo-2-0425-1B-Instruct.jsonl','7B':'results/scores_within.jsonl',
   '13B':'results/within_OLMo-2-1124-13B-Instruct.jsonl','32B':'results/within_OLMo-2-0325-32B-Instruct.jsonl'}
MS=['1B','7B','13B','32B']
# jumlah parameter sebenarnya dari metadata safetensors di Hugging Face
PAR={'1B':1.48,'7B':7.30,'13B':13.72,'32B':32.23}
# Label tangga untuk PENYAJIAN. Kunci internal (A0/A4/A5c/A5t/A5) tetap dipakai
# di data, berkas skor dan runner; hanya tampilannya yang berurutan, karena
# lubang A1-A3 adalah sisa desain kontaminasi yang tidak pernah dijalankan.
PN={'A0':'P1','A4':'P2','A5c':'P3','A5t':'P4','A5':'P5'}
NAMA={'1B':'OLMo-2 1B-Instruct','7B':'OLMo-2 7B-Instruct','13B':'OLMo-2 13B-Instruct','32B':'OLMo-2 32B-Instruct'}
WIT=pd.read_csv('data/processed/within_class_pairs.tsv',sep='\t')
GKEY={(r.drugbank_id,r.umls_cui):f"{r.umls_cui}|{r.shared_class}" for r in WIT.itertuples()}

def llm(f,within,cond='A0'):
    d=pd.DataFrame([json.loads(l) for l in open(f)])
    g=d[d.condition==cond] if within else pd.concat(
        [d[d.tier=='pos'],d[d.tier=='neg_random']]).query(f"condition=='{cond}'")
    return g.drop_duplicates(['drugbank_id','umls_cui','template'])

def matrix(g,within,sub=None):
    """Matriks AUROC per (kelompok, template), kelompok disejajarkan antar template."""
    cols={}
    for t in TPL:
        gg=g[g.template==t]
        if sub is not None: gg=gg[gg.drugbank_id.isin(sub)]
        key=[GKEY[(d,c)] for d,c in zip(gg.drugbank_id,gg.umls_cui)] if within else gg.drugbank_id.values
        ks,vs=per_group(gg.label.values,key,gg.score.values)
        cols[t]=pd.Series(vs,index=ks)
    M=pd.DataFrame(cols).dropna()
    return M.values, M.index

def eval_llm(m,within,cond='A0',sub=None):
    return summarize(matrix(llm((H if within else E)[m],within,cond),within,sub)[0])

# ================= baseline =================
DC='data/interim/drugcentral/'
_,omop=read_copy(DC+'omop_relationship.sql')
ind_set={(r['struct_id'],r['umls_cui']) for r in omop
         if r['relationship_name']=='indication' and r['umls_cui']}
by_dis=collections.defaultdict(set)
for s,c in ind_set: by_dis[c].add(s)
pil=pd.read_csv('data/processed/pilot_pairs_v5.tsv',sep='\t')
CLS=pil[pil.tier.isin(['pos','neg_random'])].reset_index(drop=True)
WITr=WIT.reset_index(drop=True)

# Dedup pasangan ganda, identik dengan src/gpu_phase2.py dan src/kg_baseline.py.
# Tanpa ini baseline dinilai pada 1.034 kelompok sementara LLM pada 1.017.
_n0=(len(CLS),len(WITr))
CLS=CLS.drop_duplicates(['drugbank_id','umls_cui']).reset_index(drop=True)
WITr=WITr.drop_duplicates(['drugbank_id','umls_cui']).reset_index(drop=True)
print(f"dedup: tingkat-kelas {_n0[0]}->{len(CLS)}, intra-kelas {_n0[1]}->{len(WITr)}")
TEST={str(s) for s in CLS.struct_id}|{str(s) for s in WITr.struct_id}
REFSET={s for s,_ in ind_set}-TEST; N_REF=len(REFSET)
ref_dis={c:(ss&REFSET) for c,ss in by_dis.items()}
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
    memb,inv=G[kind]; p=len(ref_dis.get(cui,()))/max(N_REF,1); best=None
    for g in memb.get(sid,()):
        sibs=inv.get(g)
        if not sibs: continue
        k=len(sibs&ref_dis.get(cui,set()))
        best=max(best or 0.0,(k+ALPHA*p)/(len(sibs)+ALPHA))
    return p if best is None else best
def feats(df):
    out=[]
    for sid,cui in zip(df.struct_id.astype(str),df.umls_cui):
        f={k:prior(k,sid,cui) for k in G}
        f['dis_prior']=len(ref_dis.get(cui,()))/max(N_REF,1)
        f['drug_deg']=np.log1p(sum(1 for c in by_dis if sid in by_dis[c]))
        out.append(f)
    return pd.DataFrame(out,index=df.index)
def baseline(D,within):
    X=feats(D); y=D.label.values; F=list(G)+['dis_prior','drug_deg']; M=X[F].values
    oof=np.zeros(len(D))
    for tr,te in GroupKFold(5).split(M,y,D.drugbank_id.values):
        sc=StandardScaler().fit(M[tr])
        oof[te]=LogisticRegression(max_iter=2000).fit(sc.transform(M[tr]),y[tr]
                ).predict_proba(sc.transform(M[te]))[:,1]
    key=[GKEY[(d,c)] for d,c in zip(D.drugbank_id,D.umls_cui)] if within else D.drugbank_id.values
    res={}
    for lab,sc_ in [('Regresi logistik (semua fitur)',oof),('Prior kelas EPC (lookup)',X['epc'].values),
                    ('Prior ATC level-3',X['atc3'].values),('Prior target bersama',X['tgt'].values)]:
        _,v=per_group(y,key,sc_); n=len(v)
        d=np.array([v[rng.integers(0,n,n)].mean() for _ in range(B)])
        res[lab]=(v.mean(),np.percentile(d,2.5),np.percentile(d,97.5),n)
    return res
print("menghitung baseline..."); BASE={'tingkat-kelas':baseline(CLS,False),'intra-kelas':baseline(WITr,True)}

# --- baseline graf pengetahuan (dihitung oleh src/kg_baseline.py)
# Titik acuan ATAS: metode non-LLM terkuat pada kelompok yang persis sama.
KGPG={}; KGB={}
for task,D_,w_,sfx in (('tingkat-kelas',CLS,False,'cls'),('intra-kelas',WITr,True,'wit')):
    oof=np.load(f'results/kg_oof_{sfx}.npy')
    key=[GKEY[(a,b)] for a,b in zip(D_.drugbank_id,D_.umls_cui)] if w_ else D_.drugbank_id.values
    k,v=per_group(D_.label.values,key,oof)
    KGPG[task]=pd.Series(v,index=k)
    d=np.array([v[rng.integers(0,len(v),len(v))].mean() for _ in range(B)])
    KGB[task]=(v.mean(),np.percentile(d,2.5),np.percentile(d,97.5),len(v))
KGSUM=pd.read_csv('results/kg_baseline.csv')
def paired_kg(g,task,within):
    Mm,idx=matrix(g,within)
    a=pd.Series(Mm.mean(axis=1),index=idx); b=KGPG[task]
    c=a.index.intersection(b.index); d=(a[c]-b[c]).values; n=len(d)
    dr=np.array([d[rng.integers(0,n,n)].mean() for _ in range(5000)])
    return d.mean(), np.percentile(dr,2.5), np.percentile(dr,97.5)

# ================= lantai permutasi =================
def floor_(m,within):
    g=llm((H if within else E)[m],within); gg=g[g.template=='qa']
    key=np.array([GKEY[(d,c)] for d,c in zip(gg.drugbank_id,gg.umls_cui)]) if within else gg.drugbank_id.values
    y=gg.label.values; s=gg.score.values; r=np.random.default_rng(1); out=[]
    idx=pd.Series(range(len(y))).groupby(gg.drugbank_id.values).apply(lambda v:v.values)
    for _ in range(20):
        yy=y.copy()
        for i in idx: yy[i]=r.permutation(y[i])
        out.append(auroc(yy,s))
    _,v=per_group(y,key,s)
    return float(np.mean(out)), float(v.mean())
def base_pergroup(D,within):
    X=feats(D); y=D.label.values; F=list(G)+['dis_prior','drug_deg']; M=X[F].values
    oof=np.zeros(len(D))
    for tr,te in GroupKFold(5).split(M,y,D.drugbank_id.values):
        sc=StandardScaler().fit(M[tr])
        oof[te]=LogisticRegression(max_iter=2000).fit(sc.transform(M[tr]),y[tr]
                ).predict_proba(sc.transform(M[te]))[:,1]
    key=[GKEY[(d,c)] for d,c in zip(D.drugbank_id,D.umls_cui)] if within else D.drugbank_id.values
    k,v=per_group(y,key,oof); return pd.Series(v,index=k)
BPG={'tingkat-kelas':base_pergroup(CLS,False),'intra-kelas':base_pergroup(WITr,True)}
def paired(m,task,within):
    """Selisih LLM - baseline, bootstrap pada kelompok YANG SAMA (uji berpasangan,
    jauh lebih kuat daripada melihat tumpang tindih dua CI terpisah)."""
    Mm,idx=matrix(llm((H if within else E)[m],within),within)
    a=pd.Series(Mm.mean(axis=1),index=idx); b=BPG[task]
    c=a.index.intersection(b.index); d=(a[c]-b[c]).values; n=len(d)
    dr=np.array([d[rng.integers(0,n,n)].mean() for _ in range(5000)])
    return d.mean(), np.percentile(dr,2.5), np.percentile(dr,97.5)
def paired_df(g,task,within):
    Mm,idx=matrix(g,within)
    a=pd.Series(Mm.mean(axis=1),index=idx); b=BPG[task]
    c=a.index.intersection(b.index); d=(a[c]-b[c]).values; n=len(d)
    dr=np.array([d[rng.integers(0,n,n)].mean() for _ in range(5000)])
    return d.mean(), np.percentile(dr,2.5), np.percentile(dr,97.5)
PAIRED={(t,m):paired(m,t,w) for t,w in (('tingkat-kelas',False),('intra-kelas',True)) for m in MS}
print("menghitung lantai permutasi..."); FLOOR={m:(floor_(m,False),floor_(m,True)) for m in MS}

# ================= hitung LLM =================
print("menghitung LLM...")
R={}
for task,within in (('tingkat-kelas',False),('intra-kelas',True)):
    for m in MS: R[(task,m)]=eval_llm(m,within)
np.save('results/rebuild_cache.npy',{'R':{k:(v[0],v[1],v[2],list(v[3]),v[4]) for k,v in R.items()},
                                     'BASE':BASE,'FLOOR':FLOOR},allow_pickle=True)


# --- keterangan figur: SATU sumber teks.
# Saat membangun untuk naskah (default) teks tidak dicetak ke dalam PNG
# melainkan ditulis ke paper/tables/cap-*.tex dan dipakai oleh \caption LaTeX.
# Kalau tidak, ia tertanam di gambar DAN muncul lagi di caption -- duplikasi
# yang pasti diminta jurnal untuk dibuang.
import os as _os
# Default 0: figur ditujukan untuk naskah, dan di sana keterangannya dibawa
# \caption LaTeX. FIG_CAPTIONS=1 menanamkannya ke PNG untuk dilihat lepas.
FIGCAP = _os.environ.get('FIG_CAPTIONS','0') == '1'
_os.makedirs('paper/tables',exist_ok=True)
def tex_escape(t):
    """% dan & akan mematahkan LaTeX; karakter unicode seperti >= tidak ada di
    enkode T1. Keduanya disalin apa adanya ke figur, tapi harus diubah untuk
    berkas .tex."""
    for a,b in (('\\','\\textbackslash '),('&','\\&'),('%','\\%'),('#','\\#'),
                ('_','\\_'),('$','\\$'),('\u2265','$\\geq$'),('\u2264','$\\leq$'),
                ('\u2013','--'),('\u2014','---'),('\u2212','$-$')):
        t=t.replace(a,b)
    return t
# Nama makro TeX tidak boleh memuat angka, jadi F1 -> Fone dst.
_ROMAN={'F1':'Fone','F2':'Ftwo','F3':'Fthree'}
def cap(fig,name,text,y=-0.055):
    flat=" ".join(text.replace("\n"," ").split())
    # Ditulis sebagai \newcommand, bukan teks polos: \input tidak bisa dipakai
    # di dalam \caption (argumen bergerak, dan hyperref gagal mengurainya).
    open(f'paper/tables/cap-{name}.tex','w').write(
        "\\newcommand{\\figcap"+_ROMAN[name]+"}{"+tex_escape(flat)+"}\n")
    if FIGCAP: fig.text(0.01,y,text,fontsize=8,color=MUT)

# ================= F1: dua panel, dua keluarga model =================
QW={'1.7B':'Qwen3-1.7B','8B':'Qwen3-8B','14B':'Qwen3-14B','32B':'Qwen3-32B'}
QPAR={'1.7B':2.03,'8B':8.19,'14B':14.77,'32B':32.76}
def llm_qwen(tag,within):
    pre='wit' if within else 'cls'
    d=pd.DataFrame([json.loads(l) for l in open(f'results/{pre}_{QW[tag]}.jsonl')])
    g=d[d.condition=='A0'] if within else pd.concat(
        [d[d.tier=='pos'],d[d.tier=='neg_random']]).query("condition=='A0'")
    return g.drop_duplicates(['drugbank_id','umls_cui','template'])
def qwen_eval(tag,within): return summarize(matrix(llm_qwen(tag,within),within)[0])
RQ={}
for t in QW:
    RQ[('tingkat-kelas',t)]=qwen_eval(t,False); RQ[('intra-kelas',t)]=qwen_eval(t,True)

# Ambang yang ditampilkan adalah baseline non-LLM TERKUAT per tugas -- memilih
# yang lebih lemah akan membuat LLM terlihat lebih baik daripada seharusnya.
def strongest(task):
    a=BASE[task]['Regresi logistik (semua fitur)']; b=KGB[task]
    return (b[:3],'KG') if b[0]>a[0] else (a[:3],'lookup')
bl_c,bn_c=strongest('tingkat-kelas'); bl_i,bn_i=strongest('intra-kelas')
PANEL=[("OLMo-2",MS,PAR,lambda t,m:R[(t,m)]),
       ("Qwen3",list(QW),QPAR,lambda t,m:RQ[(t,m)])]
fig,axes=plt.subplots(1,2,figsize=(11.2,4.8),sharey=True)
for ax,(fam,keys,par,get) in zip(axes,PANEL):
    x=[par[m] for m in keys]
    ax.grid(axis='y',zorder=0)
    for task,c,lab,bl in (('tingkat-kelas',S1,"Class-level task",bl_c),
                          ('intra-kelas',S2,"Within-class task",bl_i)):
        dat=[get(task,m) for m in keys]
        a=[d[0] for d in dat]; lo=[d[1] for d in dat]; hi=[d[2] for d in dat]
        ax.fill_between(x,lo,hi,color=c,alpha=0.15,lw=0,zorder=2)
        for xi,d in zip(x,dat): ax.plot([xi]*3,d[3],'o',ms=3.2,color=c,alpha=0.45,zorder=3)
        ax.plot(x,a,color=c,lw=2,marker='o',ms=7.5,mec=SURF,mew=2,zorder=4,label=lab)
        ax.axhline(bl[0],ls=(0,(5,3)),lw=1.4,color=c,alpha=0.5,zorder=1)
        for xi,v in zip(x,a):
            dy=12 if task=='tingkat-kelas' else -22
            ax.annotate(e3(v),(xi,v),xytext=(0,dy),textcoords='offset points',
                        ha='center',fontsize=8,color=SEC)
    ax.set_xscale('log'); ax.set_xticks(x); ax.set_xticklabels(keys)
    ax.set_xlim(min(x)*0.75,max(x)*1.5); ax.set_ylim(0.44,1.0)
    ax.set_title(fam,loc='left',color=INK,fontsize=11,pad=8)
    # Tanda centang diberi NAMA checkpoint, tetapi diletakkan pada jumlah
    # parameter sebenarnya -- mis. "1B" berada di 1,48. Label sumbu dibuat
    # eksplisit supaya tidak terbaca seolah "1B" berarti tepat satu miliar.
    ax.set_xlabel("Checkpoint, positioned by parameter count (log scale)")
axes[0].set_ylabel("Within-group macro AUROC")
axes[0].yaxis.set_major_formatter(dot)
axes[0].annotate(f"best non-LLM baseline {e3(bl_c[0])}",(min(PAR.values())*0.78,bl_c[0]),
                 xytext=(2,4),textcoords='offset points',ha='left',color=S1,fontsize=8,alpha=0.9)
# Label oranye dipasang di panel kanan: di panel kiri ia bertumpuk dengan titik
# 1B. Garisnya sendiri tetap tergambar di kedua panel.
axes[1].annotate(f"best non-LLM baseline {e3(bl_i[0])}",(min(QPAR.values())*0.78,bl_i[0]),
                 xytext=(2,4),textcoords='offset points',ha='left',color=S2,fontsize=8,alpha=0.9)
axes[0].legend(frameon=False,loc='upper left',fontsize=9.5,labelcolor=SEC,
               bbox_to_anchor=(0.0,0.99))
for ax,(fam,keys,par,get) in zip(axes,PANEL):
    if fam!="Qwen3": continue
    for task,c,lab in (('tingkat-kelas',S1,"Class-level"),('intra-kelas',S2,"Within-class")):
        v=get(task,keys[-1])[0]
        ax.annotate(lab,(par[keys[-1]],v),xytext=(9,0),textcoords='offset points',
                    color=c,fontsize=9,va='center')
cap(fig,'F1',
 "Within-group macro AUROC against model size for the two tasks in two model families.",
 y=-0.05)
fig.tight_layout(); fig.savefig("figures/F1_dua_kurva.png",dpi=200,bbox_inches='tight')
print("F1 ditulis (2 panel)")

# ================= F2: tangga perturbasi, 7B vs 32B =================
conds=[('A0','Canonical drug name'),('A4','Random pseudonym'),('A5c','Pharmacologic class'),
       ('A5t','Class + target'),('A5','Class + target + ATC')]
d32=pd.concat([pd.DataFrame([json.loads(l) for l in open(f)]) for f in
   ['results/easy_OLMo-2-0325-32B-Instruct.jsonl','results/ladder32_OLMo.jsonl']])
def lad(src,c):
    g=pd.concat([src[src.tier=='pos'],src[src.tier=='neg_random']]).query(f"condition=='{c}'")
    return summarize(matrix(g.drop_duplicates(['drugbank_id','umls_cui','template']),False)[0])
d7=pd.DataFrame([json.loads(l) for l in open(E['7B'])])
# Sumber tangga untuk keluarga KEDUA (eksperimen A1). Tanpa ini seluruh
# analisis mekanisme (3.4, 3.5) bersandar pada OLMo-2 saja, padahal klaim
# utama naskah adalah tentang dua keluarga model -- lubang terbesar yang
# bisa ditunjuk pengulas. Berbeda dari d32, A0 dan A4/A5c/A5t/A5 Qwen3 ada
# dalam SATU berkas, jadi tidak perlu digabung.
dq8 =pd.DataFrame([json.loads(l) for l in open('results/cls_Qwen3-8B.jsonl')])
dq32=pd.DataFrame([json.loads(l) for l in open('results/cls_Qwen3-32B.jsonl')])
LADSRC=[('OLMo-2 7B',d7),('OLMo-2 32B',d32),('Qwen3 8B',dq8),('Qwen3 32B',dq32)]
SER=[("OLMo-2 32B-Instruct",d32,S1,+0.16,9),("OLMo-2 7B-Instruct",d7,S2,-0.16,-15)]
fig,ax=plt.subplots(figsize=(7.6,4.4))
yp=np.arange(len(conds))[::-1]
ax.grid(axis='x',zorder=0)
ax.axvline(0.5,ls=(0,(4,3)),lw=1.2,color=MUT,zorder=1)
ax.text(0.5,len(conds)-0.35,"chance",color=MUT,fontsize=8.5,ha='center')
for nm,src,col,off,dy in SER:
    for y_,(code,_) in zip(yp,conds):
        p,lo,hi,per,_=lad(src,code)
        ax.plot([lo,hi],[y_+off]*2,color=col,lw=1.8,solid_capstyle='round',zorder=2)
        ax.plot([p],[y_+off],'o',ms=8,color=col,mec=SURF,mew=1.8,zorder=4)
        ax.annotate(e3(p),(p,y_+off),xytext=(0,dy),textcoords='offset points',
                    ha='center',fontsize=8,color=SEC)
    ax.plot([],[],'o-',color=col,lw=1.8,ms=8,label=nm)
ax.set_yticks(yp); ax.set_yticklabels([l for _,l in conds],color=SEC)
ax.set_xlim(0.44,0.99); ax.set_ylim(-0.6,len(conds)-0.2); ax.xaxis.set_major_formatter(dot)
ax.set_xlabel("Within-drug macro AUROC (class-level task)")
ax.legend(frameon=False,loc='lower left',fontsize=9,labelcolor=SEC)
# F2 tidak lagi masuk naskah: sepuluh angkanya sama dengan Tabel 12 dan tabel
# itu masih membawa kolom Groups. Gambarnya tetap dibuat sebagai alat periksa,
# tetapi keterangan LaTeX-nya tidak ditulis lagi supaya tidak ada makro mati.
fig.tight_layout(); fig.savefig("figures/F2_tangga_perturbasi.png",dpi=200,bbox_inches='tight')
print("F2 ditulis (7B vs 32B)")

# ================= F3 =================
raw=pd.DataFrame([json.loads(l) for l in open(E['7B'])])
SZ=[('A5c','coarse_class_size',S3,PN['A5c']),('A5t','a5t_class_size',S2,PN['A5t']),
    ('A5','sig_class_size',S1,PN['A5'])]
bins=[(1,1,'unique\n(n=1)'),(2,3,'2–3'),(4,10,'4–10'),(11,10**6,'≥11\n(unidentifiable)')]
fig,ax=plt.subplots(figsize=(7.0,4.4))
ax.grid(axis='y',zorder=0)
# garis kebetulan 0,50 sengaja TIDAK digambar: seluruh data berada di 0,72-0,92,
# memaksakannya ke dalam rentang akan mengosongkan separuh panel tanpa informasi.
xs=np.arange(len(bins),dtype=float); off={'A5c':-0.05,'A5t':0.0,'A5':0.05}
LAB={'A5c':f"{PN['A5c']}  class only",'A5t':f"{PN['A5t']}  class + target",
     'A5':f"{PN['A5']}  class+target+ATC"}
for code,fld,col,short in SZ:
    M,idx=matrix(llm(E['7B'],False,code),False)
    sz=raw[raw.condition==code].drop_duplicates('drugbank_id').set_index('drugbank_id')[fld]
    s=np.array([sz.get(d,0) for d in idx])
    ys=[];los=[];his=[]
    for lo_,hi_,_ in bins:
        sel=(s>=lo_)&(s<=hi_)
        if sel.sum()>=15:
            p,l,h,_,_=summarize(M[sel]); ys.append(p);los.append(l);his.append(h)
        else: ys.append(np.nan);los.append(np.nan);his.append(np.nan)
    xo=xs+off[code]
    for xi,l_,h_ in zip(xo,los,his):
        if not np.isnan(l_): ax.plot([xi,xi],[l_,h_],color=col,lw=1.6,alpha=0.7,solid_capstyle='round',zorder=2)
    ax.plot(xo,ys,color=col,lw=2,marker='o',ms=8,mec=SURF,mew=2,zorder=3,label=LAB[code])
    ax.annotate(short,(xo[0],ys[0]),xytext=(-12,0),textcoords='offset points',
                color=col,fontsize=9.5,va='center',ha='right')
    # Nilai UJUNG tiap seri dilabeli. Tanpa ini angka yang dibahas di prosa
    # (mis. 0,717 dan 0,855 untuk P3) hanya dapat dibaca dari posisi titik,
    # dan pembaca tidak bisa memeriksanya -- aturan yang sama dengan F1.
    # Titik awal dilabeli di ATAS, titik akhir di KANAN. Menaruh keduanya di
    # atas membuat label seri yang berakhir di bin sama saling menimpa, dan
    # label P5 jatuh tepat pada penanda P4.
    _ok=[i for i,v in enumerate(ys) if not np.isnan(v)]
    if _ok:
        _f=lambda v: f"{v:.3f}".replace('.',',')
        ax.annotate(_f(ys[_ok[0]]),(xo[_ok[0]],ys[_ok[0]]),xytext=(0,9),
                    textcoords='offset points',color=col,fontsize=8.5,
                    ha='center',va='bottom',zorder=4)
        if _ok[-1]!=_ok[0]:
            ax.annotate(_f(ys[_ok[-1]]),(xo[_ok[-1]],ys[_ok[-1]]),xytext=(8,0),
                        textcoords='offset points',color=col,fontsize=8.5,
                        ha='left',va='center',zorder=4)
ax.set_xticks(xs); ax.set_xticklabels([b[2] for b in bins],color=SEC,fontsize=9)
ax.set_xlim(-0.55,len(bins)-0.6); ax.set_ylim(0.58,1.0); ax.yaxis.set_major_formatter(dot)
ax.set_xlabel("Number of drugs sharing the same description")
ax.set_ylabel("Within-drug macro AUROC")
ax.legend(frameon=False,loc='lower right',fontsize=9,labelcolor=SEC)
cap(fig,'F3',
 "Within-drug macro AUROC in OLMo-2 7B, stratified by how many drugs share "
 "the same mechanism description.")
fig.tight_layout(); fig.savefig("figures/F3_identifiabilitas.png",dpi=200,bbox_inches='tight')
print("F3 ditulis")

# ================= TABEL =================
def glob_auroc(m,within,t='qa'):
    g=llm((H if within else E)[m],within); gg=g[g.template==t]
    return auroc(gg.label.values,gg.score.values)

L=["# Tabel hasil lengkap\n",
"Metrik: **AUROC makro dalam-kelompok**, dihitung terpisah untuk tiap template prompt "
"lalu dirata-ratakan. Kelompok = obat (tugas tingkat-kelas) atau (penyakit, kelas EPC) "
"(tugas intra-kelas, yaitu unit yang dicocokkan saat set dibangun). CI 95% dari bootstrap "
"pada tingkat kelompok, B = 2.000.\n",
"Angka utama memakai **rerata tiga template**, bukan template terbaik: memilih template "
"terbaik di set uji adalah seleksi pada data uji dan bias optimistis.\n",
"\n## Tabel 1 — Mengapa metrik lama tidak dipakai\n",
"Mengacak label di dalam tiap obat seharusnya memberi AUROC 0,50. Pada AUROC **global** "
"hasilnya jauh di atas itu, dan lantainya **berbeda per model** — jadi AUROC global tidak "
"bisa dipakai membandingkan model.\n",
"| Model | Tugas | AUROC global | Lantai permutasi | AUROC makro |","|---|---|---|---|---|"]
for m in MS:
    for i,(task,within) in enumerate((('tingkat-kelas',False),('intra-kelas',True))):
        fl,mac=FLOOR[m][i]
        L.append(f"| {NAMA[m]} | {task} | {k3(glob_auroc(m,within))} | {k3(fl)} | {k3(mac)} |")
L.append("\nSemua angka di baris ini memakai template `qa` agar sebanding. Lantai permutasi: "
         "20 ulangan pengacakan label di dalam tiap obat.\n")

L+=["\n## Tabel 2 — Matriks utama: LLM vs baseline lookup\n",
f"Kondisi A0 (nama obat kanonik). Tugas tingkat-kelas: {ribu(len(CLS))} pasangan, "
f"{KGB['tingkat-kelas'][3]} kelompok obat. Tugas intra-kelas: {ribu(len(WITr))} pasangan, "
f"{ribu(KGB['intra-kelas'][3])} kelompok (penyakit, kelas).\n",
"| Sistem | Tingkat-kelas | Intra-kelas | Jarak |","|---|---|---|---|"]
for lab in ['Prior kelas EPC (lookup)','Prior ATC level-3','Prior target bersama',
            'Regresi logistik (semua fitur)']:
    c=BASE['tingkat-kelas'][lab]; i=BASE['intra-kelas'][lab]
    nm=f"**{lab}**" if lab.startswith('Regresi') else lab
    L.append(f"| {nm} | {k3(c[0])} {ci3(c[1],c[2])} | {k3(i[0])} {ci3(i[1],i[2])} | {k3(c[0]-i[0])} |")
for r_ in KGSUM[KGSUM.tugas=='tingkat-kelas'].itertuples():
    j=KGSUM[(KGSUM.tugas=='intra-kelas')&(KGSUM.metode==r_.metode)].iloc[0]
    nm=f"**KG: {r_.metode}**" if 'kombinasi' in r_.metode else f"KG: {r_.metode}"
    L.append(f"| {nm} | {k3(r_.auroc)} {ci3(r_.lo,r_.hi)} | {k3(j.auroc)} {ci3(j.lo,j.hi)} | {k3(r_.auroc-j.auroc)} |")
for m in MS:
    c=R[('tingkat-kelas',m)]; i=R[('intra-kelas',m)]
    L.append(f"| {NAMA[m]} | {k3(c[0])} {ci3(c[1],c[2])} | {k3(i[0])} {ci3(i[1],i[2])} | {k3(c[0]-i[0])} |")
for t in QW:
    c=RQ[('tingkat-kelas',t)]; i=RQ[('intra-kelas',t)]
    L.append(f"| Qwen3 {t} | {k3(c[0])} {ci3(c[1],c[2])} | {k3(i[0])} {ci3(i[1],i[2])} | {k3(c[0]-i[0])} |")
gc=R[('tingkat-kelas','32B')][0]-R[('tingkat-kelas','1B')][0]
gi=R[('intra-kelas','32B')][0]-R[('intra-kelas','1B')][0]
L.append("\nPrior kelas murni **konstan secara konstruksi** pada tugas intra-kelas: kedua "
"saudara sekelas memakai himpunan referensi yang sama persis, sehingga fiturnya bernilai "
"identik (terkonfirmasi 0,499). Kritik \"model hanya memakai prior kelas\" karena itu tidak "
"dapat menjelaskan hasil intra-kelas sama sekali.\n")
L.append("Pola yang sama muncul di **dua keluarga model yang independen**. Pada 32B keduanya "
"nyaris identik: OLMo-2 0,921/0,689 dan Qwen3 0,932/0,700, dengan jarak 0,232 pada keduanya. "
"Qwen3 lebih kuat pada tiap ukuran yang sebanding, tetapi bentuk kurvanya sama.\n")
L.append("\n### Uji berpasangan terhadap baseline terkuat\n")
L.append("Selisih AUROC per kelompok (LLM − baseline), bootstrap pada kelompok yang sama. "
"Ini uji yang tepat; melihat tumpang tindih dua CI terpisah terlalu konservatif.\n")
L.append("Dua pembanding: baseline lookup (prior kelas/ATC/target) dan baseline KG "
"(ensembel graf indikasi, seluruh sisi uji dibuang). KG lebih kuat pada kedua tugas, "
"jadi kolom KG adalah ambang yang sebenarnya.\n")
L.append("| Model | Kelas vs lookup | Kelas vs KG | Intra vs lookup | Intra vs KG |")
L.append("|---|---|---|---|---|")
def vd(t):
    d,lo,hi=t
    tag="lebih baik" if lo>0 else ("lebih buruk" if hi<0 else "tidak terbedakan")
    return f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}] — {tag}".replace('.',',')
for m in MS:
    gc_=llm((E if True else H)[m],False); gi_=llm(H[m],True)
    L.append(f"| {NAMA[m]} | {vd(PAIRED[('tingkat-kelas',m)])} | {vd(paired_kg(gc_,'tingkat-kelas',False))} | "
             f"{vd(PAIRED[('intra-kelas',m)])} | {vd(paired_kg(gi_,'intra-kelas',True))} |")
for t in QW:
    gc_=llm_qwen(t,False); gi_=llm_qwen(t,True)
    L.append(f"| Qwen3 {t} | {vd(paired_df(gc_,'tingkat-kelas',False))} | {vd(paired_kg(gc_,'tingkat-kelas',False))} | "
             f"{vd(paired_df(gi_,'intra-kelas',True))} | {vd(paired_kg(gi_,'intra-kelas',True))} |")
L.append("\n**Ambang untuk mengungguli baseline non-LLM terkuat.** Pada tugas tingkat-kelas "
f"ambang terkuat adalah ensembel KG ({k3(KGB['tingkat-kelas'][0])}), sedangkan pada tugas "
f"intra-kelas adalah baseline lookup ({k3(BASE['intra-kelas']['Regresi logistik (semua fitur)'][0])}) "
"— setelah pasangan ganda dibuang, lookup justru melampaui KG di sana. Terhadap ambang itu, "
"keunggulan yang CI-nya seluruhnya di atas nol baru muncul pada **OLMo-2 32B** dan pada "
"**Qwen3 8B ke atas**.\n")
L.append("OLMo-2 7B tidak terbedakan dari baseline pada kedua tugas, dan OLMo-2 13B tidak "
"terbedakan pada tugas intra-kelas. Model terkecil di kedua keluarga (OLMo-2 1B-Instruct dan Qwen3 1,7B) "
"justru lebih buruk daripada baseline. Jadi pernyataan yang bisa dipertahankan adalah: LLM "
"mengungguli metode non-LLM terkuat yang kami uji mulai sekitar **8 miliar parameter pada Qwen3 "
"dan 32 miliar pada OLMo-2**. Di bawah itu, menjalankan LLM tidak memberi apa pun di atas "
"baseline yang berjalan dalam hitungan detik di laptop.\n")
AA=KGSUM[(KGSUM.tugas=='intra-kelas')&(KGSUM.metode.str.startswith('AA'))].iloc[0]
L.append(f"Catatan tentang pemilihan ambang: pada tugas intra-kelas, ensembel KG "
f"({k3(KGB['intra-kelas'][0])}) berada di bawah baseline lookup ({k3(BASE['intra-kelas']['Regresi logistik (semua fitur)'][0])}) "
f"dan juga di bawah metode KG tunggal terbaiknya, Adamic-Adar ({k3(AA.auroc)}). Kami memakai "
"angka tertinggi di antara ketiganya sebagai ambang, karena memilih yang lebih rendah akan "
"membuat LLM tampak lebih unggul daripada seharusnya. Baseline lookup dan ensembel KG keduanya "
"tervalidasi silang; Adamic-Adar tidak dipilih sebagai ambang karena memilihnya setelah melihat "
"skor ujinya adalah seleksi pada data uji, tetapi angkanya pun masih jauh di bawah kedua model "
"32B.\n")

L+=["\n## Tabel 3 — Tangga perturbasi, 7B vs 32B (tugas tingkat-kelas)\n",
"| Kondisi | Yang diberikan ke model | 7B rerata | 7B CI 95% | 32B rerata | 32B CI 95% |",
"|---|---|---|---|---|---|"]
DESC={'A0':'Nama obat kanonik','A4':'Pseudonim tak-ada-di-korpus','A5c':'Kelas farmakologis (EPC) saja',
      'A5t':'Kelas + target molekuler','A5':'Kelas + target + kategori ATC'}
for c,_ in conds:
    a=lad(d7,c); b=lad(d32,c)
    L.append(f"| {PN[c]} | {DESC[c]} | **{k3(a[0])}** | {ci3(a[1],a[2])} | **{k3(b[0])}** | {ci3(b[1],b[2])} |")
a0_7=lad(d7,'A0')[0]; a5c_7=lad(d7,'A5c')[0]
a0_32=lad(d32,'A0')[0]; a5c_32=lad(d32,'A5c')[0]

def pergroup_cond(src,c):
    """AUROC per obat untuk satu kondisi tangga (rerata atas template)."""
    g=pd.concat([src[src.tier=='pos'],src[src.tier=='neg_random']]).query(f"condition=='{c}'")
    Mm,idx=matrix(g.drop_duplicates(['drugbank_id','umls_cui','template']),False)
    return pd.Series(Mm.mean(axis=1),index=idx)
# Retensi sinyal di atas kebetulan, dihitung pada OBAT YANG SAMA.
#
# Versi lama membagi nilai P3 (270 obat) dengan nilai P1 (363 obat) dan
# menghasilkan 93% / 96%. Ke-93 obat tanpa anotasi EPC ternyata LEBIH SULIT --
# P1 atas 270 obat beranotasi adalah 0,866 di 7B lawan 0,851 atas seluruh 363 --
# jadi rasio silang-himpunan itu terlalu tinggi empat poin, tepat ke arah yang
# menguatkan klaim naskah. Angka yang benar 89% / 92%, dan cocok dengan
# sembilan persepuluh yang tersirat dari uji berpasangan Tabel 15.
#
# Tidak ada undian rng di blok ini, jadi tak satu pun selang di tabel mana pun
# bergeser karenanya.
def retensi(src):
    a=pergroup_cond(src,'A0'); b=pergroup_cond(src,'A5c')
    k=a.index.intersection(b.index)
    p1,p3=a[k].mean(),b[k].mean()
    return 100*(p3-0.5)/(p1-0.5), p1, p3, len(k), len(a)
RET={nm:retensi(src) for nm,src in LADSRC}
print("\nretensi kelas (P3) terhadap nama (P1), di atas kebetulan:")
for _nm,(_r,_p1,_p3,_n,_na) in RET.items():
    print(f"   {_nm:<11} P1={_p1:.3f} P3={_p3:.3f} pada n={_n} obat "
          f"(dari {_na}) -> {_r:.1f}%")

def paired_cond(src,c1,c2):
    """Selisih kondisi c1-c2 pada OBAT YANG SAMA, bootstrap berpasangan.
    Sebelumnya angka ini ditulis tangan di narasi; sekarang dihitung."""
    a=pergroup_cond(src,c1); b=pergroup_cond(src,c2)
    k=a.index.intersection(b.index); d=(a[k]-b[k]).values; n=len(d)
    dr=np.array([d[rng.integers(0,n,n)].mean() for _ in range(5000)])
    return d.mean(), np.percentile(dr,2.5), np.percentile(dr,97.5), n
PC={(sz,c):paired_cond(src,'A0',c) for sz,src in
    (('7B',d7),('32B',d32),('q8',dq8),('q32',dq32))
    for c in ('A4','A5c','A5t','A5')}
L.append("\nPseudonim tanpa informasi tetap persis di kebetulan pada kedua skala (0,491 dan 0,494) — "
"kontrol ini memastikan sinyalnya datang dari isi deskripsi, bukan dari format prompt.\n")
L.append("Deskripsi mekanisme saja memulihkan **"
+f"{100*(a5c_7-0.5)/(a0_7-0.5):.0f}".replace('.',',')+"% (7B)** dan **"
+f"{100*(a5c_32-0.5)/(a0_32-0.5):.0f}".replace('.',',')+"% (32B)** dari sinyal di atas kebetulan "
"yang dibawa nama obat asli. Namun uji berpasangan menunjukkan nama obat tetap unggul tipis tapi "
"nyata: " + "; ".join(
    f"di {sz} {PN['A0']} vs {PN[c]} = {k3(PC[(sz,c)][0])} {ci3(PC[(sz,c)][1],PC[(sz,c)][2])}"
    for sz in ('7B','32B') for c in ('A5c','A5t','A5')) + ".\n")
L.append("Artinya: hampir seluruh kemampuan ini bersandar pada mekanisme farmakologis, bukan pada "
"identitas obat — tetapi ada kontribusi kecil dari identitas yang menjadi terdeteksi secara "
"konsisten hanya pada skala 32B. Klaim \"deskripsi mekanisme menyamai nama obat\" harus dibatasi "
"pada A5 di 7B saja.\n")

L+=["\n## Tabel 4 — Stratifikasi identifiabilitas (7B-Instruct, tugas tingkat-kelas)\n",
"Ukuran tanda-tangan = berapa obat berbagi deskripsi yang persis sama.\n",
"| Kondisi | Ukuran tanda-tangan | n obat | Rerata | CI 95% |","|---|---|---|---|---|"]
for code,fld,_,_ in SZ:
    M,idx=matrix(llm(E['7B'],False,code),False)
    sz=raw[raw.condition==code].drop_duplicates('drugbank_id').set_index('drugbank_id')[fld]
    sv=np.array([sz.get(d,0) for d in idx])
    for lo_,hi_,lab in [(1,1,'1 (unik)'),(2,3,'2–3'),(4,10,'4–10'),(11,10**6,'≥ 11')]:
        sel=(sv>=lo_)&(sv<=hi_)
        if sel.sum()>=15:
            p,l,h,_,n=summarize(M[sel])
            L.append(f"| {code} | {lab} | {n} | {k3(p)} | {ci3(l,h)} |")
        else:
            L.append(f"| {code} | {lab} | {int(sel.sum())} | — | terlalu sedikit |")
L.append("\nTidak ada penurunan monoton; arahnya justru naik. Strata paling tidak teridentifikasi "
"tidak lebih buruk daripada strata unik pada ketiga kondisi. Penurunan A5 pada strata ≥ 11 yang "
"dilaporkan sebelumnya tidak dapat diuji di sini: dengan metrik makro hanya <15 obat memenuhi "
"syarat di strata itu.\n")
L.append("\n---\n*Dihasilkan oleh `src/rebuild_outputs.py`. Baseline dari `src/baseline_vs_llm.py`. "
"Total 180.222 forward pass, empat model OLMo-2 (batas data latih Desember 2023), tiga template prompt.*")
open('tabel-hasil.md','w').write("\n".join(L)+"\n")
pd.DataFrame([dict(tugas=t,model=NAMA[m],rerata=round(R[(t,m)][0],4),
                   lo=round(R[(t,m)][1],4),hi=round(R[(t,m)][2],4),
                   qa=round(R[(t,m)][3][0],4),tf=round(R[(t,m)][3][1],4),cloze=round(R[(t,m)][3][2],4))
              for t in ['tingkat-kelas','intra-kelas'] for m in MS]).to_csv('results/tabel_hasil.csv',index=False)
print("tabel-hasil.md + results/tabel_hasil.csv ditulis")

# ================= TABEL LaTeX (untuk paper/) =================
# Dihasilkan dari objek yang sama dengan figur & tabel Markdown, jadi angka di
# manuskrip tidak pernah menyimpang dari hasil. Desimal titik (naskah Inggris).
import os
os.makedirs('paper/tables',exist_ok=True)
def w(fn,txt): open(f'paper/tables/{fn}','w').write(txt+"\n")
def c3(t): return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"

BL='Regresi logistik (semua fitur)'
rows=[("Class prior (EPC)",BASE['tingkat-kelas']['Prior kelas EPC (lookup)'],BASE['intra-kelas']['Prior kelas EPC (lookup)']),
      ("ATC level-3 prior",BASE['tingkat-kelas']['Prior ATC level-3'],BASE['intra-kelas']['Prior ATC level-3']),
      ("Shared-target prior",BASE['tingkat-kelas']['Prior target bersama'],BASE['intra-kelas']['Prior target bersama']),
      ("\\textbf{Logistic regression (all features)}",BASE['tingkat-kelas'][BL],BASE['intra-kelas'][BL])]
EN={'1B':'OLMo-2 1B-Instruct','7B':'OLMo-2 7B-Instruct','13B':'OLMo-2 13B-Instruct','32B':'OLMo-2 32B-Instruct'}
t=["\\begin{table}[!tbp]","\\centering","\\small",
 "\\caption{Macro within-group \\auroc{} of every system on the \\clstask{} and the "
 "\\withintask{}.}",
 "\\label{tab:main}","\\begin{tabular}{lccc}","\\toprule",
 "System & Class-level & Within-class & Gap \\\\","\\midrule"]
for nm,a,b in rows: t.append(f"{nm} & {c3(a)} & {c3(b)} & ${a[0]-b[0]:.3f}$ \\\\")
for r_ in KGSUM[KGSUM.tugas=='tingkat-kelas'].itertuples():
    j=KGSUM[(KGSUM.tugas=='intra-kelas')&(KGSUM.metode==r_.metode)].iloc[0]
    # Label tabel harus bisa dibaca tanpa membuka kode; singkatan AA/TGT tidak.
    _KGEN={'AA':'Adamic--Adar','Katz':'Katz index','SVD':'Truncated SVD',
           'TGT':'Target propagation'}
    nm=_KGEN.get(r_.metode.split()[0],r_.metode.split()[0])
    nm=("\\textbf{KG ensemble}" if 'kombinasi' in r_.metode else f"KG: {nm}")
    t.append(f"{nm} & {r_.auroc:.3f} [{r_.lo:.3f}, {r_.hi:.3f}] & "
             f"{j.auroc:.3f} [{j.lo:.3f}, {j.hi:.3f}] & ${r_.auroc-j.auroc:.3f}$ \\\\")
t.append("\\midrule")
for m in MS:
    a=R[('tingkat-kelas',m)]; b=R[('intra-kelas',m)]
    t.append(f"{EN[m]} & {c3(a)} & {c3(b)} & ${a[0]-b[0]:.3f}$ \\\\")
t.append("\\midrule")
for q in QW:
    a=RQ[('tingkat-kelas',q)]; b=RQ[('intra-kelas',q)]
    t.append(f"Qwen3 {q} & {c3(a)} & {c3(b)} & ${a[0]-b[0]:.3f}$ \\\\")
t+=["\\bottomrule","\\end{tabular}","\\end{table}"]
w('tab-main.tex',"\n".join(t))

t=["\\begin{table}[!tbp]","\\centering","\\small",
 "\\caption{Paired difference between each model and the two non-LLM baselines.}",
 "\\label{tab:paired}","\\begin{tabular}{lcc}","\\toprule",
 "Model & vs.\\ lookup baseline & vs.\\ knowledge-graph ensemble \\\\","\\midrule"]
def bf(x):
    body=f"{x[0]:+.3f}\\;[{x[1]:+.3f},\\;{x[2]:+.3f}]"
    # \textbf tidak menembus mode matematika, jadi seluruh sel dijadikan
    # matematika dan ditebalkan dengan \mathbf.
    return "$\\mathbf{"+body+"}$" if (x[1]>0 or x[2]<0) else "$"+body+"$"
def block(title,task,within):
    t.append(f"\\multicolumn{{3}}{{l}}{{\\textit{{{title}}}}} \\\\")
    for m in MS:
        g=llm((H if within else E)[m],within)
        t.append(f"\\quad {EN[m]} & {bf(PAIRED[(task,m)])} & {bf(paired_kg(g,task,within))} \\\\")
    for q in QW:
        g=llm_qwen(q,within)
        t.append(f"\\quad Qwen3 {q} & {bf(paired_df(g,task,within))} & {bf(paired_kg(g,task,within))} \\\\")
block("Class-level task",'tingkat-kelas',False)
t.append("\\addlinespace")
block("Within-class task",'intra-kelas',True)
t+=["\\bottomrule","\\end{tabular}","\\end{table}"]
w('tab-paired.tex',"\n".join(t))

DE={'A0':'Canonical drug name','A4':'Corpus-absent pseudonym','A5c':'Pharmacologic (EPC) class only',
    'A5t':'Class + molecular targets','A5':'Class + targets + ATC category'}
# Kolom "What the model is given" DIBUANG: definisi tiap kondisi beserta contoh
# teks penggantinya sudah ada di Tabel~\ref{tab:ladderdef} di Metodologi, dan
# ruangnya dibutuhkan untuk dua kolom keluarga kedua.
t=["\\begin{table}[!tbp]","\\centering","\\small",
 "\\setlength{\\tabcolsep}{4pt}",
 "\\caption{Macro within-group \\auroc{} under the perturbation ladder on the class-level "
 "task, in both model families.}",
 "\\label{tab:ladder}","\\begin{tabular}{@{}lccccc@{}}","\\toprule",
 "& & \\multicolumn{2}{c}{OLMo-2} & \\multicolumn{2}{c}{Qwen3} \\\\",
 "\\cmidrule(lr){3-4}\\cmidrule(l){5-6}",
 "Condition & Groups & 7B & 32B & 8B & 32B \\\\","\\midrule"]
for c,_ in conds:
    # Jumlah kelompok BERBEDA antar kondisi (anotasi EPC tidak lengkap); tanpa
    # kolom ini baris P1 dan P3 terlihat sebanding padahal bukan. SATU kolom
    # cukup hanya selama keempat model memberi jumlah yang sama -- diperiksa,
    # bukan diasumsikan, supaya tidak diam-diam jadi dusta kalau set berubah.
    _vs=[lad(src,c) for _,src in LADSRC]
    _ng={v[4] for v in _vs}
    assert len(_ng)==1, f"jumlah kelompok berbeda antar model pada {c}: {_ng}"
    # Baris P1 adalah KUANTITAS YANG SAMA dengan kolom tingkat-kelas tab-main.
    # Yang ditampilkan karena itu diambil dari R[]/RQ[] -- nilai yang dipakai
    # tab-main -- supaya dua tabel tidak memberi dua selang berlainan untuk satu
    # kuantitas, yang akan dibaca pengulas sebagai galat. lad() DI ATAS TETAP
    # DIPANGGIL walau hasilnya tidak dipakai untuk P1, karena ia memakan undian
    # dari rng global; menghapusnya akan menggeser setiap selang di tabel-tabel
    # sesudah blok ini. Idiomnya sama dengan baris Instruct di tab-base.
    _disp=_vs
    if c=='A0':
        _disp=[R[('tingkat-kelas','7B')],R[('tingkat-kelas','32B')],
               RQ[('tingkat-kelas','8B')],RQ[('tingkat-kelas','32B')]]
        for _a,_b in zip(_vs,_disp):
            assert abs(_a[0]-_b[0])<5e-4, \
                f"P1 tidak cocok dengan tab-main: {_a[0]:.4f} lawan {_b[0]:.4f}"
    # c3() menyisakan spasi setelah koma; dengan empat kolom model itu cukup
    # untuk meluber dari lebar teks, jadi di tabel ini saja dipakai versi rapat.
    _c3t=lambda v: f"{v[0]:.3f} [{v[1]:.3f},{v[2]:.3f}]"
    t.append(f"{PN[c]} & {_vs[0][4]} & "+" & ".join(_c3t(v) for v in _disp)+" \\\\")
t+=["\\bottomrule","\\end{tabular}","\\end{table}"]
w('tab-ladder.tex',"\n".join(t))

# ---- Tabel: skala template + biaya penggabungan ----
def _tpl_stats():
    """Mean/SD skor DAN AUROC, keduanya PER MODEL. Versi lama memakai statistik
    skor 7B saja di sebelah kolom AUROC dua model, sehingga pembaca mengira
    skala itu berlaku untuk keduanya -- padahal rerata cloze 7B +3,45 sementara
    32B -0,03."""
    out={}
    for m,src in (('7B',E['7B']),('32B',E['32B'])):
        gg=llm(src,False); per=[]; ms=[]
        for tt in TPL:
            h=gg[gg.template==tt]
            ms.append((h.score.mean(),h.score.std()))
            _,v=per_group(h.label.values,h.drugbank_id.values,h.score.values)
            per.append(np.nanmean(v))
        _,vp=per_group(gg.label.values,gg.drugbank_id.values,gg.score.values)
        out[m]=dict(ms=ms,per=per,pooled=float(np.nanmean(vp)))
    return out
_tp=_tpl_stats()
SPREAD={m:max(x[0] for x in _tp[m]['ms'])-min(x[0] for x in _tp[m]['ms']) for m in _tp}
t=["\\begin{table}[!tbp]","\\centering","\\small",
 "\\caption{Score scale of each prompt template and the cost of pooling them.}",
 "\\label{tab:template}","\\begin{tabular}{@{}lrrcrrc@{}}","\\toprule",
 "& \\multicolumn{3}{c}{OLMo-2 7B} & \\multicolumn{3}{c}{OLMo-2 32B} \\\\",
 "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}",
 "Template & Mean & SD & \\auroc{} & Mean & SD & \\auroc{} \\\\","\\midrule"]
for i,tt in enumerate(TPL):
    a,b=_tp['7B']['ms'][i], _tp['32B']['ms'][i]
    t.append(f"\\texttt{{{tt}}} & {a[0]:+.2f} & {a[1]:.2f} & {_tp['7B']['per'][i]:.3f} & "
             f"{b[0]:+.2f} & {b[1]:.2f} & {_tp['32B']['per'][i]:.3f} \\\\")
t.append("\\midrule")
t.append(f"Mean of the three & & & \\textbf{{{np.mean(_tp['7B']['per']):.3f}}} & & & "
         f"\\textbf{{{np.mean(_tp['32B']['per']):.3f}}} \\\\")
t.append(f"Pooled into one ranking & & & {_tp['7B']['pooled']:.3f} & & & "
         f"{_tp['32B']['pooled']:.3f} \\\\")
# Blok contoh: skor MENTAH satu pasangan, bukan agregat. Dipakai \multicolumn
# selebar blok tiap model supaya tidak bisa terbaca sebagai Mean/SD/AUROC.
# Diambil dari berkas skor, bukan diketik -- angka di prosa harus ada di tabel.
EX_DB,EX_CUI='DB00713','C0154640'
def _exrow(src):
    g=llm(src,False)
    h=g[(g.drugbank_id==EX_DB)&(g.umls_cui==EX_CUI)]
    return {r.template:r.score for r in h.itertuples()}
_e7,_e32=_exrow(E['7B']),_exrow(E['32B'])
assert set(_e7)>=set(TPL) and set(_e32)>=set(TPL), "pasangan contoh tidak lengkap"
t.append("\\midrule")
# Nama model DIULANG di sini: kepala rentang ada 8 baris di atas, terpisah dua
# \midrule, sehingga pembaca kehilangan jejak kolom mana milik model mana.
t.append("\\multicolumn{1}{@{}l}{\\textit{Raw score, oxacillin + staph. meningitis}}"
         " & \\multicolumn{3}{c}{OLMo-2 7B} & \\multicolumn{3}{c}{OLMo-2 32B} \\\\")
for tt in TPL:
    t.append(f"\\quad\\texttt{{{tt}}} & \\multicolumn{{3}}{{c}}{{${_e7[tt]:+.2f}$}} & "
             f"\\multicolumn{{3}}{{c}}{{${_e32[tt]:+.2f}$}} \\\\")
t+=["\\bottomrule","\\end{tabular}","\\end{table}"]
w('tab-template.tex',"\n".join(t))
print(f"  rentang mean antar-template: 7B {SPREAD['7B']:.2f}, 32B {SPREAD['32B']:.2f}")

# ---- Tabel: uji berpasangan antar kondisi tangga ----
# Angka ini dibahas di prosa 3.4; tanpa tabel ia menggantung tanpa rumah.
# Empat kolom berdampingan meluber 60 pt dari lebar teks, bahkan pada
# \footnotesize. Dua keluarga karena itu ditumpuk sebagai BLOK BARIS -- idiom
# yang sama dengan tab-paired.tex -- sehingga tabel kembali ke dua kolom angka
# dan selang kepercayaan penuh tetap terbaca pada \small.
_PB=["\\begin{table}[!tbp]","\\centering","\\small",
 "\\caption{Paired difference between the canonical name and each perturbation.}",
 "\\label{tab:ladderpaired}","\\begin{tabular}{@{}lcc@{}}","\\toprule",
 "Comparison & \\multicolumn{2}{c}{Checkpoint} \\\\","\\midrule"]
def _pb(v):
    s=rf"{v[0]:+.3f}\;[{v[1]:+.3f},\;{v[2]:+.3f}]"
    return "$\\mathbf{"+s+"}$" if (v[1]>0 or v[2]<0) else "$"+s+"$"
# P2 disertakan: tanpa baris itu tabel punya lubang yang terbaca sebagai
# kondisi yang dijalankan tapi tidak dilaporkan, dan P1-P2 adalah skala
# pembanding yang membuat +0,039 bisa dinilai besar atau kecil.
# Satu kepala kolom bersama tidak bisa menyebut ukuran, karena checkpoint kecil
# OLMo-2 adalah 7B sementara Qwen3 adalah 8B -- menuliskan salah satu akan salah
# untuk keluarga yang lain. Label generik "Smaller/Larger checkpoint" dipakai
# sebelumnya, tetapi pembaca harus memetakannya sendiri ke kepala blok. Baris
# kepala tiap blok karena itu SEKALIGUS menjadi kepala kolom blok tersebut:
# nama keluarga di kolom pertama, dua ukurannya tepat di atas angkanya sendiri.
for _fam,_sa,_sb,_a,_b in (("OLMo-2",'7B','32B','7B','32B'),
                           ("Qwen3",'8B','32B','q8','q32')):
    if _fam=="Qwen3": _PB.append("\\addlinespace")
    _PB.append(f"\\textit{{{_fam}}} & \\textit{{{_sa}}} & \\textit{{{_sb}}} \\\\")
    for c in ('A4','A5c','A5t','A5'):
        _PB.append(f"\\quad {PN['A0']} versus {PN[c]} & "
                   f"{_pb(PC[(_a,c)])} & {_pb(PC[(_b,c)])} \\\\")
_PB+=["\\bottomrule","\\end{tabular}","\\end{table}"]
w('tab-ladderpaired.tex',"\n".join(_PB))


t=["\\begin{table}[!tbp]","\\centering","\\small",
 "\\caption{The global \\auroc{} and its within-drug permutation floor.}",
 "\\label{tab:metric}","\\begin{tabular}{llccc}","\\toprule",
 "Model & Task & Global \\auroc{} & Permutation floor & \\textbf{Macro \\auroc{}} \\\\",
 "\\midrule"]
for m in MS:
    for i,(task,within) in enumerate((('class-level',False),('within-class',True))):
        fl,mac=FLOOR[m][i]
        # Kolom Macro ditebalkan: konvensi yang sama dengan tab-template, yaitu
        # tebal menandai nilai yang BENAR-BENAR dipakai studi ini. Kolom global
        # dan lantainya ada semata untuk menunjukkan apa yang ditolak.
        t.append(f"{EN[m]} & {task} & {glob_auroc(m,within):.3f} & {fl:.3f} & "
                 f"\\textbf{{{mac:.3f}}} \\\\")
    if m!=MS[-1]: t.append("\\addlinespace")   # pisahkan tiap model, 2 baris per model
t+=["\\bottomrule","\\end{tabular}","\\end{table}"]
w('tab-metric.tex',"\n".join(t))

# ============================================================================
# EKSPERIMEN A3 -- checkpoint base lawan Instruct pada tiga ukuran yang cocok.
#
# DILETAKKAN DI PALING AKHIR DENGAN SENGAJA. Bootstrap di bawah ini memakan
# undian dari `rng` GLOBAL. Kalau blok ini ditaruh lebih awal, setiap selang
# kepercayaan di seluruh tabel sebelumnya akan bergeser walau titik estimasinya
# tidak berubah -- dan setiap angka yang sudah diperiksa di prosa harus
# diperiksa ulang. Di posisi ini angka lama tetap identik bit-per-bit.
#
# Tiga ukuran, bukan empat: pasangan base/Instruct ada di 1B, 7B dan 13B.
# Pasangan base dari 32B tidak diperoleh (lihat Section Limitations); repo base
# OLMo-2 disimpan dalam fp32 sehingga 32B berukuran 120 GiB, dan unduhannya
# tidak selesai dalam anggaran komputasi yang tersedia.
# ============================================================================
EB={'1B':'results/easy_OLMo-2-0425-1B.jsonl',
    '7B':'results/easy_OLMo-2-1124-7B.jsonl',
    '13B':'results/easy_OLMo-2-1124-13B.jsonl'}
HB={'1B':'results/within_OLMo-2-0425-1B.jsonl',
    '7B':'results/within_OLMo-2-1124-7B.jsonl',
    '13B':'results/within_OLMo-2-1124-13B.jsonl'}
BMS=['1B','7B','13B']

def _mat_df(f,within):
    """Matriks AUROC (kelompok x template) sebagai DataFrame berindeks kelompok.

    Berkas lama menyimpan beberapa kondisi dalam satu berkas dan dibangun dari
    versi TSV yang lebih longgar, jadi pasangan yang tidak lagi ada di
    within_class_pairs.tsv dibuang di sini. Tanpa saringan itu GKEY melempar
    KeyError, dan yang lebih berbahaya: kelompoknya tidak akan sejajar antara
    base dan Instruct sehingga uji berpasangan membandingkan himpunan berbeda.
    """
    g=llm(f,within)
    if within:
        ok=[(d,c) in GKEY for d,c in zip(g.drugbank_id,g.umls_cui)]
        g=g[pd.Series(ok,index=g.index).values]
    cols={}
    for t_ in TPL:
        gg=g[g.template==t_]
        key=([GKEY[(d,c)] for d,c in zip(gg.drugbank_id,gg.umls_cui)]
             if within else gg.drugbank_id.values)
        ks,vs=per_group(gg.label.values,key,gg.score.values)
        cols[t_]=pd.Series(vs,index=ks)
    return pd.DataFrame(cols).dropna()

def _sum_df(M_):
    return summarize(M_.values)

def _paired_df2(a,b):
    """Selisih a-b pada KELOMPOK YANG SAMA, bootstrap berpasangan."""
    k=a.index.intersection(b.index)
    D=a.loc[k].values-b.loc[k].values; n=len(k)
    dr=np.array([D[rng.integers(0,n,n)].mean(axis=0).mean() for _ in range(B)])
    return D.mean(axis=0).mean(), np.percentile(dr,2.5), np.percentile(dr,97.5), n

MB={}; MI={}
for _m in BMS:
    for _t,_w in (('tingkat-kelas',False),('intra-kelas',True)):
        MB[(_t,_m)]=_mat_df((HB if _w else EB)[_m],_w)
        MI[(_t,_m)]=_mat_df((H  if _w else E )[_m],_w)
RB={k:_sum_df(v) for k,v in MB.items()}
RI={k:_sum_df(v) for k,v in MI.items()}
PB3={k:_paired_df2(MB[k],MI[k]) for k in MB}

# Kelompok harus sejajar antara base dan Instruct, kalau tidak tabelnya
# membandingkan himpunan obat yang berbeda tanpa kelihatan.
for k in MB:
    _n=len(MB[k].index.intersection(MI[k].index))
    assert _n==len(MB[k])==len(MI[k]), \
        f"kelompok tidak sejajar pada {k}: base={len(MB[k])} instruct={len(MI[k])} irisan={_n}"

EBN={'1B':'1B','7B':'7B','13B':'13B'}
t=["\\begin{table}[!tbp]","\\centering","\\small",
 "\\caption{Macro within-group \\auroc{} of the base and instruction-tuned checkpoints of "
 "OLMo-2 at three matched sizes.}",
 "\\label{tab:base}","\\begin{tabular}{@{}lccc@{}}","\\toprule",
 "Checkpoint & Class-level & Within-class & Gap \\\\","\\midrule"]
# Baris Instruct memakai R[] -- nilai YANG SAMA yang dipakai tab-main, bukan RI
# yang dihitung ulang di blok ini. Titik estimasinya identik, tetapi selangnya
# berbeda satu digit karena posisi undian rng berbeda, dan dua tabel yang
# memberi selang berlainan untuk checkpoint yang sama akan dibaca sebagai galat.
for _lab,_R in (("Base",RB),("Instruction-tuned",R)):
    if _lab=="Instruction-tuned": t.append("\\addlinespace")
    t.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\textit{{{_lab}}}}} \\\\")
    for _m in BMS:
        a=_R[('tingkat-kelas',_m)]; b=_R[('intra-kelas',_m)]
        t.append(f"\\quad OLMo-2 {EBN[_m]} & {c3(a)} & {c3(b)} & "
                 f"${a[0]-b[0]:.3f}$ \\\\")
t+=["\\bottomrule","\\end{tabular}","\\end{table}"]
w('tab-base.tex',"\n".join(t))

# Tabel berpasangan dipisah, bukan dijadikan kolom keempat: sel selisih bertanda
# dengan selang penuh selebar ~85 pt, dan tab-main sudah hampir penuh dengan
# tiga kolom. Idiomnya sama dengan tab-ladderpaired.
t=["\\begin{table}[!tbp]","\\centering","\\small",
 "\\caption{Paired difference between the base and the instruction-tuned "
 "checkpoints.}",
 "\\label{tab:basepaired}","\\begin{tabular}{@{}lcc@{}}","\\toprule",
 "Checkpoint & Class-level task & Within-class task \\\\","\\midrule"]
for _m in BMS:
    t.append(f"OLMo-2 {EBN[_m]} & {bf(PB3[('tingkat-kelas',_m)])} & "
             f"{bf(PB3[('intra-kelas',_m)])} \\\\")
t+=["\\bottomrule","\\end{tabular}","\\end{table}"]
w('tab-basepaired.tex',"\n".join(t))

print("\n=== A3: base lawan Instruct ===")
for _t in ('tingkat-kelas','intra-kelas'):
    print(f"-- {_t}")
    for _m in BMS:
        print(f"   {_m:>4}  base {c3(RB[(_t,_m)])}   instruct {c3(RI[(_t,_m)])}"
              f"   selisih {c3(PB3[(_t,_m)])}  n={PB3[(_t,_m)][3]}")
print("\njurang (kelas - intra):")
for _lab,_R in (("base",RB),("instruct",RI)):
    print(f"   {_lab:>8}  " + "  ".join(
        f"{_m}={_R[('tingkat-kelas',_m)][0]-_R[('intra-kelas',_m)][0]:.3f}" for _m in BMS))
print("\n6 tabel LaTeX ditulis ke paper/tables/")
