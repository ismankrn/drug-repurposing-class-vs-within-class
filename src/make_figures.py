"""Tiga figur utama paper. CI dari bootstrap berkluster (klaster = obat).

Palet: slot kategorikal tervalidasi dari referensi dataviz, urutan tetap
(1 biru, 2 jingga, 3 aqua). Tidak ada sumbu ganda. AUROC TIDAK dibatang dari
nol -- baseline-nya 0,5, jadi dipakai titik+whisker dengan garis acuan 0,5.
"""
import json, numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.ticker import FuncFormatter

koma = FuncFormatter(lambda v,_: f"{v:.2f}".replace('.',','))
def k3(v): return f"{v:.3f}".replace('.',',')

S1,S2,S3 = "#2a78d6","#eb6834","#1baf7a"
SURF,GRID,AXIS = "#fcfcfb","#e1e0d9","#c3c2b7"
INK,SEC,MUT = "#0b0b0b","#52514e","#898781"
rng = np.random.default_rng(20261102); B = 2000

mpl.rcParams.update({
    "figure.facecolor":SURF,"axes.facecolor":SURF,"savefig.facecolor":SURF,
    "axes.edgecolor":AXIS,"axes.labelcolor":SEC,"text.color":INK,
    "xtick.color":MUT,"ytick.color":MUT,"grid.color":GRID,"grid.linewidth":0.8,
    "axes.linewidth":0.8,"font.size":10,"axes.titlesize":12,
    "axes.spines.top":False,"axes.spines.right":False,
})

def auroc(y,s):
    y=np.asarray(y); s=np.asarray(s); n1=y.sum(); n0=len(y)-n1
    if n1==0 or n0==0: return np.nan
    r=pd.Series(s).rank().values
    return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)

def boot(df, col='drugbank_id', score='score'):
    g=defaultdict(list)
    for i,c in enumerate(df[col].values): g[c].append(i)
    keys=list(g); idx={k:np.array(v) for k,v in g.items()}
    y=df['label'].values; s=df[score].values; out=[]
    for _ in range(B):
        pick=rng.choice(len(keys),len(keys),replace=True)
        sel=np.concatenate([idx[keys[p]] for p in pick])
        a=auroc(y[sel],s[sel])
        if not np.isnan(a): out.append(a)
    o=np.array(out); return auroc(y,s), np.percentile(o,2.5), np.percentile(o,97.5)

def load(f, within, cond='A0'):
    d=pd.DataFrame([json.loads(l) for l in open(f)])
    if within: g=d[d.condition==cond]
    else:
        g=pd.concat([d[d.tier=='pos'],d[d.tier=='neg_random']]); g=g[g.condition==cond]
    return g.reset_index(drop=True)

E={'1B':'results/easy_OLMo-2-0425-1B.jsonl','7B':'results/scores_5cond.jsonl',
   '13B':'results/easy_OLMo-2-1124-13B-Instruct.jsonl','32B':'results/easy_OLMo-2-0325-32B-Instruct.jsonl'}
H={'1B':'results/within_OLMo-2-0425-1B.jsonl','7B':'results/scores_within.jsonl',
   '13B':'results/within_OLMo-2-1124-13B-Instruct.jsonl','32B':'results/within_OLMo-2-0325-32B-Instruct.jsonl'}
P={'1B':1.0,'7B':7.3,'13B':13.7,'32B':32.2}

# ---------------- F1: dua kurva ----------------
cls=[boot(load(E[m],False)) for m in P]; itr=[boot(load(H[m],True)) for m in P]
x=[P[m] for m in P]
fig,ax=plt.subplots(figsize=(6.6,4.4))
ax.grid(axis='y',zorder=0)
ax.axhline(0.5,ls=(0,(4,3)),lw=1.2,color=MUT,zorder=1)
ax.text(4.5,0.494,"kebetulan (0,50)",color=MUT,fontsize=8.5,va='top',ha='center')
for (dat,c,lab) in ((cls,S1,"Tugas tingkat-kelas"),(itr,S2,"Tugas intra-kelas")):
    a=[d[0] for d in dat]; lo=[d[1] for d in dat]; hi=[d[2] for d in dat]
    ax.fill_between(x,lo,hi,color=c,alpha=0.16,lw=0,zorder=2)
    ax.plot(x,a,color=c,lw=2,marker='o',ms=8,mec=SURF,mew=2,zorder=3,label=lab)
    ax.annotate(lab,(x[-1],a[-1]),xytext=(8,0),textcoords='offset points',
                color=c,fontsize=10,va='center',fontweight='medium')
for xi,d in zip(x,cls): ax.annotate(k3(d[0]),(xi,d[0]),xytext=(0,11),textcoords='offset points',ha='center',fontsize=8.5,color=SEC)
for xi,d in zip(x,itr): ax.annotate(k3(d[0]),(xi,d[0]),xytext=(0,-17),textcoords='offset points',ha='center',fontsize=8.5,color=SEC)
ax.set_xscale('log'); ax.set_xticks(x); ax.set_xticklabels(list(P))
ax.set_xlim(0.85,52); ax.set_ylim(0.44,0.96); ax.yaxis.set_major_formatter(koma)
ax.set_xlabel("Ukuran model (miliar parameter, skala log)")
ax.set_ylabel("AUROC")
ax.set_title("Diskriminasi tingkat-kelas tumbuh jauh lebih cepat\ndaripada resolusi tingkat-obat",
             loc='left',color=INK,pad=12)
ax.legend(frameon=False,loc='upper left',fontsize=9.5,labelcolor=SEC)
fig.text(0.01,-0.02,"Pita = CI 95% bootstrap berkluster (klaster: obat, B=2.000). OLMo-2.",
         fontsize=8,color=MUT)
fig.tight_layout(); fig.savefig("figures/F1_dua_kurva.png",dpi=200,bbox_inches='tight')
print("F1 ditulis")

# ---------------- F2: tangga perturbasi 7B ----------------
conds=[('A0','Nama obat asli'),('A4','Pseudonim acak'),
       ('A5c','Kelas farmakologis'),('A5t','Kelas + target'),('A5','Kelas + target + ATC')]
res=[boot(load(E['7B'],False,cond=c)) for c,_ in conds]
fig,ax=plt.subplots(figsize=(6.8,3.8))
ypos=np.arange(len(conds))[::-1]
ax.grid(axis='x',zorder=0)
ax.axvline(0.5,ls=(0,(4,3)),lw=1.2,color=MUT,zorder=1)
for yp,(a,lo,hi),(code,lab) in zip(ypos,res,conds):
    col = S2 if code=='A4' else (S1 if code=='A0' else S3)
    ax.plot([lo,hi],[yp,yp],color=col,lw=2,solid_capstyle='round',zorder=2)
    ax.plot([a],[yp],'o',ms=9,color=col,mec=SURF,mew=2,zorder=3)
    ax.annotate(k3(a),(a,yp),xytext=(0,10),textcoords='offset points',
                ha='center',fontsize=9,color=SEC)
ax.set_yticks(ypos); ax.set_yticklabels([l for _,l in conds],color=SEC)
ax.text(0.5,len(conds)-0.42,"kebetulan",color=MUT,fontsize=8.5,ha='center')
ax.set_xlim(0.44,0.82); ax.set_ylim(-0.6,len(conds)-0.25); ax.xaxis.set_major_formatter(koma)
ax.set_xlabel("AUROC (tugas tingkat-kelas)")
ax.set_title("Deskripsi kelas farmakologis menyamai — bahkan melampaui —\nnama obat asli; pseudonim tanpa informasi jatuh ke kebetulan",
             loc='left',color=INK,pad=12)
fig.text(0.01,-0.06,"OLMo-2 7B-Instruct. Garis = CI 95% bootstrap berkluster (klaster: obat, B=2.000).\n"
         "Hijau = tiga kondisi yang mengganti nama obat dengan deskripsi mekanisme saja.",
         fontsize=8,color=MUT)
fig.tight_layout(); fig.savefig("figures/F2_tangga_perturbasi.png",dpi=200,bbox_inches='tight')
print("F2 ditulis")

# ---------------- F3: stratifikasi identifiabilitas ----------------
d=pd.DataFrame([json.loads(l) for l in open(E['7B'])])
sub=pd.concat([d[d.tier=='pos'],d[d.tier=='neg_random']])
SZ=[('A5c','coarse_class_size',S3,'A5c  kelas saja'),
    ('A5t','a5t_class_size',S2,'A5t  kelas + target'),
    ('A5','sig_class_size',S1,'A5   kelas+target+ATC')]
bins=[(1,1,'unik\n(n=1)'),(2,3,'2–3'),(4,10,'4–10'),(11,10**6,'≥11\n(tak teridentifikasi)')]
fig,ax=plt.subplots(figsize=(7.0,4.4))
ax.grid(axis='y',zorder=0)
ax.axhline(0.5,ls=(0,(4,3)),lw=1.2,color=MUT,zorder=1)
ax.text(2.95,0.505,"kebetulan (0,50)",color=MUT,fontsize=8.5,va='bottom',ha='right')
xs=np.arange(len(bins),dtype=float)
off={'A5c':-0.045,'A5t':0.0,'A5':0.045}
for code,fld,col,lab in SZ:
    g=sub[(sub.condition==code)&(sub[fld]>0)]
    a=[];lo=[];hi=[]
    for l_,h_,_ in bins:
        gg=g[(g[fld]>=l_)&(g[fld]<=h_)]
        if len(gg)>30:
            m,p1,p2=boot(gg)
        else: m=p1=p2=np.nan
        a.append(m);lo.append(p1);hi.append(p2)
    xo=xs+off[code]
    for xi,l_,h_ in zip(xo,lo,hi):
        ax.plot([xi,xi],[l_,h_],color=col,lw=1.6,alpha=0.75,solid_capstyle='round',zorder=2)
    ax.plot(xo,a,color=col,lw=2,marker='o',ms=8,mec=SURF,mew=2,zorder=3,label=lab)
    ax.annotate(code,(xo[-1],a[-1]),xytext=(10,0),textcoords='offset points',
                color=col,fontsize=9.5,va='center')
ax.set_xticks(xs); ax.set_xticklabels([b[2] for b in bins],color=SEC,fontsize=9)
ax.set_xlim(-0.35,len(bins)-0.5); ax.set_ylim(0.46,0.92)
ax.yaxis.set_major_formatter(koma)
ax.set_xlabel("Jumlah obat yang berbagi deskripsi yang sama")
ax.set_ylabel("AUROC")
ax.set_title("Performa tidak turun saat obat makin sulit diidentifikasi —\nuntuk deskripsi kelas saja (A5c) justru naik",
             loc='left',color=INK,pad=12)
ax.legend(frameon=False,loc='upper left',fontsize=9,labelcolor=SEC)
fig.text(0.01,-0.055,
  "OLMo-2 7B-Instruct, tugas tingkat-kelas. Makin ke kanan = obat makin sulit diidentifikasi dari deskripsinya.\n"
  "Whisker = CI 95% bootstrap berkluster. Strata terkanan A5 adalah yang terkecil (n=522 pasangan).",
  fontsize=8,color=MUT)
fig.tight_layout(); fig.savefig("figures/F3_identifiabilitas.png",dpi=200,bbox_inches='tight')
print("F3 ditulis")
