"""Analisis pilot: apakah kebenaran model berhubungan dgn frekuensi korpus?

Sesuai Temuan A, frekuensi diperlakukan ORDINAL (kuintil) sebagai analisis
primer; regresi kontinu sekunder -- karena hitungan CNF aproksimatif dan
parameter-sensitif di ujung atas.
"""
import csv, json, math, sys
import numpy as np, pandas as pd

sc=pd.DataFrame([json.loads(l) for l in open('results/pilot_scores.jsonl')])
fr=pd.read_csv('data/processed/pilot_corpus_freq.tsv', sep='\t')
print(f"skor: {len(sc)} baris | frekuensi: {len(fr)} pasangan\n")

key=['drugbank_id','umls_cui']
fr2=fr[key+['pair_count','drug_count','disease_count','pair_approx','used_fallback']]
d=sc.merge(fr2, on=key, how='left')
d['pair_count']=pd.to_numeric(d['pair_count'], errors='coerce')
d['correct']=((d['score']>0).astype(int)==d['label']).astype(int)

def auroc(y, s):
    y=np.asarray(y); s=np.asarray(s)
    if len(set(y))<2: return float('nan')
    r=pd.Series(s).rank().values
    n1=y.sum(); n0=len(y)-n1
    return (r[y==1].sum() - n1*(n1+1)/2) / (n1*n0)

print("="*64)
print("1. AUROC per kondisi x template")
print("="*64)
tab=[]
for (c,t),g in d.groupby(['condition','template']):
    tab.append({'kondisi':c,'template':t,'n':len(g),'AUROC':auroc(g['label'],g['score']),
                'akurasi':g['correct'].mean()})
T=pd.DataFrame(tab).pivot(index='template',columns='kondisi',values='AUROC')
print(T.round(3).to_string())
print("\nrata-rata lintas template:")
for c,g in d.groupby('condition'):
    print(f"  {c}: AUROC={auroc(g['label'],g['score']):.3f}  akurasi={g['correct'].mean():.3f}")
a0=auroc(d[d.condition=='A0']['label'], d[d.condition=='A0']['score'])
a4=auroc(d[d.condition=='A4']['label'], d[d.condition=='A4']['score'])
print(f"\n  >> delta AUROC (A0 - A4) = {a0-a4:+.3f}   <- sinyal memorizasi leksikal")

print("\n"+"="*64)
print("2. PERTANYAAN INTI: kebenaran vs frekuensi korpus (kuintil)")
print("="*64)
a=d[(d.condition=='A0') & d.pair_count.notna()].copy()
if len(a)==0:
    print("  belum ada data frekuensi"); sys.exit()
a['q']=pd.qcut(a['pair_count'].rank(method='first'), 5, labels=['Q1 (jarang)','Q2','Q3','Q4','Q5 (sering)'])
g=a.groupby('q', observed=True).agg(n=('correct','size'), akurasi=('correct','mean'),
                                     median_freq=('pair_count','median'))
g['AUROC']=[auroc(a[a.q==q]['label'], a[a.q==q]['score']) for q in g.index]
print(g.round(3).to_string())
lo=g['akurasi'].iloc[0]; hi=g['akurasi'].iloc[-1]
print(f"\n  >> gradien akurasi Q1->Q5 = {hi-lo:+.3f}")
print("     (positif & monoton = bukti memorisasi; datar = tidak ada sinyal)")

print("\n"+"="*64)
print("3. Regresi logistik (sekunder, kontinu)")
print("="*64)
try:
    import statsmodels.formula.api as smf
    a['logf']=np.log1p(a['pair_count']); a['logd']=np.log1p(a['drug_degree'])
    a['logs']=np.log1p(a['disease_degree'])
    m=smf.logit('correct ~ logf + logd + logs', data=a).fit(disp=0)
    print(m.summary2().tables[1].round(4).to_string())
    p=m.pvalues.get('logf', float('nan'))
    print(f"\n  >> koefisien logf = {m.params.get('logf',float('nan')):+.4f}  p = {p:.4f}")
except Exception as e:
    print("  regresi gagal:", e)

print("\n"+"="*64)
print("4. Baseline derajat (apakah LLM mengalahkan tebakan popularitas?)")
print("="*64)
base=auroc(a['label'], a['drug_degree'])
print(f"  AUROC baseline derajat obat : {base:.3f}")
print(f"  AUROC LLM (A0)              : {a0:.3f}")
print(f"  AUROC LLM (A4)              : {a4:.3f}")
print(f"\n  >> LLM A0 unggul {a0-base:+.3f} atas baseline; A4 unggul {a4-base:+.3f}")
if a4 <= base:
    print("     CATATAN: dgn identitas disamarkan, LLM TIDAK mengalahkan baseline sepele.")
