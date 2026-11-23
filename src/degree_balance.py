"""Penyeimbangan derajat untuk set pasangan v2.

Temuan V menunjukkan baseline derajat mencapai AUROC 0,716/0,809 -- itu ARTEFAK
sampling, bukan hasil: positif diambil dari indikasi dan negatif dari
kontraindikasi tanpa mencocokkan distribusi derajat, sehingga derajat
membocorkan label lewat konstruksi set.

Perbaikan: pencocokan 1:1 pada (kuintil derajat obat, kuintil derajat penyakit).
Keluaran: pilot_pairs_v3.tsv + laporan keseimbangan sebelum/sesudah.
"""
import sys, os, csv, collections, random
OUT='data/processed'; rng=random.Random(20261101)

rows=list(csv.DictReader(open(f'{OUT}/pilot_pairs_v2.tsv'), delimiter='\t'))
for r in rows:
    r['drug_degree']=int(r['drug_degree']); r['disease_degree']=int(r['disease_degree'])

def quint(vals):
    s=sorted(vals); n=len(s)
    return [s[int(n*f)] for f in (0.2,0.4,0.6,0.8)]

qd=quint([r['drug_degree'] for r in rows])
qs=quint([r['disease_degree'] for r in rows])
def b(v,q): return sum(1 for x in q if v>x)
for r in rows: r['stratum']=(b(r['drug_degree'],qd), b(r['disease_degree'],qs))

pos=[r for r in rows if r['tier']=='pos']
print(f"kolam: pos={len(pos)}", end='')
for t in ('neg_contra','neg_random'):
    print(f"  {t}={len([r for r in rows if r['tier']==t])}", end='')
print()

def auroc(y,s):
    import statistics
    pair=sorted(zip(s,y)); ranks={}
    # rank rata-rata utk ikatan
    i=0; rk=[0]*len(pair)
    while i<len(pair):
        j=i
        while j+1<len(pair) and pair[j+1][0]==pair[i][0]: j+=1
        avg=(i+j)/2+1
        for k in range(i,j+1): rk[k]=avg
        i=j+1
    n1=sum(y); n0=len(y)-n1
    s1=sum(rk[k] for k in range(len(pair)) if pair[k][1]==1)
    return (s1-n1*(n1+1)/2)/(n1*n0) if n1 and n0 else float('nan')

def report(label, P, N):
    y=[1]*len(P)+[0]*len(N)
    dd=[r['drug_degree'] for r in P+N]; sd=[r['disease_degree'] for r in P+N]
    print(f"  {label:<22} n={len(P)}+{len(N)}  AUROC_derajat_obat={auroc(y,dd):.3f}  AUROC_derajat_penyakit={auroc(y,sd):.3f}")

print("\n=== SEBELUM penyeimbangan ===")
for t in ('neg_contra','neg_random'):
    report(f"pos vs {t}", pos, [r for r in rows if r['tier']==t])

# pencocokan 1:1 per stratum
out=[]
for t in ('neg_contra','neg_random'):
    N=[r for r in rows if r['tier']==t]
    byst=collections.defaultdict(list)
    for r in N: byst[r['stratum']].append(r)
    for v in byst.values(): rng.shuffle(v)
    mP, mN = [], []
    for p in pos:
        pool=byst.get(p['stratum'])
        if pool:
            mN.append(pool.pop()); mP.append(p)
    print(f"\n=== SESUDAH penyeimbangan ({t}) ===")
    report(f"pos vs {t}", mP, mN)
    print(f"  tercocok: {len(mP)}/{len(pos)} positif ({100*len(mP)/len(pos):.0f}%)")
    out.append((t,mP,mN))

# tulis v3: gabungan pasangan tercocok
seen=set(); final=[]
for t,mP,mN in out:
    for r in mP+mN:
        k=(r['drugbank_id'],r['umls_cui'],r['tier'])
        if k not in seen: seen.add(k); final.append(r)
flds=[k for k in rows[0].keys() if k!='stratum']
with open(f'{OUT}/pilot_pairs_v3.tsv','w',newline='') as fh:
    w=csv.DictWriter(fh, fieldnames=flds, delimiter='\t', extrasaction='ignore')
    w.writeheader(); w.writerows(final)
print(f"\nditulis: {OUT}/pilot_pairs_v3.tsv ({len(final)} baris)")
for t in ('pos','neg_contra','neg_random'):
    print(f"  {t:<12} {len([r for r in final if r['tier']==t])}")
