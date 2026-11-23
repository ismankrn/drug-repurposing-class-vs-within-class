"""Bangun set pasangan pilot (~200) dgn sebaran frekuensi yg lebar.

Stratifikasi berdasarkan DERAJAT OBAT (proksi popularitas) supaya pilot
mencakup obat umum maupun langka -- penting krn yg diuji adalah GRADIEN
frekuensi-kebenaran. Tanpa sebaran, gradien tak bisa terdeteksi.

Negatif: pakai tier N2 (kegagalan Phase 3), bukan seluruh kegagalan
(lihat Temuan C: hanya 2,3% kegagalan terkait efikasi).
"""
import sys, os, csv, collections, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pgcopy import read_copy
DC='data/interim/drugcentral/'; OUT='data/processed'
rng=random.Random(20261031)

with open('data/raw/repodb/full.csv',newline='',encoding='utf-8-sig') as fh:
    repodb=list(csv.DictReader(fh))

# nama kanonik A0 + sinonim dari name_variants
nv={r['drugbank_id']:r for r in csv.DictReader(open(f'{OUT}/name_variants.tsv'), delimiter='\t')}

# derajat
dd=collections.Counter(r['drug_id'] for r in repodb)
sd=collections.Counter(r['ind_id'] for r in repodb)

# buang 76 false negative (Temuan I) dan pasangan status bertentangan
_, ident=read_copy(DC+'identifier.sql')
db2sid=collections.defaultdict(set)
for r in ident:
    if r['id_type']=='DRUGBANK_ID': db2sid[r['identifier']].add(r['struct_id'])
_, omop=read_copy(DC+'omop_relationship.sql')
dc_ind={(s,r['umls_cui']) for r in omop if r['relationship_name']=='indication' and r['umls_cui']
        for s in [r['struct_id']]}
def is_dc_ind(r): return bool({(s,r['ind_id']) for s in db2sid.get(r['drug_id'],())} & dc_ind)

bystat=collections.defaultdict(set)
for r in repodb: bystat[(r['drug_id'],r['ind_id'])].add(r['status'])
conflict={k for k,v in bystat.items() if 'Approved' in v and len(v)>1}

pos=[r for r in repodb if r['status']=='Approved'
     and (r['drug_id'],r['ind_id']) not in conflict
     and nv.get(r['drug_id'],{}).get('A0_canonical')]
neg=[r for r in repodb if r['status']!='Approved' and r['phase']=='Phase 3'
     and not is_dc_ind(r)                                   # buang false negative
     and (r['drug_id'],r['ind_id']) not in conflict
     and nv.get(r['drug_id'],{}).get('A0_canonical')]
print(f"kolam positif (Approved, bersih)      : {len(pos)}")
print(f"kolam negatif (Phase 3 gagal, bersih) : {len(neg)}")

def stratify(pool, n):
    """Bagi ke 4 kuartil derajat obat, ambil merata."""
    degs=sorted({dd[r['drug_id']] for r in pool})
    q=[degs[int(len(degs)*f)] for f in (0.25,0.5,0.75)]
    buckets=collections.defaultdict(list)
    for r in pool:
        d=dd[r['drug_id']]
        b=0 if d<=q[0] else 1 if d<=q[1] else 2 if d<=q[2] else 3
        buckets[b].append(r)
    out=[]
    per=n//4
    for b in range(4):
        rng.shuffle(buckets[b]); out+=buckets[b][:per]
    return out, q

P,qp = stratify(pos,100); N,qn = stratify(neg,100)
print(f"ambang kuartil derajat (pos): {qp}")
print(f"terpilih: {len(P)} positif + {len(N)} negatif = {len(P)+len(N)}")

# pseudonim A4 terverifikasi (resep Temuan O); tetapkan per OBAT agar konsisten
BANNED={'IMMU','INCB','LOXO','MEDI','ASTX','BMS','MOR','TRV','INC','PM','UX','DCC','NS','ABT','GSK','AZD','MK','BI','LY','PF'}
CONS='BCDFGHJKLMNPQRSTVWXZ'
drugs=sorted({r['drug_id'] for r in P+N})
pseudo={}; seen=set()
for d in drugs:
    while True:
        p=''.join(rng.choice(CONS) for _ in range(4))
        if p not in BANNED and p not in seen: break
    seen.add(p); pseudo[d]=f"{p}-{rng.randint(100,999)}"

rows=[]
for r,lab in [(x,1) for x in P]+[(x,0) for x in N]:
    a0=nv[r['drug_id']]['A0_canonical']
    rows.append({'drugbank_id':r['drug_id'],'umls_cui':r['ind_id'],
                 'A0':a0,'A4':pseudo[r['drug_id']],'disease':r['ind_name'],
                 'label':lab,'status':r['status'],'phase':r['phase'],
                 'drug_degree':dd[r['drug_id']],'disease_degree':sd[r['ind_id']]})
rng.shuffle(rows)
with open(f'{OUT}/pilot_pairs.tsv','w',newline='') as fh:
    w=csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter='\t')
    w.writeheader(); w.writerows(rows)
print(f"\nditulis: {OUT}/pilot_pairs.tsv ({len(rows)} baris)")
print(f"  positif={sum(r['label'] for r in rows)}  negatif={sum(1-r['label'] for r in rows)}")
print(f"  obat unik={len({r['drugbank_id'] for r in rows})}  penyakit unik={len({r['umls_cui'] for r in rows})}")
dg=sorted(r['drug_degree'] for r in rows)
print(f"  derajat obat: min={dg[0]} median={dg[len(dg)//2]} max={dg[-1]}")
print("\ncontoh 5:")
for r in rows[:5]:
    print(f"  [{r['label']}] A0={r['A0'][:24]:<24} A4={r['A4']:<10} | {r['disease'][:40]}")
