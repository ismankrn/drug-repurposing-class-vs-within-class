"""Negatif DALAM-KELAS: uji resolusi spesifik-obat.

Untuk tiap positif (obat A, penyakit X), cari obat B di kelas EPC yg SAMA yg
TIDAK diindikasikan utk X (dan bukan kontraindikasi). Model harus memilih
anggota kelas yg benar -- inilah yg sebenarnya dituntut repurposing.

Catatan konstruksi: A5c (kelas saja) pd pasangan ini IDENTIK antara positif
dan negatif -> AUROC-nya WAJIB ~0,5. Itu pemeriksaan kebenaran, bukan temuan.
Uji sesungguhnya ada di A0 (nama asli).
"""
import sys, os, csv, collections, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pgcopy import read_copy
DC='data/interim/drugcentral/'; OUT='data/processed'
rng=random.Random(20261102)

nv={r['drugbank_id']:r for r in csv.DictReader(open(f'{OUT}/name_variants.tsv'), delimiter='\t')}
_, ident=read_copy(DC+'identifier.sql')
sid2db=collections.defaultdict(set)
for r in ident:
    if r['id_type']=='DRUGBANK_ID': sid2db[r['struct_id']].add(r['identifier'])
_, pc=read_copy(DC+'pharma_class.sql')
epc=collections.defaultdict(set); cls=collections.defaultdict(set)
for r in pc:
    if r['type']=='EPC' and r.get('name') and r.get('struct_id'):
        epc[r['struct_id']].add(r['name']); cls[r['name']].add(r['struct_id'])
_, omop=read_copy(DC+'omop_relationship.sql')
ind=[r for r in omop if r['relationship_name']=='indication' and r['umls_cui']]
ind_set={(r['struct_id'],r['umls_cui']) for r in ind}
con_set={(r['struct_id'],r['umls_cui']) for r in omop
         if r['relationship_name']=='contraindication' and r['umls_cui']}
cname={r['umls_cui']:(r.get('concept_name') or '') for r in ind if r.get('concept_name')}
dd=collections.Counter(r['struct_id'] for r in ind); sd=collections.Counter(r['umls_cui'] for r in ind)

def nm(sid):
    for db in sid2db.get(sid,()):
        a0=nv.get(db,{}).get('A0_canonical')
        if a0: return db,a0
    return None,None

rows=[]; rng.shuffle(ind)
used=set()
for r in ind:
    sid, cui = r['struct_id'], r['umls_cui']
    cn=cname.get(cui)
    dbA,a0A=nm(sid)
    if not cn or not a0A: continue
    # kandidat obat lain di kelas EPC yg sama
    sib=set()
    for c in epc.get(sid,()):
        if 2 <= len(cls[c]) <= 60: sib |= cls[c]
    sib.discard(sid)
    cand=[s for s in sib if (s,cui) not in ind_set and (s,cui) not in con_set and nm(s)[1]]
    if not cand: continue
    # PENCOCOKAN DERAJAT: pilih saudara sekelas dgn jumlah indikasi paling mirip.
    # Tanpa ini derajat obat bocor (terukur 0,730) -- model cukup memilih obat
    # yg lebih "terkenal", bukan yg benar utk penyakit itu.
    tgt=dd[sid]
    cand.sort(key=lambda s_: abs(dd[s_]-tgt))
    near=[s_ for s_ in cand if abs(dd[s_]-tgt) <= max(2, 0.25*tgt)]
    if not near: continue
    sB=rng.choice(near[:5]); dbB,a0B=nm(sB)
    key=(dbA,cui)
    if key in used: continue
    used.add(key)
    shared=sorted(set(epc.get(sid,())) & set(epc.get(sB,())))[:1]
    for db,a0,s_,lab in ((dbA,a0A,sid,1),(dbB,a0B,sB,0)):
        rows.append({'drugbank_id':db,'struct_id':s_,'umls_cui':cui,'A0':a0,
                     'disease':cn,'label':lab,'tier':'within_class',
                     'shared_class':shared[0] if shared else '',
                     'drug_degree':dd[s_],'disease_degree':sd[cui]})
    if len(rows)>=3000: break

# pseudonim A4
CONS='BCDFGHJKLMNPQRSTVWXZ'; pseudo={}; seen=set()
for d in sorted({r['drugbank_id'] for r in rows}):
    while True:
        p=''.join(rng.choice(CONS) for _ in range(4))
        if p not in seen: break
    seen.add(p); pseudo[d]=f"{p}-{rng.randint(100,999)}"
for r in rows: r['A4']=pseudo[r['drugbank_id']]

def auroc(y,s):
    pair=sorted(zip(s,y)); rk=[0]*len(pair); i=0
    while i<len(pair):
        j=i
        while j+1<len(pair) and pair[j+1][0]==pair[i][0]: j+=1
        for k in range(i,j+1): rk[k]=(i+j)/2+1
        i=j+1
    n1=sum(y); n0=len(y)-n1
    return (sum(rk[k] for k in range(len(pair)) if pair[k][1]==1)-n1*(n1+1)/2)/(n1*n0)
y=[r['label'] for r in rows]
print(f"pasangan dalam-kelas: {len(rows)} ({sum(y)} positif / {len(y)-sum(y)} negatif)")
print(f"  penyakit unik : {len({r['umls_cui'] for r in rows})}")
print(f"  obat unik     : {len({r['drugbank_id'] for r in rows})}")
print(f"  kelas unik    : {len({r['shared_class'] for r in rows if r['shared_class']})}")
print(f"\n  baseline derajat obat     = {auroc(y,[r['drug_degree'] for r in rows]):.3f}")
print(f"  baseline derajat penyakit = {auroc(y,[r['disease_degree'] for r in rows]):.3f}")
with open(f'{OUT}/within_class_pairs.tsv','w',newline='') as fh:
    w=csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter='\t'); w.writeheader(); w.writerows(rows)
print(f"\nditulis: {OUT}/within_class_pairs.tsv")
print("\ncontoh 4 (positif vs negatif sekelas, penyakit sama):")
for i in range(0,8,2):
    a,b=rows[i],rows[i+1]
    print(f"  [{a['shared_class'][:34]:<34}] {a['disease'][:30]}")
    print(f"      + {a['A0'][:26]:<26}   - {b['A0'][:26]}")
