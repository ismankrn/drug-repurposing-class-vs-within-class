"""Set pilot v2: negatif KONTRAINDIKASI + negatif acak tercocok-derajat.

Perbaikan atas v1 (Temuan V): negatif kegagalan Phase 3 tidak dapat dibedakan
secara farmakologis (hanya 2,3% terkait efikasi, Temuan C). Kontraindikasi
adalah pasangan yg memang SEHARUSNYA ditolak -- sinyal farmakologis nyata.

Tiga tier dalam satu set agar bisa dibandingkan dalam satu run:
  pos        = indikasi disetujui
  neg_contra = kontraindikasi (26.851 tersedia)
  neg_random = pasangan acak, distribusi derajat dicocokkan
"""
import sys, os, csv, collections, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pgcopy import read_copy
DC='data/interim/drugcentral/'; OUT='data/processed'
rng=random.Random(20261101)

nv={r['drugbank_id']:r for r in csv.DictReader(open(f'{OUT}/name_variants.tsv'), delimiter='\t')}
_, ident=read_copy(DC+'identifier.sql')
sid2db=collections.defaultdict(set); db2sid=collections.defaultdict(set)
for r in ident:
    if r['id_type']=='DRUGBANK_ID':
        db2sid[r['identifier']].add(r['struct_id']); sid2db[r['struct_id']].add(r['identifier'])

_, omop=read_copy(DC+'omop_relationship.sql')
ind=[r for r in omop if r['relationship_name']=='indication' and r['umls_cui']]
con=[r for r in omop if r['relationship_name']=='contraindication' and r['umls_cui']]
print(f"indikasi={len(ind)}  kontraindikasi={len(con)}")

def nm(sid):
    for db in sid2db.get(sid,()):
        a0=nv.get(db,{}).get('A0_canonical')
        if a0: return db,a0
    return None,None

ind_set={(r['struct_id'],r['umls_cui']) for r in ind}
dd=collections.Counter(r['struct_id'] for r in ind)
sd=collections.Counter(r['umls_cui'] for r in ind)

def mk(rows, label, tier, n):
    out=[]; rng.shuffle(rows)
    for r in rows:
        db,a0=nm(r['struct_id'])
        if not a0: continue
        cn=r.get('concept_name') or r.get('snomed_full_name')
        if not cn: continue
        out.append({'drugbank_id':db,'struct_id':r['struct_id'],'umls_cui':r['umls_cui'],
                    'A0':a0,'disease':cn,'alt_disease':r.get('snomed_full_name') or cn,
                    'label':label,'tier':tier,
                    'drug_degree':dd[r['struct_id']],'disease_degree':sd[r['umls_cui']]})
        if len(out)>=n: break
    return out

P   = mk(list(ind), 1, 'pos', 100)
NC  = mk([r for r in con if (r['struct_id'],r['umls_cui']) not in ind_set], 0, 'neg_contra', 100)

# negatif acak tercocok-derajat
sids=[r['struct_id'] for r in ind]; cuis=[r['umls_cui'] for r in ind]
cname={r['umls_cui']:(r.get('concept_name') or '') for r in ind}
con_set={(r['struct_id'],r['umls_cui']) for r in con}
NR=[]
while len(NR)<100:
    s,c = rng.choice(sids), rng.choice(cuis)
    if (s,c) in ind_set or (s,c) in con_set: continue
    db,a0=nm(s)
    if not a0 or not cname.get(c): continue
    NR.append({'drugbank_id':db,'struct_id':s,'umls_cui':c,'A0':a0,'disease':cname[c],
               'alt_disease':cname[c],'label':0,'tier':'neg_random',
               'drug_degree':dd[s],'disease_degree':sd[c]})

rows=P+NC+NR
# pseudonim A4 per obat
BANNED={'IMMU','INCB','LOXO','MEDI','ASTX','BMS','MOR','TRV','INC','PM','UX','DCC','NS','ABT','GSK','AZD','MK','BI','LY','PF'}
CONS='BCDFGHJKLMNPQRSTVWXZ'; pseudo={}; seen=set()
for d in sorted({r['drugbank_id'] for r in rows}):
    while True:
        p=''.join(rng.choice(CONS) for _ in range(4))
        if p not in BANNED and p not in seen: break
    seen.add(p); pseudo[d]=f"{p}-{rng.randint(100,999)}"
for r in rows: r['A4']=pseudo[r['drugbank_id']]
rng.shuffle(rows)

with open(f'{OUT}/pilot_pairs_v2.tsv','w',newline='') as fh:
    w=csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter='\t'); w.writeheader(); w.writerows(rows)
print(f"\nditulis: {OUT}/pilot_pairs_v2.tsv ({len(rows)})")
for t in ('pos','neg_contra','neg_random'):
    g=[r for r in rows if r['tier']==t]
    print(f"  {t:<12} n={len(g)}  obat unik={len({r['drugbank_id'] for r in g})}")
print("\ncontoh:")
for t in ('pos','neg_contra','neg_random'):
    r=next(x for x in rows if x['tier']==t)
    print(f"  [{t:<10}] {r['A0'][:22]:<22} / {r['disease'][:42]}")
