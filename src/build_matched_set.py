"""Bangun set tercocok-derajat dari kolam besar (gabungan v2 + penyeimbangan).

Kolam besar -> pencocokan 1:1 per (kuintil derajat obat, kuintil derajat penyakit).
Target: >=300 pasangan tercocok per tier.
"""
import sys, os, csv, collections, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pgcopy import read_copy
DC='data/interim/drugcentral/'; OUT='data/processed'
rng=random.Random(20261101)
POOL=int(sys.argv[1]) if len(sys.argv)>1 else 1500

nv={r['drugbank_id']:r for r in csv.DictReader(open(f'{OUT}/name_variants.tsv'), delimiter='\t')}
_, ident=read_copy(DC+'identifier.sql')
sid2db=collections.defaultdict(set)
for r in ident:
    if r['id_type']=='DRUGBANK_ID': sid2db[r['struct_id']].add(r['identifier'])
_, omop=read_copy(DC+'omop_relationship.sql')
ind=[r for r in omop if r['relationship_name']=='indication' and r['umls_cui']]
con=[r for r in omop if r['relationship_name']=='contraindication' and r['umls_cui']]
ind_set={(r['struct_id'],r['umls_cui']) for r in ind}
con_set={(r['struct_id'],r['umls_cui']) for r in con}
dd=collections.Counter(r['struct_id'] for r in ind)
sd=collections.Counter(r['umls_cui'] for r in ind)
cname={r['umls_cui']:(r.get('concept_name') or '') for r in ind+con if r.get('concept_name')}

def nm(sid):
    for db in sid2db.get(sid,()):
        a0=nv.get(db,{}).get('A0_canonical')
        if a0: return db,a0
    return None,None

def mk(src, tier, label, n):
    out=[]; src=list(src); rng.shuffle(src)
    for r in src:
        db,a0=nm(r['struct_id'])
        cn=r.get('concept_name') or cname.get(r['umls_cui'])
        if not a0 or not cn: continue
        out.append({'drugbank_id':db,'struct_id':r['struct_id'],'umls_cui':r['umls_cui'],
                    'A0':a0,'disease':cn,'label':label,'tier':tier,
                    'drug_degree':dd[r['struct_id']],'disease_degree':sd[r['umls_cui']]})
        if len(out)>=n: break
    return out

P  = mk(ind, 'pos', 1, POOL)
NC = mk([r for r in con if (r['struct_id'],r['umls_cui']) not in ind_set], 'neg_contra', 0, POOL)
sids=[r['struct_id'] for r in ind]; cuis=[r['umls_cui'] for r in ind]
NR=[]
while len(NR)<POOL:
    s,c = rng.choice(sids), rng.choice(cuis)
    if (s,c) in ind_set or (s,c) in con_set: continue
    db,a0=nm(s)
    if not a0 or not cname.get(c): continue
    NR.append({'drugbank_id':db,'struct_id':s,'umls_cui':c,'A0':a0,'disease':cname[c],
               'label':0,'tier':'neg_random','drug_degree':dd[s],'disease_degree':sd[c]})
print(f"kolam: pos={len(P)} contra={len(NC)} random={len(NR)}")

allr=P+NC+NR
def quint(v):
    s=sorted(v); n=len(s); return [s[int(n*f)] for f in (0.2,0.4,0.6,0.8)]
qd=quint([r['drug_degree'] for r in allr]); qs=quint([r['disease_degree'] for r in allr])
def b(v,q): return sum(1 for x in q if v>x)
for r in allr: r['stratum']=(b(r['drug_degree'],qd), b(r['disease_degree'],qs))

def auroc(y,s):
    pair=sorted(zip(s,y)); rk=[0]*len(pair); i=0
    while i<len(pair):
        j=i
        while j+1<len(pair) and pair[j+1][0]==pair[i][0]: j+=1
        for k in range(i,j+1): rk[k]=(i+j)/2+1
        i=j+1
    n1=sum(y); n0=len(y)-n1
    s1=sum(rk[k] for k in range(len(pair)) if pair[k][1]==1)
    return (s1-n1*(n1+1)/2)/(n1*n0) if n1 and n0 else float('nan')

final=[]; usedP=set()
for N,tier in ((NC,'neg_contra'),(NR,'neg_random')):
    byst=collections.defaultdict(list)
    for r in N: byst[r['stratum']].append(r)
    for v in byst.values(): rng.shuffle(v)
    mP,mN=[],[]
    for p in P:
        pool=byst.get(p['stratum'])
        if pool: mN.append(pool.pop()); mP.append(p)
    y=[1]*len(mP)+[0]*len(mN)
    print(f"\n{tier}: tercocok {len(mP)} pasang")
    print(f"  AUROC derajat obat     = {auroc(y,[r['drug_degree'] for r in mP+mN]):.3f}")
    print(f"  AUROC derajat penyakit = {auroc(y,[r['disease_degree'] for r in mP+mN]):.3f}")
    final+=mN
    for r in mP:
        k=(r['drugbank_id'],r['umls_cui'])
        if k not in usedP: usedP.add(k); final.append(r)

BANNED={'IMMU','INCB','LOXO','MEDI','ASTX','BMS','MOR','TRV','INC','PM','UX','DCC','NS'}
CONS='BCDFGHJKLMNPQRSTVWXZ'; pseudo={}; seen=set()
for d in sorted({r['drugbank_id'] for r in final}):
    while True:
        p=''.join(rng.choice(CONS) for _ in range(4))
        if p not in BANNED and p not in seen: break
    seen.add(p); pseudo[d]=f"{p}-{rng.randint(100,999)}"
for r in final: r['A4']=pseudo[r['drugbank_id']]
rng.shuffle(final)

flds=[k for k in final[0].keys() if k!='stratum']
with open(f'{OUT}/pilot_pairs_v3.tsv','w',newline='') as fh:
    w=csv.DictWriter(fh, fieldnames=flds, delimiter='\t', extrasaction='ignore')
    w.writeheader(); w.writerows(final)
print(f"\nditulis: {OUT}/pilot_pairs_v3.tsv ({len(final)})")
for t in ('pos','neg_contra','neg_random'):
    g=[r for r in final if r['tier']==t]
    print(f"  {t:<12} {len(g):>4}  obat unik={len({r['drugbank_id'] for r in g})}")
