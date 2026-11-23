"""Bangun varian nama A0/A1/A2 untuk obat repoDB dari DrugCentral.

A0 = nama kanonik (structures.name)   -- WAJIB dari DrugCentral, bukan repoDB (260 baris 'NA')
A1 = nama dagang (ob_product.trade_name, tersaring)
A2 = sinonim / varian nomenklatur (synonyms.name)

Keluaran: data/processed/name_variants.tsv
"""
import sys, os, csv, collections, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pgcopy import read_copy
DC='data/interim/drugcentral/'

def norm(s): return re.sub(r'[^a-z0-9]','',(s or '').lower())

# A0
names={}
with open(DC+'structures.sql', encoding='utf-8', errors='replace') as fh:
    indata=False
    for line in fh:
        if not indata:
            if line.startswith('COPY public.structures ('):
                cols=[x.strip() for x in line[line.index('(')+1:line.index(')')].split(',')]
                i_id,i_nm=cols.index('id'),cols.index('name'); indata=True
            continue
        if line.startswith('\\.'): break
        v=line.rstrip('\n').split('\t')
        if len(v)>max(i_id,i_nm): names[v[i_id]]=None if v[i_nm]=='\\N' else v[i_nm]

# A1
_, s2ob=read_copy(DC+'struct2obprod.sql'); _, obp=read_copy(DC+'ob_product.sql')
ob={r['id']:r for r in obp}
trade=collections.defaultdict(set)
dropped=collections.Counter()
for r in s2ob:
    p=ob.get(r['prod_id']); sid=r['struct_id']
    if not p or not p.get('trade_name'): continue
    tn=p['trade_name']; ing=p.get('ingredient') or ''
    if ';' in ing:                       dropped['kombinasi (ingredient ;)']+=1; continue
    if ' AND ' in tn.upper():            dropped['kombinasi (nama AND)']+=1; continue
    cn=norm(names.get(sid)); tnn=norm(tn)
    if cn and (tnn==cn or cn in tnn):    dropped['generik (= nama kanonik)']+=1; continue
    trade[sid].add(tn)

# A1b -- sumber kedua: NDC (mencakup biologik, yg absen di Orange Book)
_, ai=read_copy(DC+'active_ingredient.sql'); _, prod=read_copy(DC+'product.sql')
ndc2prod={r['ndc_product_code']:r for r in prod if r.get('ndc_product_code')}
for r in ai:
    sid=r.get('struct_id'); ndc=r.get('ndc_product_code')
    if not sid or not ndc: continue
    p_=ndc2prod.get(ndc)
    if not p_ or not p_.get('product_name'): continue
    try:
        if int(p_.get('active_ingredient_count') or 1) > 1:
            dropped['kombinasi (NDC count>1)']+=1; continue
    except ValueError: pass
    pn=p_['product_name']; cn=norm(names.get(sid))
    if cn and (norm(pn)==cn or cn in norm(pn)):
        dropped['generik NDC (= nama kanonik)']+=1; continue
    trade[sid].add(pn)

# A2
_, syn=read_copy(DC+'synonyms.sql')
synby=collections.defaultdict(set)
for r in syn:
    sid, nm = r.get('id'), r.get('name')
    if sid and nm and norm(nm)!=norm(names.get(sid)): synby[sid].add(nm)

# obat repoDB
with open('data/raw/repodb/full.csv',newline='',encoding='utf-8-sig') as fh:
    repodb=list(csv.DictReader(fh))
_, ident=read_copy(DC+'identifier.sql')
db2sid=collections.defaultdict(set)
for r in ident:
    if r['id_type']=='DRUGBANK_ID': db2sid[r['identifier']].add(r['struct_id'])
rb_drugs=sorted({r['drug_id'] for r in repodb})

print("alasan trade_name dibuang:", dropped.most_common())
os.makedirs('data/processed', exist_ok=True)
nA0=nA1=nA2=0
with open('data/processed/name_variants.tsv','w',newline='') as fh:
    w=csv.writer(fh, delimiter='\t')
    w.writerow(['drugbank_id','struct_ids','A0_canonical','A1_trade','A2_synonyms'])
    for d in rb_drugs:
        sids=sorted(db2sid.get(d,()))
        a0=next((names[s] for s in sids if names.get(s)), '')
        a1=sorted({x for s in sids for x in trade.get(s,())})
        a2=sorted({x for s in sids for x in synby.get(s,())})
        nA0+=bool(a0); nA1+=bool(a1); nA2+=bool(a2)
        w.writerow([d,'|'.join(sids),a0,'|'.join(a1[:10]),'|'.join(a2[:10])])

n=len(rb_drugs)
print(f"\nobat repoDB: {n}")
print(f"  A0 (kanonik) tersedia : {nA0} ({100*nA0/n:.1f}%)")
print(f"  A1 (nama dagang)      : {nA1} ({100*nA1/n:.1f}%)")
print(f"  A2 (sinonim)          : {nA2} ({100*nA2/n:.1f}%)")
print("ditulis: data/processed/name_variants.tsv")
