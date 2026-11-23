"""GERBANG 1.5 final: hitung pasangan post-cutoff (Des 2023) yg benar2 tersedia."""
import sys, csv, json, collections, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pgcopy import read_copy
DC='data/interim/drugcentral/'
CUTOFF='2023-12-31'   # cutoff OLMo-2, seragam semua ukuran

# --- repoDB + jembatan ID ---
with open('data/raw/repodb/full.csv',newline='',encoding='utf-8-sig') as fh:
    repodb=list(csv.DictReader(fh))
_, ident=read_copy(DC+'identifier.sql')
db2sid=collections.defaultdict(set)
for r in ident:
    if r['id_type']=='DRUGBANK_ID': db2sid[r['identifier']].add(r['struct_id'])
_, s2ob=read_copy(DC+'struct2obprod.sql')
_, obp=read_copy(DC+'ob_product.sql')
ob={r['id']:r for r in obp}

sid2nda=collections.defaultdict(set)
for r in s2ob:
    p=ob.get(r['prod_id'])
    if p and p.get('appl_no') and p.get('appl_type')=='N':
        sid2nda[r['struct_id']].add('NDA'+str(p['appl_no']).zfill(6))

rb_drugs={r['drug_id'] for r in repodb}
drug2nda={d:set().union(*[sid2nda[s] for s in db2sid.get(d,()) if s in sid2nda] or [set()])
          for d in rb_drugs}
drug2nda={k:v for k,v in drug2nda.items() if v}
print(f"obat repoDB dgn >=1 NDA: {len(drug2nda)}/{len(rb_drugs)} ({100*len(drug2nda)/len(rb_drugs):.1f}%)")

# --- openFDA cache ---
apps=json.load(open('data/raw/openfda/drugsfda_efficacy_apps.json'))
print(f"aplikasi EFFICACY di cache: {len(apps)}")
by_type=collections.Counter(str(a.get('application_number',''))[:3] for a in apps)
print(f"  jenis aplikasi: {by_type.most_common()}")

appmap={a.get('application_number'):a for a in apps}

def eff_dates(appl):
    a=appmap.get(appl)
    if not a: return []
    out=[]
    for s in a.get('submissions',[]):
        if s.get('submission_class_code')=='EFFICACY' and s.get('submission_status')=='AP':
            d=str(s.get('submission_status_date') or '')
            if len(d)==8: out.append(f"{d[:4]}-{d[4:6]}-{d[6:]}")
    return out

# --- irisan ---
hit_drugs={}; post_supp=0; post_drugs=set(); all_supp=0
for d,ndas in drug2nda.items():
    ds=[x for n in ndas for x in eff_dates(n)]
    if ds:
        hit_drugs[d]=ds; all_supp+=len(ds)
        p=[x for x in ds if x>CUTOFF]
        if p: post_drugs.add(d); post_supp+=len(p)

print(f"\n=== IRISAN repoDB x openFDA EFFICACY ===")
print(f"  obat repoDB dgn >=1 supplement EFFICACY (kapan pun): {len(hit_drugs)}")
print(f"  total supplement EFFICACY pd obat tsb              : {all_supp}")
print(f"  obat dgn supplement EFFICACY SETELAH {CUTOFF}      : {len(post_drugs)}")
print(f"  jumlah supplement post-cutoff                      : {post_supp}")

# pasangan repoDB yg terkena
rows_post=[r for r in repodb if r['drug_id'] in post_drugs]
print(f"  baris repoDB dr obat2 tsb                          : {len(rows_post)}")

print(f"\n=== BATAS ATAS PASANGAN POST-CUTOFF ===")
print(f"  (1 supplement EFFICACY ~ <=1 indikasi baru)")
print(f"  batas atas dari jalur NDA/Orange Book              : {post_supp}")

# --- cek bias biologik ---
bla=[a for a in apps if str(a.get('application_number','')).startswith('BLA')]
bla_post=0
for a in bla:
    for s in a.get('submissions',[]):
        if s.get('submission_class_code')=='EFFICACY' and s.get('submission_status')=='AP':
            d=str(s.get('submission_status_date') or '')
            if len(d)==8 and f"{d[:4]}-{d[4:6]}-{d[6:]}">CUTOFF: bla_post+=1
print(f"\n=== ⚠️ CELAH BIOLOGIK (BLA tidak ada di Orange Book) ===")
print(f"  aplikasi BLA dgn supplement EFFICACY : {len(bla)}")
print(f"  supplement EFFICACY BLA post-cutoff  : {bla_post}")
print(f"  -> biologik TIDAK terjangkau jembatan Orange Book; butuh Purple Book")
