"""Selesaikan empat keputusan normalisasi dgn angka konkret + siapkan jembatan openFDA."""
import sys, csv, collections, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pgcopy import read_copy
DC='data/interim/drugcentral/'

with open('data/raw/repodb/full.csv', newline='', encoding='utf-8-sig') as fh:
    repodb=list(csv.DictReader(fh))
_, ident = read_copy(DC+'identifier.sql')
db2sid=collections.defaultdict(set)
for r in ident:
    if r['id_type']=='DRUGBANK_ID': db2sid[r['identifier']].add(r['struct_id'])
_, s2p = read_copy(DC+'struct2parent.sql')
sid2par={r['struct_id']:r['parent_id'] for r in s2p}
_, omop = read_copy(DC+'omop_relationship.sql')
dc_ind={(r['struct_id'],r['umls_cui']) for r in omop
        if r['relationship_name']=='indication' and r['umls_cui']}
_, obp = read_copy(DC+'ob_product.sql')
_, prod = read_copy(DC+'product.sql')

print("="*62)
print("KEPUTUSAN 1 — garam/ester: kolaps ke molekul induk?")
multi={k:v for k,v in db2sid.items() if len(v)>1}
resolved=0
for k,v in multi.items():
    pars={sid2par.get(s,s) for s in v}
    if len(pars)==1: resolved+=1
print(f"  DrugBank ID -> >1 struct_id          : {len(multi)}")
print(f"  teratasi bila dikolaps ke induk      : {resolved} ({100*resolved/max(len(multi),1):.0f}%)")
print(f"  sisa ambigu setelah kolaps           : {len(multi)-resolved}")
print(f"  struct2parent tersedia utk            : {len(sid2par)} struct_id")
print("  -> REKOMENDASI: kolaps ke molekul induk (active moiety). Alasan: identitas")
print("     farmakologis ditentukan active moiety, bukan bentuk garamnya.")

print("\n"+"="*62)
print("KEPUTOSAN 2 — produk kombinasi")
cnt=collections.Counter()
for r in prod:
    try: cnt[int(r['active_ingredient_count'] or 0)]+=1
    except ValueError: pass
total=sum(cnt.values())
mono=cnt.get(1,0); combo=total-mono-cnt.get(0,0)
print(f"  produk total                         : {total}")
print(f"  mono-ingredient (count=1)            : {mono} ({100*mono/max(total,1):.1f}%)")
print(f"  kombinasi (count>=2)                 : {combo} ({100*combo/max(total,1):.1f}%)")
print(f"  distribusi count: {sorted(cnt.items())[:8]}")
print("  -> CATATAN: kombinasi ada di tingkat PRODUK, bukan struct_id. repoDB")
print("     memakai DrugBank ID (zat tunggal), jadi isu ini TIDAK mengenai pasangan")
print("     repoDB. Keputusan: tidak ada aksi; dokumentasikan sbg tidak-berlaku.")

print("\n"+"="*62)
print("KEPUTUSAN 4 — duplikat & status bertentangan")
bystat=collections.defaultdict(set)
rowsby=collections.defaultdict(list)
for r in repodb:
    k=(r['drug_id'],r['ind_id']); bystat[k].add(r['status']); rowsby[k].append(r)
conf={k:v for k,v in bystat.items() if 'Approved' in v and len(v)>1}
print(f"  pasangan dgn status bertentangan     : {len(conf)}")
for k in list(conf)[:21]:
    r0=rowsby[k][0]
    sts=sorted({x['status'] for x in rowsby[k]})
    print(f"    {r0['drug_name'][:26]:<26} | {r0['ind_name'][:34]:<34} | {sts}")
print("  -> REKOMENDASI: Approved menang (bukti regulatori > kegagalan uji klinis)")

print("\n"+"="*62)
print("TEMUAN: pasangan 'GAGAL' di repoDB yg kini jadi indikasi disetujui")
def exp(r): return {(s,r['ind_id']) for s in db2sid.get(r['drug_id'],())}
late=[r for r in repodb if r['status']!='Approved' and (exp(r) & dc_ind)]
print(f"  jumlah: {len(late)}")
print(f"  obat unik: {len({r['drug_id'] for r in late})}")
print("  contoh 12:")
for r in late[:12]:
    print(f"    {r['drug_name'][:24]:<24} | {r['ind_name'][:36]:<36} | {r['status']:<11} {r['phase']}")

print("\n"+"="*62)
print("JEMBATAN openFDA: appl_no untuk obat repoDB")
sid2appl=collections.defaultdict(set)
ing2appl=collections.defaultdict(set)
for r in obp:
    if r.get('appl_no'): ing2appl[(r.get('ingredient') or '').upper()].add(r['appl_no'])
rb_sids={s for r in repodb for s in db2sid.get(r['drug_id'],())}
hit=sum(1 for r in repodb if (r['drug_name'] or '').upper() in ing2appl)
print(f"  ob_product baris                     : {len(obp)}")
print(f"  ingredient unik di Orange Book       : {len(ing2appl)}")
print(f"  baris repoDB yg namanya cocok OB     : {hit}/{len(repodb)} ({100*hit/len(repodb):.1f}%)")
print(f"  rentang approval_date OB             : ", end="")
ds=sorted(r['approval_date'] for r in obp if r.get('approval_date'))
print(f"{ds[0]} .. {ds[-1]} (n={len(ds)})" if ds else "tidak ada")
