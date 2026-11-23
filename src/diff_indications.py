"""Diff indikasi DrugCentral 2023-11-01 vs 2026-09-25 -> pasangan indikasi BARU.

Pakai: python3 -I src/diff_indications.py
Prasyarat: data/interim/dc2023/omop_relationship.sql dan
           data/interim/drugcentral/omop_relationship.sql
"""
import sys, os, csv, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pgcopy import read_copy

OLD = 'data/interim/dc2023/omop_relationship.sql'
NEW = 'data/interim/drugcentral/omop_relationship.sql'
CUT = '2023-12-31'

def ind_pairs(path):
    _, rows = read_copy(path)
    out = {}
    for r in rows:
        if r.get('relationship_name') == 'indication' and r.get('umls_cui'):
            out[(r['struct_id'], r['umls_cui'])] = r.get('concept_name') or ''
    return out

old = ind_pairs(OLD)
new = ind_pairs(NEW)
added   = {k: v for k, v in new.items() if k not in old}
removed = {k: v for k, v in old.items() if k not in new}

print(f"indikasi 2023-11-01 : {len(old)}")
print(f"indikasi 2026-09-25 : {len(new)}")
print(f"  DITAMBAH          : {len(added)}")
print(f"  DIHAPUS           : {len(removed)}  (tanda perubahan kurasi, bukan de-approval)")
print(f"  obat unik di tambahan: {len({k[0] for k in added})}")
print(f"  penyakit unik        : {len({k[1] for k in added})}")

# nama obat utk keterbacaan
names = {}
if os.path.exists('data/interim/drugcentral/structures.sql'):
    _, st = read_copy('data/interim/drugcentral/structures.sql')
    for r in st:
        if r.get('id'): names[r['id']] = r.get('name') or ''

print("\ncontoh 20 pasangan indikasi BARU:")
for (sid, cui), cname in list(added.items())[:20]:
    print(f"  {names.get(sid,'struct '+str(sid))[:30]:<30} | {cname[:44]:<44} | {cui}")

os.makedirs('data/processed', exist_ok=True)
with open('data/processed/new_indications_2023_2026.tsv', 'w', newline='') as fh:
    w = csv.writer(fh, delimiter='\t')
    w.writerow(['struct_id', 'drug_name', 'umls_cui', 'concept_name'])
    for (sid, cui), cname in sorted(added.items()):
        w.writerow([sid, names.get(sid, ''), cui, cname])
print(f"\nditulis: data/processed/new_indications_2023_2026.tsv ({len(added)} baris)")
