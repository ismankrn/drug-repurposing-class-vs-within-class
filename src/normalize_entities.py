"""Normalisasi entitas: repoDB (DrugBank ID, UMLS CUI) <-> DrugCentral (struct_id).

Keluaran:
  data/interim/map_drugbank_structid.tsv
  data/interim/repodb_normalized.tsv
Mencetak laporan cakupan + angka untuk empat keputusan normalisasi.
"""
import sys, csv, collections, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pgcopy import read_copy

DC = 'data/interim/drugcentral/'
OUT = 'data/interim/'

# ---------- 1. repoDB ----------
with open('data/raw/repodb/full.csv', newline='', encoding='utf-8-sig') as fh:
    repodb = list(csv.DictReader(fh))
rb_drugs = {r['drug_id'] for r in repodb}
rb_cuis = {r['ind_id'] for r in repodb}
print(f"repoDB: {len(repodb)} pasangan | {len(rb_drugs)} obat (DrugBank) | {len(rb_cuis)} penyakit (UMLS CUI)")

# ---------- 2. identifier: DRUGBANK_ID -> struct_id ----------
_, ident = read_copy(DC+'identifier.sql')
db_rows = [r for r in ident if r['id_type'] == 'DRUGBANK_ID']
db2sid = collections.defaultdict(set)
pm = collections.Counter()
for r in db_rows:
    db2sid[r['identifier']].add(r['struct_id'])
    pm[r['parent_match']] += 1
print(f"\nDRUGBANK_ID di DrugCentral: {len(db_rows)} baris -> {len(db2sid)} ID unik")
print(f"  parent_match: {dict(pm)}")
multi = {k: v for k, v in db2sid.items() if len(v) > 1}
print(f"  DrugBank ID yg memetakan ke >1 struct_id: {len(multi)}  <- KEPUTUSAN 1 (garam/ester)")

# ---------- 3. cakupan pemetaan obat ----------
mapped = {d for d in rb_drugs if d in db2sid}
unmapped = rb_drugs - mapped
print(f"\n=== CAKUPAN OBAT ===")
print(f"  terpetakan   : {len(mapped)}/{len(rb_drugs)} ({100*len(mapped)/len(rb_drugs):.1f}%)")
print(f"  tidak terpeta: {len(unmapped)}")
ex = [r['drug_name'] for r in repodb if r['drug_id'] in unmapped][:8]
print(f"  contoh gagal : {sorted(set(ex))}")

# ---------- 4. omop_relationship ----------
_, omop = read_copy(DC+'omop_relationship.sql')
ind = [r for r in omop if r['relationship_name'] == 'indication' and r['umls_cui']]
contra = [r for r in omop if r['relationship_name'] == 'contraindication' and r['umls_cui']]
dc_ind_pairs = {(r['struct_id'], r['umls_cui']) for r in ind}
dc_contra_pairs = {(r['struct_id'], r['umls_cui']) for r in contra}
print(f"\nDrugCentral: {len(dc_ind_pairs)} pasangan indikasi | {len(dc_contra_pairs)} pasangan kontraindikasi (ber-CUI)")

# ---------- 5. tumpang tindih repoDB x DrugCentral ----------
def expand(rec):
    """repoDB row -> himpunan (struct_id, cui)."""
    return {(s, rec['ind_id']) for s in db2sid.get(rec['drug_id'], ())}

appr = [r for r in repodb if r['status'] == 'Approved']
fail = [r for r in repodb if r['status'] != 'Approved']

def overlap(rows, target):
    hit = sum(1 for r in rows if expand(r) & target)
    return hit, len(rows)

h, n = overlap(appr, dc_ind_pairs)
print(f"\n=== VALIDASI SILANG ===")
print(f"  repoDB Approved yg juga indikasi di DrugCentral : {h}/{n} ({100*h/n:.1f}%)")
h2, n2 = overlap(fail, dc_ind_pairs)
print(f"  repoDB GAGAL yg ternyata indikasi di DrugCentral: {h2}/{n2} ({100*h2/n2:.1f}%)  <- KONTAMINASI LABEL")
h3, _ = overlap(appr, dc_contra_pairs)
print(f"  repoDB Approved yg justru KONTRAindikasi        : {h3}/{n}")

# ---------- 6. keputusan 3: granularitas penyakit ----------
cui_names = collections.defaultdict(set)
for r in repodb:
    cui_names[r['ind_id']].add(r['ind_name'])
amb = {k: v for k, v in cui_names.items() if len(v) > 1}
print(f"\n  CUI dgn >1 nama penyakit di repoDB: {len(amb)}  <- KEPUTUSAN 3 (granularitas)")

# ---------- 7. keputusan 4: duplikat ----------
key = collections.Counter((r['drug_id'], r['ind_id']) for r in repodb)
dup = {k: v for k, v in key.items() if v > 1}
print(f"  pasangan (obat,penyakit) duplikat : {len(dup)}  <- KEPUTUSAN 4 (duplikat)")
conflict = 0
bystat = collections.defaultdict(set)
for r in repodb:
    bystat[(r['drug_id'], r['ind_id'])].add(r['status'] == 'Approved')
conflict = sum(1 for k, v in bystat.items() if len(v) > 1)
print(f"  pasangan dgn status BERTENTANGAN (approved & gagal): {conflict}  <- kritis")

# ---------- 8. tulis keluaran ----------
with open(OUT+'map_drugbank_structid.tsv', 'w', newline='') as fh:
    w = csv.writer(fh, delimiter='\t')
    w.writerow(['drugbank_id', 'struct_id', 'n_struct_ids'])
    for k in sorted(db2sid):
        for s in sorted(db2sid[k]):
            w.writerow([k, s, len(db2sid[k])])

with open(OUT+'repodb_normalized.tsv', 'w', newline='') as fh:
    w = csv.writer(fh, delimiter='\t')
    w.writerow(['drug_name','drugbank_id','struct_ids','ind_name','umls_cui',
                'status','phase','detailed_status','in_dc_indication','in_dc_contraindication'])
    for r in repodb:
        sids = sorted(db2sid.get(r['drug_id'], ()))
        p = expand(r)
        w.writerow([r['drug_name'], r['drug_id'], '|'.join(sids), r['ind_name'], r['ind_id'],
                    r['status'], r['phase'], r['DetailedStatus'],
                    int(bool(p & dc_ind_pairs)), int(bool(p & dc_contra_pairs))])
print(f"\nditulis: {OUT}map_drugbank_structid.tsv, {OUT}repodb_normalized.tsv")
