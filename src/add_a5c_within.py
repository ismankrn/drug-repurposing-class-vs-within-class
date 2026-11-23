"""Tambah A5c ke set dalam-kelas sbg pemeriksaan konstruksi.

Positif & negatif berasal dari kelas EPC yg SAMA -> deskripsi A5c-nya identik
-> AUROC A5c WAJIB ~0,5. Bila tidak, ada yg salah di pipeline.
"""
import csv
P='data/processed/within_class_pairs.tsv'
rows=list(csv.DictReader(open(P), delimiter='\t'))
for r in rows:
    c=(r.get('shared_class') or '').strip()
    r['A5c']=f'a compound of the pharmacologic class "{c.lower()}"' if c else ''
    r['sig_class_size']=0; r['a5t_class_size']=0; r['coarse_class_size']=0
with open(P,'w',newline='') as fh:
    w=csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter='\t')
    w.writeheader(); w.writerows(rows)
print(f"A5c terisi: {sum(1 for r in rows if r['A5c'])}/{len(rows)}")
