"""Dekomposisi kebocoran A5: komponen mana yang membuat deskripsi teridentifikasi?

Bukan pengganti audit adversarial berbasis model -- ini mengukur, untuk setiap
kombinasi komponen, berapa obat yg berbagi tanda tangan sama. Hasilnya memandu
komponen mana yg harus dibuang sebelum audit model dijalankan.
"""
import sys, os, csv, collections, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pgcopy import read_copy
DC='data/interim/drugcentral/'

_, pc = read_copy(DC+'pharma_class.sql')
epc=collections.defaultdict(set); moa=collections.defaultdict(set)
for r in pc:
    if not r.get('name') or not r.get('struct_id'): continue
    if r['type']=='EPC': epc[r['struct_id']].add(r['name'])
    elif r['type']=='MoA': moa[r['struct_id']].add(r['name'])

_, act = read_copy(DC+'act_table_full.sql')
targets=collections.defaultdict(set)
for r in act:
    at=(r.get('action_type') or '').strip(); g=(r.get('gene') or '').strip()
    if r.get('struct_id') and at and g and '|' not in g:
        targets[r['struct_id']].add((g, at.lower()))

_, s2a = read_copy(DC+'struct2atc.sql'); _, atc = read_copy(DC+'atc.sql')
code2l4={r['code']:(r.get('l4_name') or r.get('l3_name')) for r in atc if r.get('code')}
atcby=collections.defaultdict(set)
for r in s2a:
    lbl=code2l4.get(r.get('atc_code') or r.get('code'))
    if r.get('struct_id') and lbl: atcby[r['struct_id']].add(lbl)

drugs=[s for s in set(list(epc)+list(moa)) if (epc.get(s) or moa.get(s))]
def comp(sid, which):
    out=[]
    if 'C' in which:
        e=sorted(epc.get(sid,()), key=len); m=sorted(moa.get(sid,()), key=len)
        out.append((e[0] if e else (m[0] if m else '')).lower())
    if 'A' in which:
        a=sorted(atcby.get(sid,())); out.append(a[0].lower() if a else '')
    if 'T' in which:
        out.append(tuple(sorted(sorted(targets.get(sid,()))[:3])))
    return tuple(out)

print(f"obat dgn kelas farmakologis: {len(drugs)}\n")
print(f"{'kombinasi':<26} {'unik(n=1)':>10} {'<=3':>8} {'>=11':>8}  {'median':>7}")
print("-"*64)
LBL={'C':'kelas','A':'ATC','T':'target'}
for k in (1,2,3):
    for combo in itertools.combinations('CAT', k):
        w=''.join(combo)
        if 'C' not in w: continue   # kelas selalu jadi inti deskripsi
        sig=collections.defaultdict(set)
        for s in drugs: sig[comp(s,w)].add(s)
        sizes=[len(sig[comp(s,w)]) for s in drugs]
        uniq=sum(1 for x in sizes if x==1); le3=sum(1 for x in sizes if x<=3)
        ge11=sum(1 for x in sizes if x>=11)
        med=sorted(sizes)[len(sizes)//2]
        name=' + '.join(LBL[c] for c in w)
        n=len(sizes)
        print(f"{name:<26} {100*uniq/n:>9.0f}% {100*le3/n:>7.0f}% {100*ge11/n:>7.0f}% {med:>8}")

print("\n=== kontribusi kebocoran tiap komponen (vs kelas saja) ===")
base=collections.defaultdict(set)
for s in drugs: base[comp(s,'C')].add(s)
b_uniq=sum(1 for s in drugs if len(base[comp(s,'C')])==1)/len(drugs)
for w,lab in (('CA','+ ATC'),('CT','+ target'),('CAT','+ ATC + target')):
    sig=collections.defaultdict(set)
    for s in drugs: sig[comp(s,w)].add(s)
    u=sum(1 for s in drugs if len(sig[comp(s,w)])==1)/len(drugs)
    print(f"  {lab:<18} unik naik {100*b_uniq:.0f}% -> {100*u:.0f}%  (+{100*(u-b_uniq):.0f} poin)")
print("\n-> komponen dgn kenaikan terbesar adalah sumber kebocoran utama.")
