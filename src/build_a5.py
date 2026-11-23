"""Bangun varian A5: deskripsi mekanisme tanpa identitas + kovariat identifiabilitas.

Temuan R: 51% kelas EPC hanya berisi 1 obat, jadi A5 TIDAK BISA dibuat
non-identifikasi untuk semua obat. Desainnya: UKUR identifiabilitas (ukuran
kelas), lalu stratifikasi:
  kelas >=11 anggota -> analisis A5 utama (uji penalaran mekanistik sejati)
  kelas <=3  anggota -> KONTROL POSITIF (model seharusnya benar; bila tidak, probe rusak)
"""
import sys, os, csv, collections, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pgcopy import read_copy
DC='data/interim/drugcentral/'; OUT='data/processed'

# nama kanonik (utk scrubbing)
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

_, pc = read_copy(DC+'pharma_class.sql')
epc=collections.defaultdict(set); moa=collections.defaultdict(set)
for r in pc:
    if not r.get('name') or not r.get('struct_id'): continue
    if r['type']=='EPC': epc[r['struct_id']].add(r['name'])
    elif r['type']=='MoA': moa[r['struct_id']].add(r['name'])

# ukuran kelas = proksi identifiabilitas
cls_members=collections.defaultdict(set)
for sid,cs in epc.items():
    for c in cs: cls_members[c].add(sid)

_, act = read_copy(DC+'act_table_full.sql')
targets=collections.defaultdict(set)
for r in act:
    at=(r.get('action_type') or '').strip()
    g=(r.get('gene') or '').strip()
    if r.get('struct_id') and at and g and '|' not in g:   # buang kompleks multi-gen
        targets[r['struct_id']].add((g, at.lower()))

_, s2a = read_copy(DC+'struct2atc.sql'); _, atc = read_copy(DC+'atc.sql')
code2l4={r['code']:(r.get('l4_name') or r.get('l3_name')) for r in atc if r.get('code')}
atcby=collections.defaultdict(set)
for r in s2a:
    lbl=code2l4.get(r.get('atc_code') or r.get('code'))
    if r.get('struct_id') and lbl: atcby[r['struct_id']].add(lbl)

# penanda identitas yg harus dibuang dari deskripsi
DATE=re.compile(r'\b(19|20)\d{2}\b')
ORDINAL=re.compile(r'\b(first|second|only|novel|new)\b', re.I)

def scrub(txt, sid):
    """Buang nama obat, tahun, dan penanda urutan dari teks."""
    nm=(names.get(sid) or '').lower()
    if nm and len(nm)>3:
        txt=re.sub(re.escape(nm), 'the compound', txt, flags=re.I)
        # juga akar nama (8 huruf pertama) utk varian garam
        if len(nm)>8: txt=re.sub(re.escape(nm[:8]), 'the compound', txt, flags=re.I)
    txt=DATE.sub('', txt); txt=ORDINAL.sub('', txt)
    return re.sub(r'\s+',' ', txt).strip()

# indeks tanda tangan deskripsi -> himpunan obat (dihitung sekali)
def _sig(sid):
    e=sorted(epc.get(sid,()), key=len); m=sorted(moa.get(sid,()), key=len)
    core = e[0] if e else (m[0] if m else None)
    if not core: return None
    a=sorted(atcby.get(sid,()))
    return (core.lower(), (a[0].lower() if a else ''), tuple(sorted(sorted(targets.get(sid,()))[:3])))

sig_members=collections.defaultdict(set)
for sid in set(list(epc)+list(moa)):
    sg=_sig(sid)
    if sg: sig_members[sg].add(sid)
print(f"tanda tangan deskripsi unik: {len(sig_members)} utk {sum(len(v) for v in sig_members.values())} obat")

def describe(sid):
    parts=[]
    e=sorted(epc.get(sid,()), key=len)
    m=sorted(moa.get(sid,()), key=len)
    core = e[0] if e else (m[0] if m else None)
    if not core: return None, 0
    parts.append(f"a compound of the pharmacologic class \"{core.lower()}\"")
    tg=sorted(targets.get(sid,()))[:3]
    if tg:
        ts=", ".join(f"{a}s {g}" if a.endswith(('e','r')) else f"{a} of {g}" for g,a in tg)
        parts.append(f"which acts as {ts}")
    a=sorted(atcby.get(sid,()))
    if a: parts.append(f"therapeutic category: {a[0].lower()}")
    # KOVARIAT IDENTIFIABILITAS: berapa obat lain yg berbagi TANDA TANGAN YG SAMA
    # (EPC + ATC + himpunan target). Ukuran kelas EPC saja tidak cukup -- mis.
    # flibanserin punya kelas EPC besar tapi kombinasi ATC+reseptornya unik.
    sig = (core.lower(),
           (a[0].lower() if a else ''),
           tuple(sorted(tg)))
    size = len(sig_members.get(sig, {sid}))
    return scrub("; ".join(parts), sid), size

def describe_coarse(sid):
    """Varian kasar: HANYA kelas farmakologis. Jauh kurang informatif utk
    penalaran, tapi jauh kurang membocorkan identitas."""
    e=sorted(epc.get(sid,()), key=len); m=sorted(moa.get(sid,()), key=len)
    core = e[0] if e else (m[0] if m else None)
    if not core: return None, 0
    size = len(cls_members.get(core, {sid})) if core in cls_members else 1
    return scrub(f'a compound of the pharmacologic class "{core.lower()}"', sid), size

rows=list(csv.DictReader(open(f'{OUT}/pilot_pairs_v3.tsv'), delimiter='\t'))
_, ident=read_copy(DC+'identifier.sql')
db2sid=collections.defaultdict(set)
for r in ident:
    if r['id_type']=='DRUGBANK_ID': db2sid[r['identifier']].add(r['struct_id'])

ok=0; miss=0; strat=collections.Counter()
for r in rows:
    desc,size=None,0
    for s in db2sid.get(r['drugbank_id'],()):
        d,sz=describe(s)
        if d: desc,size=d,sz; break
    dc,cs=None,0
    for s_ in db2sid.get(r['drugbank_id'],()):
        d2,sz2=describe_coarse(s_)
        if d2: dc,cs=d2,sz2; break
    r['A5']=desc or ''
    r['A5c']=dc or ''
    r['sig_class_size']=size
    r['coarse_class_size']=cs
    r['ident_stratum']=('tanpa_deskripsi' if not desc else
                        'unik (n=1)' if size<=1 else
                        'teridentifikasi (2-3)' if size<=3 else
                        'menengah (4-10)' if size<=10 else 'non_identifikasi (>=11)')
    strat[r['ident_stratum']]+=1
    ok += bool(desc); miss += not desc

flds=list(rows[0].keys())
with open(f'{OUT}/pilot_pairs_v4.tsv','w',newline='') as fh:
    w=csv.DictWriter(fh, fieldnames=flds, delimiter='\t'); w.writeheader(); w.writerows(rows)

print(f"pasangan: {len(rows)} | A5 terbangun: {ok} ({100*ok/len(rows):.0f}%) | gagal: {miss}\n")
print("=== stratum identifiabilitas (ukuran kelas EPC) ===")
for k in ('unik (n=1)','teridentifikasi (2-3)','menengah (4-10)','non_identifikasi (>=11)','tanpa_deskripsi'):
    n=strat[k]; print(f"  {k:<18} {n:>5} ({100*n/len(rows):.0f}%)")
print(f"\n  -> analisis A5 UTAMA  : non_identifikasi >=11 ({strat['non_identifikasi (>=11)']} pasangan)")
print(f"  -> KONTROL POSITIF    : unik n=1 ({strat['unik (n=1)']} pasangan)")
print("\n=== contoh deskripsi A5 ===")
seen=set()
for r in rows:
    if r['A5'] and r['ident_stratum'] not in seen:
        seen.add(r['ident_stratum'])
        print(f"\n[{r['ident_stratum']}, kelas={r['sig_class_size']}] obat asli: {r['A0']}")
        print(f"  A5: {r['A5'][:200]}")
import collections as _c
cb=_c.Counter()
for r in rows:
    cs=r['coarse_class_size']
    cb['tanpa' if not r['A5c'] else 'unik (n=1)' if cs<=1 else '2-3' if cs<=3
       else '4-10' if cs<=10 else '>=11'] += 1
print("\n=== varian KASAR (hanya kelas farmakologis) ===")
for k in ('unik (n=1)','2-3','4-10','>=11','tanpa'):
    print(f"  {k:<14} {cb[k]:>5} ({100*cb[k]/len(rows):.0f}%)")
print(f"  -> kolam non-identifikasi varian kasar: {cb['>=11']} vs {strat['non_identifikasi (>=11)']} pd varian penuh")
print(f"\nditulis: {OUT}/pilot_pairs_v4.tsv")
