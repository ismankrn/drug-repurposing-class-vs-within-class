"""Tambah varian A5t (kelas + target, TANPA ATC) -> gradien 3 tingkat lengkap.

Temuan Y: target menyumbang +26 poin kebocoran, ATC +19 poin. A5t menguji
titik tengah: cukup kaya utk menalar, tanpa lapisan ATC yg paling identifikasi.
"""
import sys, os, csv, collections, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pgcopy import read_copy
DC='data/interim/drugcentral/'; OUT='data/processed'

names={}
with open(DC+'structures.sql', encoding='utf-8', errors='replace') as fh:
    indata=False
    for line in fh:
        if not indata:
            if line.startswith('COPY public.structures ('):
                c=[x.strip() for x in line[line.index('(')+1:line.index(')')].split(',')]
                i_id,i_nm=c.index('id'),c.index('name'); indata=True
            continue
        if line.startswith('\\.'): break
        v=line.rstrip('\n').split('\t')
        if len(v)>max(i_id,i_nm): names[v[i_id]]=None if v[i_nm]=='\\N' else v[i_nm]

_, pc=read_copy(DC+'pharma_class.sql')
epc=collections.defaultdict(set); moa=collections.defaultdict(set)
for r in pc:
    if not r.get('name') or not r.get('struct_id'): continue
    if r['type']=='EPC': epc[r['struct_id']].add(r['name'])
    elif r['type']=='MoA': moa[r['struct_id']].add(r['name'])
_, act=read_copy(DC+'act_table_full.sql')
targets=collections.defaultdict(set)
for r in act:
    at=(r.get('action_type') or '').strip(); g=(r.get('gene') or '').strip()
    if r.get('struct_id') and at and g and '|' not in g: targets[r['struct_id']].add((g, at.lower()))

DATE=re.compile(r'\b(19|20)\d{2}\b'); ORD=re.compile(r'\b(first|second|only|novel|new)\b', re.I)
def scrub(t, sid):
    nm=(names.get(sid) or '').lower()
    if nm and len(nm)>3:
        t=re.sub(re.escape(nm),'the compound',t,flags=re.I)
        if len(nm)>8: t=re.sub(re.escape(nm[:8]),'the compound',t,flags=re.I)
    return re.sub(r'\s+',' ', ORD.sub('', DATE.sub('', t))).strip()

def sig_t(sid):
    e=sorted(epc.get(sid,()),key=len); m=sorted(moa.get(sid,()),key=len)
    core=e[0] if e else (m[0] if m else None)
    if not core: return None
    return (core.lower(), tuple(sorted(sorted(targets.get(sid,()))[:3])))
sigm=collections.defaultdict(set)
for sid in set(list(epc)+list(moa)):
    s=sig_t(sid)
    if s: sigm[s].add(sid)

def describe_t(sid):
    e=sorted(epc.get(sid,()),key=len); m=sorted(moa.get(sid,()),key=len)
    core=e[0] if e else (m[0] if m else None)
    if not core: return None,0
    parts=[f'a compound of the pharmacologic class "{core.lower()}"']
    tg=sorted(targets.get(sid,()))[:3]
    if tg:
        parts.append("which acts as "+", ".join(f"{a}s {g}" if a.endswith(('e','r')) else f"{a} of {g}" for g,a in tg))
    return scrub("; ".join(parts), sid), len(sigm.get(sig_t(sid), {sid}))

rows=list(csv.DictReader(open(f'{OUT}/pilot_pairs_v4.tsv'), delimiter='\t'))
_, ident=read_copy(DC+'identifier.sql')
db2sid=collections.defaultdict(set)
for r in ident:
    if r['id_type']=='DRUGBANK_ID': db2sid[r['identifier']].add(r['struct_id'])

ok=0; bins=collections.Counter()
for r in rows:
    d,sz=None,0
    for s in db2sid.get(r['drugbank_id'],()):
        dd,ss=describe_t(s)
        if dd: d,sz=dd,ss; break
    r['A5t']=d or ''; r['a5t_class_size']=sz
    ok+=bool(d)
    bins['tanpa' if not d else 'unik (n=1)' if sz<=1 else '2-3' if sz<=3 else '4-10' if sz<=10 else '>=11']+=1

with open(f'{OUT}/pilot_pairs_v5.tsv','w',newline='') as fh:
    w=csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter='\t'); w.writeheader(); w.writerows(rows)

print(f"A5t terbangun: {ok}/{len(rows)} ({100*ok/len(rows):.0f}%)\n")
print("=== gradien identifiabilitas 3 tingkat (pasangan non-identifikasi >=11) ===")
def cnt(field,size):
    c=collections.Counter()
    for r in rows:
        v=int(r.get(size,0) or 0)
        c['tanpa' if not r.get(field) else 'unik' if v<=1 else '2-3' if v<=3 else '4-10' if v<=10 else '>=11']+=1
    return c
for lbl,f,s in (('A5c  kelas saja','A5c','coarse_class_size'),
                ('A5t  kelas+target','A5t','a5t_class_size'),
                ('A5   kelas+target+ATC','A5','sig_class_size')):
    c=cnt(f,s); n=len(rows)
    print(f"  {lbl:<24} non-identifikasi={c['>=11']:>5} ({100*c['>=11']/n:>2.0f}%)   unik={c['unik']:>5} ({100*c['unik']/n:.0f}%)")
print(f"\nditulis: {OUT}/pilot_pairs_v5.tsv")
