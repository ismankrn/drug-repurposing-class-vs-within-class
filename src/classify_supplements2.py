"""v2: klasifikasi supplement EFFICACY post-cutoff -> pasangan obat-penyakit BARU?

Perbaikan atas v1:
  1. unduh via curl (urllib gagal pd URL bertanda ';' = surat multi-aplikasi)
  2. pola ekstraksi diperluas + deteksi PDF tanpa lapisan teks (hasil scan)
  3. aturan EKSKLUSI dijalankan LEBIH DULU (penghapusan, limitations of use,
     ekstensi pediatrik, perangkat/dosis) sebelum deteksi indikasi baru
"""
import json, os, re, sys, time, subprocess, collections, random

CUT='2023-12-31'; PDFDIR='data/raw/fda_letters'
os.makedirs(PDFDIR, exist_ok=True)

apps=json.load(open('data/raw/openfda/drugsfda_efficacy_apps.json'))
items=[]
for a in apps:
    for s in a.get('submissions',[]):
        if s.get('submission_class_code')!='EFFICACY' or s.get('submission_status')!='AP': continue
        d=str(s.get('submission_status_date') or '')
        if not(len(d)==8 and f"{d[:4]}-{d[4:6]}-{d[6:]}">CUT): continue
        L=next((x.get('url') for x in (s.get('application_docs') or []) if x.get('type')=='Letter'), None)
        if L: items.append({'appl':a.get('application_number'),'suppl':s.get('submission_number'),
                            'date':f"{d[:4]}-{d[4:6]}-{d[6:]}",'url':L})

# pola ekstraksi: beberapa varian frasa
PATS=[re.compile(p, re.I|re.S) for p in [
 r'provides?\s+for\s+(.{10,420}?)(?:\.\s|\sAPPROVAL|\sWe have completed)',
 r'supplement(?:al)?[^.]{0,120}?\bis\s+for\s+(.{10,420}?)(?:\.\s|\sAPPROVAL)',
 r'approved?\s+for\s+the\s+following\s+indication[s]?\s*:?\s*(.{10,420}?)(?:\.\s|\sAPPROVAL)',
 r'(?:seeking|requests?|proposes?)\s+(?:approval\s+)?(?:for\s+)?(.{10,420}?)(?:\.\s|\sAPPROVAL)',
]]

# EKSKLUSI dulu -- urutan penting
EXCL=[
 ('penghapusan indikasi', r'\b(removal of|remove the|removing|withdraw(?:al|ing)? (?:of )?the (?:following )?indication)'),
 ('limitations of use',   r'limitations? of use'),
 ('perangkat/sediaan',    r'\b(autoinjector|prefilled|pen injector|new (?:dosage )?form|single-dose|vial|syringe|device)'),
 ('ekstensi pediatrik',   r'\b(pediatric (?:patients|use|labeling|population)|section 8\.4|PREA|neonat|adolescent)'),
 ('dosis/kekuatan',       r'\b(new (?:dosage|dosing|strength)|dose adjustment|dosage regimen|section 2\.\d)'),
 ('label lain',           r'\b(warnings|precautions|adverse reaction|section 5\.\d|section 6\.\d|section 14|boxed warning|clinical studies)'),
]
NEWIND=r'\b(a new indication|an additional indication|addition of (?:a |an |the )?indicat|extension of the indication|for the treatment of|as a diagnostic|in combination with .{0,60}for the)'

def fetch(it):
    fn=os.path.join(PDFDIR, f"{it['appl']}_s{it['suppl']}.pdf")
    if not os.path.exists(fn) or os.path.getsize(fn)<2000:
        r=subprocess.run(['curl','-sSL','--max-time','90','-o',fn,it['url']],capture_output=True)
        if r.returncode!=0 or not os.path.exists(fn): return None,'unduh gagal'
        time.sleep(0.4)
    try:
        out=subprocess.run(['pdftotext','-layout',fn,'-'],capture_output=True,timeout=120)
        t=out.stdout.decode('utf-8','replace')
        if len(t.strip())<200: return None,'PDF tanpa lapisan teks (scan)'
        return t,None
    except Exception as e: return None,f'pdftotext: {e}'

n=int(sys.argv[1]) if len(sys.argv)>1 else 120
random.Random(42).shuffle(items)
sample=items[:n]
print(f"total supplement post-cutoff dgn surat: {len(items)} | sampel: {n}\n", flush=True)

cls=collections.Counter(); keep=[]
for i,it in enumerate(sample,1):
    txt,err=fetch(it)
    if err: cls[f'GAGAL: {err}']+=1; continue
    t=re.sub(r'\s+',' ',txt)
    prov=''
    for p in PATS:
        m=p.search(t)
        if m: prov=m.group(1).strip(); break
    if not prov: cls['pola tidak ditemukan']+=1; continue
    lab=None
    for name,pat in EXCL:
        if re.search(pat, prov, re.I): lab=f'EKSKLUSI: {name}'; break
    if lab is None:
        lab='KANDIDAT pasangan baru' if re.search(NEWIND, prov, re.I) else 'tak terklasifikasi'
    cls[lab]+=1
    if lab.startswith('KANDIDAT'): keep.append((it,prov))
    if i%20==0: print(f"  ...{i}/{n}", flush=True)

print(f"\n=== DISTRIBUSI (n={n}) ===")
for k,v in cls.most_common(): print(f"  {k:<34} {v:>3} ({100*v/n:.0f}%)")
k=cls['KANDIDAT pasangan baru']
lo,hi=max(0,k-1.96*(k*(n-k)/n)**.5), k+1.96*(k*(n-k)/n)**.5
print(f"\nKANDIDAT pasangan baru: {k}/{n} = {100*k/n:.1f}%  (95% CI ~{100*lo/n:.0f}-{100*hi/n:.0f}%)")
print(f"ekstrapolasi ke {len(items)}: ~{round(len(items)*k/n)} pasangan (CI ~{round(len(items)*lo/n)}-{round(len(items)*hi/n)})")

with open('data/processed/candidate_new_pairs.tsv','w') as fh:
    fh.write('appl\tsuppl\tdate\tprovides_for\n')
    for it,prov in keep:
        fh.write(f"{it['appl']}\t{it['suppl']}\t{it['date']}\t{re.sub(chr(9),' ',prov)[:300]}\n")
print(f"\nditulis: data/processed/candidate_new_pairs.tsv ({len(keep)} kandidat)")
print("\n=== 12 kandidat teratas utk tinjauan manual ===")
for it,prov in keep[:12]:
    print(f"  {it['appl']:<12} s{it['suppl']:<4} {it['date']}  {prov[:118]}")
