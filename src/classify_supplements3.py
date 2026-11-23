"""v3: klasifikasi LENGKAP 665 surat supplement EFFICACY post-cutoff.

Perbaikan atas v2:
  1. unduhan berlapis: Letter -> ulang dgn http -> fallback ke dokumen Label
  2. aturan EKSKLUSI diperluas (PBPK/PK, "updates to", "revisions to", section 17, CMC)
  3. bucket "PERLU TINJAUAN MANUAL" ditulis ke file sebagai worklist, bukan dibuang
"""
import json, os, re, sys, time, subprocess, collections

CUT='2023-12-31'; PDFDIR='data/raw/fda_letters'
os.makedirs(PDFDIR, exist_ok=True)
OUT='data/processed'

apps=json.load(open('data/raw/openfda/drugsfda_efficacy_apps.json'))
items=[]
for a in apps:
    for s in a.get('submissions',[]):
        if s.get('submission_class_code')!='EFFICACY' or s.get('submission_status')!='AP': continue
        d=str(s.get('submission_status_date') or '')
        if not(len(d)==8 and f"{d[:4]}-{d[4:6]}-{d[6:]}">CUT): continue
        docs=s.get('application_docs') or []
        L=next((x.get('url') for x in docs if x.get('type')=='Letter'), None)
        B=next((x.get('url') for x in docs if x.get('type')=='Label'), None)
        if L or B:
            items.append({'appl':a.get('application_number'),'suppl':s.get('submission_number'),
                          'date':f"{d[:4]}-{d[4:6]}-{d[6:]}",'letter':L,'label':B})

PATS=[re.compile(p, re.I|re.S) for p in [
 r'provides?\s+for\s+(.{10,420}?)(?:\.\s|\sAPPROVAL|\sWe have completed)',
 r'supplement(?:al)?[^.]{0,120}?\bis\s+for\s+(.{10,420}?)(?:\.\s|\sAPPROVAL)',
 r'approved?\s+for\s+the\s+following\s+indication[s]?\s*:?\s*(.{10,420}?)(?:\.\s|\sAPPROVAL)',
 r'(?:seeking|requests?|proposes?)\s+(?:approval\s+)?(?:for\s+)?(.{10,420}?)(?:\.\s|\sAPPROVAL)',
]]

# EKSKLUSI -- urutan penting, yg paling spesifik dulu
EXCL=[
 ('penghapusan indikasi', r'\b(removal of|remove the|removing|withdraw(?:al|ing)?\s+(?:of\s+)?the\s+(?:following\s+)?indication)'),
 ('limitations of use',   r'limitations?\s+of\s+use'),
 ('perangkat/sediaan',    r'\b(autoinjector|prefilled|pen injector|new (?:dosage )?form|single-dose|vial|syringe|device|co-?packag)'),
 ('ekstensi pediatrik',   r'\b(pediatric (?:patients|use|labeling|population|subjects)|section 8\.4|\bPREA\b|neonat|adolescent)'),
 ('dosis/kekuatan/PK',    r'\b(new (?:dosage|dosing|strength)|dose adjustment|dosage regimen|section 2\.\d|pharmacokinetic|\bPBPK\b|bioequival|coadminist)'),
 ('keamanan/peringatan',  r'\b(warnings|precautions|adverse reaction|contraindicat|section 5\.\d|section 6\.\d|boxed warning|mortality imbalance)'),
 ('data klinis/label',    r'\b(section 14|clinical studies|section 17|patient counseling|overall survival data|final .{0,20}data)'),
 ('pembaruan label umum', r'^\s*(?:updates?|revisions?|changes?)\s+to\s+(?:the\s+)?(?:USPI|US Prescribing Information|prescribing information|PI\b|label)'),
 ('CMC/manufaktur',       r'\b(manufactur|chemistry|facility|container closure|shelf life|stability)'),
]
NEWIND=r'\b(a new indication|an additional indication|the addition of (?:a |an |the )?(?:new )?indicat|extension of the indication|for the treatment of|as a diagnostic|the indication of)'

def run(cmd, t=100):
    return subprocess.run(cmd, capture_output=True, timeout=t)

def enc(u):
    """URL surat FDA sering memuat SPASI literal (81/665) -> curl menolak:
    'Malformed input to a URL function'. Encode spasi sbg %20. Penyebab
    seluruh 12% kegagalan unduh di v2."""
    return u.replace(' ', '%20') if u else u

def try_dl(url, fn):
    if not url: return False
    url=enc(url)
    for u in (url, url.replace('https://','http://')):
        r=run(['curl','-sSL','-f','--retry','2','--retry-delay','2','--max-time','90','-o',fn,u])
        if r.returncode==0 and os.path.exists(fn) and os.path.getsize(fn)>5000:
            with open(fn,'rb') as fh:
                if fh.read(5)==b'%PDF-': return True
    return False

def get_text(it):
    base=os.path.join(PDFDIR, f"{it['appl']}_s{it['suppl']}")
    fn=base+'.pdf'
    src='letter'
    ok = (os.path.exists(fn) and os.path.getsize(fn)>5000)
    if not ok: ok=try_dl(it['letter'], fn)
    if not ok:
        fn=base+'_label.pdf'; src='label'
        ok = (os.path.exists(fn) and os.path.getsize(fn)>5000) or try_dl(it['label'], fn)
    if not ok: return None,'unduh gagal (letter+label)',src
    try:
        out=run(['pdftotext','-layout',fn,'-'],t=150)
        t=out.stdout.decode('utf-8','replace')
        if len(t.strip())<200: return None,'PDF tanpa lapisan teks',src
        return t,None,src
    except Exception as e: return None,f'pdftotext: {type(e).__name__}',src

print(f"total supplement post-cutoff: {len(items)}", flush=True)
cls=collections.Counter(); srcs=collections.Counter()
cand=[]; review=[]; failed=[]
for i,it in enumerate(items,1):
    txt,err,src=get_text(it)
    srcs[src]+=1
    if err:
        cls[f'GAGAL: {err}']+=1; failed.append((it,err)); 
        if i%50==0: print(f"  ...{i}/{len(items)}", flush=True)
        continue
    t=re.sub(r'\s+',' ',txt)
    prov=''
    for p in PATS:
        m=p.search(t)
        if m: prov=m.group(1).strip(); break
    if not prov:
        cls['pola tidak ditemukan']+=1; review.append((it,'(pola tidak ditemukan)'))
    else:
        lab=None
        for name,pat in EXCL:
            if re.search(pat, prov, re.I): lab=f'EKSKLUSI: {name}'; break
        if lab is None:
            if re.search(NEWIND, prov, re.I): lab='KANDIDAT pasangan baru'; cand.append((it,prov))
            else: lab='PERLU TINJAUAN MANUAL'; review.append((it,prov))
        cls[lab]+=1
    if i%50==0: print(f"  ...{i}/{len(items)}", flush=True)

n=len(items)
print(f"\n=== DISTRIBUSI LENGKAP (n={n}) ===")
for k,v in cls.most_common(): print(f"  {k:<34} {v:>4} ({100*v/n:.1f}%)")
print(f"\nsumber dokumen: {dict(srcs)}")
k=cls['KANDIDAT pasangan baru']; r=cls['PERLU TINJAUAN MANUAL']+cls['pola tidak ditemukan']
print(f"\nKANDIDAT otomatis      : {k} ({100*k/n:.1f}%)")
print(f"PERLU tinjauan manual  : {r} ({100*r/n:.1f}%)")
print(f"batas bawah pasangan   : {k}")
print(f"batas atas (bila semua tinjauan ternyata indikasi baru): {k+r}")

os.makedirs(OUT, exist_ok=True)
with open(f'{OUT}/new_pairs_auto.tsv','w') as fh:
    fh.write('appl\tsuppl\tdate\tprovides_for\n')
    for it,p in cand: fh.write(f"{it['appl']}\t{it['suppl']}\t{it['date']}\t{p[:400].replace(chr(9),' ')}\n")
with open(f'{OUT}/manual_review_worklist.tsv','w') as fh:
    fh.write('appl\tsuppl\tdate\tprovides_for\tlabel_BARU_atau_BUKAN\n')
    for it,p in review: fh.write(f"{it['appl']}\t{it['suppl']}\t{it['date']}\t{p[:400].replace(chr(9),' ')}\t\n")
with open(f'{OUT}/download_failures.tsv','w') as fh:
    fh.write('appl\tsuppl\terror\tletter_url\n')
    for it,e in failed: fh.write(f"{it['appl']}\t{it['suppl']}\t{e}\t{it['letter']}\n")
print(f"\nditulis: new_pairs_auto.tsv ({len(cand)}), manual_review_worklist.tsv ({len(review)}), download_failures.tsv ({len(failed)})")
