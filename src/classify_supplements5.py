"""v4: perbaiki bug ekstraksi boilerplate + jalankan ulang atas PDF tersimpan.

Bug v3: regex 'provides for' menangkap kalimat boilerplate
  "...provides for advisory comments on proposed introductory advertising and
   promotional labeling"
yang merupakan bagian LAIN surat (tawaran FDA atas materi iklan), bukan isi
supplement. Terkena ~43% bucket tinjauan manual.

Perbaikan: iterasi SELURUH kecocokan, ambil yg pertama yg BUKAN boilerplate.
"""
import json, os, re, sys, subprocess, collections

CUT='2023-12-31'; PDFDIR='data/raw/fda_letters'; OUT='data/processed'

apps=json.load(open('data/raw/openfda/drugsfda_efficacy_apps.json'))
items=[]
for a in apps:
    for s in a.get('submissions',[]):
        if s.get('submission_class_code')!='EFFICACY' or s.get('submission_status')!='AP': continue
        d=str(s.get('submission_status_date') or '')
        if not(len(d)==8 and f"{d[:4]}-{d[4:6]}-{d[6:]}">CUT): continue
        docs=s.get('application_docs') or []
        if any(x.get('type') in ('Letter','Label') for x in docs):
            items.append({'appl':a.get('application_number'),'suppl':s.get('submission_number'),
                          'date':f"{d[:4]}-{d[4:6]}-{d[6:]}"})

PATS=[re.compile(p, re.I|re.S) for p in [
 r'provides?\s+(?:for\s+)?(.{10,500}?)(?:\.\s|\sAPPROVAL|\sWe have completed)',
 r'supplement(?:al)?[^.]{0,120}?\bis\s+for\s+(.{10,500}?)(?:\.\s|\sAPPROVAL)',
 r'approved?\s+for\s+the\s+following\s+indication[s]?\s*:?\s*(.{10,500}?)(?:\.\s|\sAPPROVAL)',
 r'(?:sNDA|sBLA|supplement(?:al)?[^.]{0,80})\s+(?:updates?|revis\w+)\s+(.{10,500}?)(?:\.\s|\sAPPROVAL)',
]]

# boilerplate: BUKAN deskripsi supplement -> lewati, lanjut cari kecocokan berikutnya
BOILER=re.compile(r'(advisory comments|introductory advertising|promotional labeling|'
                  r'patent information|orange book|to inform the FDA of the\s*$|the following:\s*\d?\s*$)', re.I)

EXCL=[
 ('penghapusan indikasi', r'\b(removal of|remove the|removing|withdraw(?:al|ing)?\s+(?:of\s+)?the\s+(?:following\s+)?indication)'),
 ('limitations of use',   r'limitations?\s+of\s+use'),
 ('perangkat/sediaan',    r'\b(autoinjector|prefilled|pen injector|new (?:dosage )?form|single-dose|vial|syringe|device|co-?packag|presentation of)'),
 ('ekstensi pediatrik',   r'\b(pediatric (?:patients|use|labeling|population|subjects)|section 8\.4|\bPREA\b|neonat|adolescent|years of age and older with active polyarticular)'),
 ('dosis/kekuatan/PK',    r'\b(oral loading|subcutaneous injection during|administration of .{0,40}by |new (?:dosage|dosing|strength)|dose adjustment|dosage regimen|section 2\.\d|pharmacokinetic|\bPBPK\b|bioequival|coadminist|frequency of the liver monitoring)'),
 ('revisi populasi',      r'\b(revising the following indications|to reflect the population|PD-?L1\s*.?\s*\d)'),
 ('keamanan/peringatan',  r'\b(warnings|precautions|adverse reaction|contraindicat|section 5\.\d|section 6\.\d|boxed warning|mortality imbalance)'),
 ('data klinis/label',    r'\b(section 14|clinical studies|section 17|patient counseling|overall survival data|final .{0,20}data|clinical data from Trial|radiographic response section|long-term safety and efficacy data|updates? to (?:the )?(?:USPI|labeling))'),
 ('postmarketing req',    r'\b(postmarketing requirement|subpart h|confirmatory study|analysis and datasets)'),
 ('pembaruan label umum', r'\b(?:updates?|revisions?|modifications?|changes?)\s+to\s+(?:the\s+)?(?:USPI|US Prescribing Information|prescribing information|PI\b|label|[A-Z][a-z]+\s+(?:\([a-z]+\)\s+)?U\.?S)'),
 ('CMC/manufaktur',       r'\b(manufactur|chemistry|facility|container closure|shelf life|stability|storage)'),
]
NEWIND=r'\b(a new indication|an additional indication|the addition of (?:a |an |the )?(?:new )?indicat|(?:expansion|extension) of the indication|for the treatment of|as a diagnostic|the indication of|the use of .{0,80}(?:for|in combination with)|as a new combination regimen)'

def text_of(it):
    for suf in ('.pdf','_label.pdf'):
        fn=os.path.join(PDFDIR, f"{it['appl']}_s{it['suppl']}{suf}")
        if os.path.exists(fn) and os.path.getsize(fn)>5000:
            try:
                o=subprocess.run(['pdftotext','-layout',fn,'-'],capture_output=True,timeout=150)
                t=o.stdout.decode('utf-8','replace')
                if len(t.strip())>=200: return t
            except Exception: pass
    return None

def extract(t):
    """Ambil kecocokan pertama yg bukan boilerplate."""
    for p in PATS:
        for m in p.finditer(t):
            c=m.group(1).strip()
            if len(c)<15: continue
            if BOILER.search(c): continue
            return c
    return ''

print(f"total supplement: {len(items)}", flush=True)
cls=collections.Counter(); cand=[]; review=[]
for i,it in enumerate(items,1):
    t=text_of(it)
    if t is None: cls['GAGAL: tidak ada PDF']+=1; continue
    t=re.sub(r'\s+',' ',t)
    prov=extract(t)
    if not prov:
        cls['pola tidak ditemukan']+=1; review.append((it,'(pola tidak ditemukan)')); continue
    lab=None
    for name,pat in EXCL:
        if re.search(pat, prov, re.I): lab=f'EKSKLUSI: {name}'; break
    if lab is None:
        if re.search(NEWIND, prov, re.I): lab='KANDIDAT pasangan baru'; cand.append((it,prov))
        else: lab='PERLU TINJAUAN MANUAL'; review.append((it,prov))
    cls[lab]+=1
    if i%150==0: print(f"  ...{i}/{len(items)}", flush=True)

n=len(items)
print(f"\n=== DISTRIBUSI v4 (n={n}) ===")
for k,v in cls.most_common(): print(f"  {k:<34} {v:>4} ({100*v/n:.1f}%)")
k=cls['KANDIDAT pasangan baru']; r=cls['PERLU TINJAUAN MANUAL']+cls['pola tidak ditemukan']
print(f"\nKANDIDAT otomatis     : {k} ({100*k/n:.1f}%)   [v3: 95, v4: 110]")
print(f"PERLU tinjauan manual : {r} ({100*r/n:.1f}%)   [v3: 331, v4: 295]")

with open(f'{OUT}/new_pairs_auto.tsv','w') as fh:
    fh.write('appl\tsuppl\tdate\tprovides_for\n')
    for it,p in cand: fh.write(f"{it['appl']}\t{it['suppl']}\t{it['date']}\t{p[:400].replace(chr(9),' ')}\n")
with open(f'{OUT}/manual_review_worklist.tsv','w') as fh:
    fh.write('appl\tsuppl\tdate\tprovides_for\tlabel_BARU_atau_BUKAN\n')
    for it,p in review: fh.write(f"{it['appl']}\t{it['suppl']}\t{it['date']}\t{p[:400].replace(chr(9),' ')}\t\n")
print(f"\nditulis: new_pairs_auto.tsv ({len(cand)}), manual_review_worklist.tsv ({len(review)})")
