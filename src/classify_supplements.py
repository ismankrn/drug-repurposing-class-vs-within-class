"""Ekstrak kalimat 'provides for ...' dari surat persetujuan FDA, lalu klasifikasikan.

Menjawab: dari supplement EFFICACY post-cutoff, berapa yg benar2 INDIKASI BARU?
Pakai: python3 -I src/classify_supplements.py [n_sampel]
"""
import json, os, re, sys, time, subprocess, urllib.request, collections, random

CUT='2023-12-31'
PDFDIR='data/raw/fda_letters'
os.makedirs(PDFDIR, exist_ok=True)

apps=json.load(open('data/raw/openfda/drugsfda_efficacy_apps.json'))
items=[]
for a in apps:
    for s in a.get('submissions',[]):
        if s.get('submission_class_code')!='EFFICACY' or s.get('submission_status')!='AP': continue
        d=str(s.get('submission_status_date') or '')
        if not(len(d)==8 and f"{d[:4]}-{d[4:6]}-{d[6:]}">CUT): continue
        letter=next((x.get('url') for x in (s.get('application_docs') or []) if x.get('type')=='Letter'), None)
        if letter:
            items.append({'appl':a.get('application_number'),'suppl':s.get('submission_number'),
                          'date':f"{d[:4]}-{d[4:6]}-{d[6:]}",'url':letter})
print(f"supplement EFFICACY post-cutoff dgn surat: {len(items)}")

n=int(sys.argv[1]) if len(sys.argv)>1 else 25
random.Random(42).shuffle(items)
sample=items[:n]

PROV=re.compile(r'(?:This|The)\s+[^.]{0,120}?(?:application|sNDA|sBLA|supplement)[^.]{0,80}?provides?\s+for\s+(.{10,400}?)\.\s',
                re.I|re.S)

RULES=[
 ('INDIKASI BARU', r'(new indication|a new indication|the treatment of|treatment of (?:adult|pediatric|patients)|an additional indication|expand(?:ed|ing)? the indication|indication for)'),
 ('populasi/pediatrik', r'(pediatric use|section 8\.4|pediatric patients aged|age range|neonat|adolescent)'),
 ('dosis/regimen', r'(dosage|dosing|regimen|section 2\.|administration|strength)'),
 ('keamanan/label lain', r'(warnings|precautions|adverse|contraindicat|section 5\.|section 6\.|boxed)'),
]

def get_text(it):
    fn=os.path.join(PDFDIR, f"{it['appl']}_s{it['suppl']}.pdf")
    if not os.path.exists(fn):
        try:
            req=urllib.request.Request(it['url'], headers={'User-Agent':'research-script'})
            with urllib.request.urlopen(req, timeout=90) as r, open(fn,'wb') as fh:
                fh.write(r.read())
        except Exception as e:
            return None, f"unduh gagal: {type(e).__name__}"
        time.sleep(0.5)
    try:
        out=subprocess.run(['pdftotext','-layout',fn,'-'],capture_output=True,timeout=120)
        return out.stdout.decode('utf-8','replace'), None
    except Exception as e:
        return None, f"pdftotext gagal: {e}"

cls=collections.Counter(); rows=[]
for i,it in enumerate(sample,1):
    txt,err=get_text(it)
    if err: cls['GAGAL']+=1; print(f"  [{i}/{n}] {it['appl']} s{it['suppl']}: {err}", flush=True); continue
    t=re.sub(r'\s+',' ',txt)
    m=PROV.search(t)
    prov=m.group(1).strip() if m else ''
    label='TIDAK TERKLASIFIKASI'
    for name,pat in RULES:
        if prov and re.search(pat, prov, re.I): label=name; break
    if not prov: label='POLA TIDAK DITEMUKAN'
    cls[label]+=1
    rows.append((it,label,prov))
    print(f"  [{i}/{n}] {it['appl']:<12} s{it['suppl']:<4} {it['date']} [{label}]", flush=True)
    if prov: print(f"        \"{prov[:150]}\"", flush=True)

print(f"\n=== DISTRIBUSI KLASIFIKASI (n={n}) ===")
for k,v in cls.most_common(): print(f"  {k:<24} {v:>3} ({100*v/n:.0f}%)")
ind=cls['INDIKASI BARU']
print(f"\n=> proporsi INDIKASI BARU: {ind}/{n} = {100*ind/n:.0f}%")
print(f"=> ekstrapolasi ke {len(items)} supplement: ~{round(len(items)*ind/n)} indikasi baru")
