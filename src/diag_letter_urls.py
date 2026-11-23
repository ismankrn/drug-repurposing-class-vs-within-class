"""Diagnosis: mengapa sebagian unduhan surat gagal."""
import json, os, re, subprocess, random, collections

CUT='2023-12-31'
apps=json.load(open('data/raw/openfda/drugsfda_efficacy_apps.json'))
items=[]
for a in apps:
    for s in a.get('submissions',[]):
        if s.get('submission_class_code')!='EFFICACY' or s.get('submission_status')!='AP': continue
        d=str(s.get('submission_status_date') or '')
        if not(len(d)==8 and f"{d[:4]}-{d[4:6]}-{d[6:]}">CUT): continue
        L=next((x.get('url') for x in (s.get('application_docs') or []) if x.get('type')=='Letter'), None)
        if L: items.append({'appl':a.get('application_number'),'suppl':s.get('submission_number'),'url':L})

print(f"total: {len(items)}")
# karakteristik URL
feat=collections.Counter()
for it in items:
    u=it['url']
    if ';' in u: feat['memuat ;']+=1
    if u.startswith('http://'): feat['http:// (bukan https)']+=1
    if len(u)>180: feat['panjang >180 char']+=1
    if ' ' in u: feat['memuat spasi']+=1
print("karakteristik URL:", feat.most_common())

random.Random(42).shuffle(items)
sample=items[:120]
print(f"\n=== uji HTTP status 120 sampel (HEAD) ===")
codes=collections.Counter(); bad=[]
for it in sample:
    r=subprocess.run(['curl','-sS','-o','/dev/null','-w','%{http_code}','-IL','--max-time','45',it['url']],
                     capture_output=True)
    code=r.stdout.decode().strip() or f"rc={r.returncode}"
    err=r.stderr.decode().strip()[:90]
    codes[code]+=1
    if code!='200': bad.append((it['appl'], it['suppl'], code, err, it['url']))
print(dict(codes))
print(f"\n=== {len(bad)} URL bermasalah ===")
for b in bad[:16]:
    print(f"  {b[0]:<12} s{b[1]:<4} [{b[2]}] {b[3]}")
    print(f"       {b[4][:140]}")
