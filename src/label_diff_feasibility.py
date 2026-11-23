"""Cek kelayakan diff label: adakah label PEMBANDING sebelum tiap supplement post-cutoff?"""
import json, collections

CUT='2023-12-31'
apps=json.load(open('data/raw/openfda/drugsfda_efficacy_apps.json'))

def dt(s):
    s=str(s or '')
    return f"{s[:4]}-{s[4:6]}-{s[6:]}" if len(s)==8 else None

def label_url(sub):
    return next((x.get('url') for x in (sub.get('application_docs') or []) if x.get('type')=='Label'), None)

stat=collections.Counter(); pairs=[]; urls=set()
for a in apps:
    subs=[]
    for s in a.get('submissions',[]):
        d=dt(s.get('submission_status_date'))
        if d and s.get('submission_status')=='AP':
            subs.append({'d':d,'num':s.get('submission_number'),
                         'cls':s.get('submission_class_code'),'url':label_url(s)})
    subs.sort(key=lambda x:(x['d'], str(x['num'])))
    for i,s in enumerate(subs):
        if s['cls']!='EFFICACY' or s['d']<=CUT: continue
        stat['supplement post-cutoff']+=1
        if not s['url']: stat['  tanpa label sendiri']+=1; continue
        prev=next((p for p in reversed(subs[:i]) if p['url']), None)
        if not prev: stat['  tanpa label pembanding']+=1; continue
        stat['  LAYAK di-diff']+=1
        pairs.append({'appl':a.get('application_number'),'suppl':s['num'],'date':s['d'],
                      'new_url':s['url'],'old_url':prev['url'],
                      'old_date':prev['d'],'old_suppl':prev['num']})
        urls.add(s['url']); urls.add(prev['url'])

for k,v in stat.most_common(): print(f"  {k:<26} {v}")
print(f"\nPDF unik yg perlu diunduh : {len(urls)}")
print(f"perkiraan ukuran (~1,2 MB) : ~{len(urls)*1.2/1000:.1f} GB")
print(f"aplikasi unik              : {len({p['appl'] for p in pairs})}")

gap=[]
from datetime import date
for p in pairs:
    y1,m1,d1=map(int,p['old_date'].split('-')); y2,m2,d2=map(int,p['date'].split('-'))
    gap.append((date(y2,m2,d2)-date(y1,m1,d1)).days)
gap.sort()
print(f"\njarak waktu ke label pembanding (hari):")
print(f"  median={gap[len(gap)//2]}  q1={gap[len(gap)//4]}  q3={gap[3*len(gap)//4]}  maks={gap[-1]}")
print(f"  <=180 hari: {sum(1 for g in gap if g<=180)}/{len(gap)} ({100*sum(1 for g in gap if g<=180)/len(gap):.0f}%)")
print("  (jarak pendek = lebih sedikit perubahan tak-terkait, diff lebih bersih)")

json.dump(pairs, open('data/processed/label_diff_pairs.json','w'))
print(f"\nditulis: data/processed/label_diff_pairs.json ({len(pairs)} pasang label)")
print("\ncontoh 3:")
for p in pairs[:3]:
    print(f"  {p['appl']} s{p['suppl']} ({p['date']}) <- s{p['old_suppl']} ({p['old_date']})")
