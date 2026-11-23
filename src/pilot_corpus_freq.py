"""Kumpulkan frekuensi korpus Dolma utk pasangan pilot (Probe D, bagian D1).

Untuk tiap pasangan: hitung dokumen yg memuat nama obat DAN nama penyakit.
Catatan dari Temuan N: max_clause_freq hanya sah utk query CNF, bukan count tunggal.
Nama penyakit panjang sering ber-count 0; disediakan fallback bentuk ringkas.
"""
import csv, json, os, re, sys, time, urllib.request

API="https://api.infini-gram.io/"; INDEX="v4_dolma-v1_7_llama"
IN='data/processed/pilot_pairs.tsv'; OUT='data/processed/pilot_corpus_freq.tsv'
CACHE='data/interim/infinigram_cache.json'

cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}

def query(payload, tries=4):
    key=json.dumps(payload, sort_keys=True)
    if key in cache: return cache[key]
    body=json.dumps(payload).encode()
    for i in range(tries):
        try:
            req=urllib.request.Request(API, data=body, headers={"Content-Type":"application/json"})
            with urllib.request.urlopen(req, timeout=90) as r:
                d=json.loads(r.read().decode())
            cache[key]=d; return d
        except Exception as e:
            if i==tries-1: return {"error":f"{type(e).__name__}: {e}"}
            time.sleep(2*(i+1))
    return {"error":"habis percobaan"}

def count_single(term):
    d=query({"index":INDEX,"query_type":"count","query":term})
    return d.get("count"), d.get("approx"), d.get("error")

def count_pair(a,b):
    d=query({"index":INDEX,"query_type":"count","query":f"{a} AND {b}",
             "max_clause_freq":500000})
    return d.get("count"), d.get("approx"), d.get("error")

STOP={'of','the','and','with','due','to','in','a','an','disease','disorder','syndrome','stage','nos'}
def simplify(name):
    """Bentuk ringkas: 2 kata konten terpanjang, utk fallback bila nama penuh ber-count 0."""
    w=[x for x in re.findall(r"[A-Za-z][A-Za-z\-']+", name) if x.lower() not in STOP]
    w=sorted(w, key=len, reverse=True)[:2]
    return ' '.join(w) if w else name

rows=list(csv.DictReader(open(IN), delimiter='\t'))
print(f"pasangan pilot: {len(rows)}", flush=True)
out=[]
for i,r in enumerate(rows,1):
    drug, dis = r['A0'], r['disease']
    dn,da,de = count_single(drug)
    sn,sa,se = count_single(dis)
    pn,pa,pe = count_pair(drug, dis)
    used=dis; fallback=0
    if pn==0:
        dis2=simplify(dis)
        if dis2.lower()!=dis.lower():
            p2,a2,e2 = count_pair(drug, dis2)
            if p2 and p2>0: pn,pa,pe,used,fallback = p2,a2,e2,dis2,1
    out.append({**r,'drug_count':dn,'disease_count':sn,'pair_count':pn,
                'pair_approx':pa,'disease_used':used,'used_fallback':fallback,
                'err':de or se or pe or ''})
    if i%25==0:
        json.dump(cache, open(CACHE,'w'))
        print(f"  ...{i}/{len(rows)}", flush=True)

json.dump(cache, open(CACHE,'w'))
with open(OUT,'w',newline='') as fh:
    w=csv.DictWriter(fh, fieldnames=list(out[0].keys()), delimiter='\t')
    w.writeheader(); w.writerows(out)

ok=[r for r in out if isinstance(r['pair_count'],int)]
zeros=[r for r in ok if r['pair_count']==0]
appr=[r for r in ok if r['pair_approx']]
fb=[r for r in ok if r['used_fallback']]
print(f"\nberhasil   : {len(ok)}/{len(out)}")
print(f"pair_count 0: {len(zeros)} ({100*len(zeros)/max(len(ok),1):.0f}%)")
print(f"aproksimasi : {len(appr)} ({100*len(appr)/max(len(ok),1):.0f}%)")
print(f"pakai fallback nama ringkas: {len(fb)}")
if ok:
    c=sorted(r['pair_count'] for r in ok)
    print(f"sebaran pair_count: min={c[0]} q1={c[len(c)//4]} median={c[len(c)//2]} q3={c[3*len(c)//4]} max={c[-1]}")
print(f"\nditulis: {OUT}")
