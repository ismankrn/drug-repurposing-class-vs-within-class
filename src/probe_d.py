"""Probe D: frekuensi korpus Dolma utk sampel pasangan obat-penyakit.

Perubahan dari versi yg memicu blokir (Temuan: konkurensi 12 -> diblokir):
  - KONKURENSI 1, jeda 0,4 dtk antar permintaan
  - BERHENTI TOTAL pada 403 pertama (tidak retry, tidak lanjut)
  - cache append-only JSONL -> bisa dilanjutkan bila terputus
  - sampel terstratifikasi, bukan seluruh pasangan

Pertanyaan (versi baru, sesuai temuan jurang kelas-vs-obat):
  apakah performa tingkat-kelas melacak frekuensi korpus, sedangkan
  resolusi intra-kelas tidak?
"""
import csv, json, os, random, re, sys, time, urllib.request

API="https://api.infini-gram.io/"; INDEX="v4_dolma-v1_7_llama"
DELAY=0.4
IN=sys.argv[1] if len(sys.argv)>1 else 'data/processed/pilot_pairs_v5.tsv'
OUT=sys.argv[2] if len(sys.argv)>2 else 'data/processed/probe_d_v5.tsv'
N=int(sys.argv[3]) if len(sys.argv)>3 else 3000
CACHE='data/interim/ig_cache.jsonl'
os.makedirs('data/interim', exist_ok=True)

cache={}
if os.path.exists(CACHE):
    for line in open(CACHE):
        try: o=json.loads(line); cache[o['k']]=o['v']
        except Exception: pass
print(f"cache awal: {len(cache)}", flush=True)
fh_cache=open(CACHE,'a')

class Blocked(Exception): pass

def api(payload):
    k=json.dumps(payload, sort_keys=True)
    if k in cache: return cache[k]
    body=json.dumps(payload).encode()
    for i in range(3):
        try:
            req=urllib.request.Request(API, data=body, headers={"Content-Type":"application/json"})
            with urllib.request.urlopen(req, timeout=90) as r:
                raw=r.read().decode()
            d=json.loads(raw)
            if isinstance(d,dict) and str(d.get('message','')).lower()=='forbidden':
                raise Blocked("API membalas Forbidden")
            cache[k]=d; fh_cache.write(json.dumps({'k':k,'v':d})+'\n'); fh_cache.flush()
            time.sleep(DELAY)
            return d
        except Blocked: raise
        except urllib.error.HTTPError as e:
            if e.code in (403,429): raise Blocked(f"HTTP {e.code}")
            time.sleep(2*(i+1))
        except Exception:
            time.sleep(2*(i+1))
    return {"error":"habis percobaan"}

def count1(t): return api({"index":INDEX,"query_type":"count","query":t})
def count2(a,b): return api({"index":INDEX,"query_type":"count",
                             "query":f"{a} AND {b}","max_clause_freq":500000})

STOP={'of','the','and','with','due','to','in','a','an','nos','unspecified'}
def variants(name):
    v=[name]
    s=re.sub(r'\s*,\s*(unspecified|NOS).*$','',name,flags=re.I)
    if s!=name: v.append(s)
    s2=re.sub(r'\b(stage|grade)\s+[IVX0-9]+\b','',s,flags=re.I).strip(' ,')
    if s2 and s2 not in v: v.append(s2)
    w=[x for x in re.findall(r"[A-Za-z][A-Za-z\-']+", s) if x.lower() not in STOP]
    if len(w)>=2:
        c=' '.join(sorted(w,key=len,reverse=True)[:2])
        if c not in v: v.append(c)
    return v[:3]

rows=[r for r in csv.DictReader(open(IN), delimiter='\t') if r.get('A0') and r.get('disease')]
rng=random.Random(20261102)
# stratifikasi per tier agar proporsinya terjaga
byt={}
for r in rows: byt.setdefault(r.get('tier','?'),[]).append(r)
sample=[]
for t,g in byt.items():
    rng.shuffle(g); sample += g[:max(1,int(N*len(g)/len(rows)))]
rng.shuffle(sample)
print(f"populasi {len(rows)} -> sampel {len(sample)}  tier={ {t:sum(1 for r in sample if r.get('tier')==t) for t in byt} }", flush=True)

out=[]; t0=time.time()
try:
    for i,r in enumerate(sample,1):
        drug, dis = r['A0'], r['disease']
        dn=count1(drug).get('count')
        best=(0,None,None)
        for form in variants(dis):
            d=count2(drug, form); c=d.get('count')
            if c is None: continue
            if c>0: best=(c,form,d.get('approx')); break
            if best[1] is None: best=(0,form,d.get('approx'))
        sn=count1(best[1] or dis).get('count')
        out.append({**r,'drug_count':dn,'disease_count':sn,'pair_count':best[0],
                    'disease_form':best[1],'pair_approx':best[2]})
        if i%100==0:
            el=time.time()-t0
            print(f"  {i}/{len(sample)}  {el/60:.1f} mnt  sisa ~{(len(sample)-i)*el/i/60:.0f} mnt", flush=True)
except Blocked as e:
    print(f"\n!! DIHENTIKAN: {e}", flush=True)
    print(f"   {len(out)} pasangan selesai; cache tersimpan, job bisa dilanjutkan nanti.", flush=True)

if out:
    with open(OUT,'w',newline='') as fh:
        w=csv.DictWriter(fh, fieldnames=list(out[0].keys()), delimiter='\t')
        w.writeheader(); w.writerows(out)
    ok=[r for r in out if isinstance(r['pair_count'],int)]
    z=[r for r in ok if r['pair_count']==0]
    c=sorted(r['pair_count'] for r in ok)
    print(f"\nselesai {len(out)} dlm {(time.time()-t0)/60:.0f} mnt")
    print(f"  pair_count 0 : {len(z)} ({100*len(z)/max(len(ok),1):.0f}%)")
    print(f"  sebaran: min={c[0]} q1={c[len(c)//4]} med={c[len(c)//2]} q3={c[3*len(c)//4]} max={c[-1]}")
    print(f"ditulis: {OUT}")
