"""Probe D skala penuh: frekuensi korpus Dolma utk SELURUH pasangan repoDB.

Efisiensi: hitungan obat dan penyakit dipakai ulang lintas pasangan
  1.571 obat + ~2.000 penyakit + 10.800 pasangan = ~14.400 query, bukan 32.400.

Kesopanan: konkurensi dibatasi 12. infini-gram adalah layanan riset publik
gratis; dokumentasinya menyatakan uptime tidak dijamin. Probe D sepenuhnya
bergantung padanya, jadi tidak sepadan mempercepat dgn risiko diblokir.

Cakupan: nama penyakit panjang sering ber-count 0 (46% pd pilot). Dicoba
beberapa bentuk permukaan berurutan sampai dapat >0; bentuk yg dipakai dicatat.
"""
import csv, json, os, re, sys, threading, time, urllib.request, collections
from concurrent.futures import ThreadPoolExecutor, as_completed

API="https://api.infini-gram.io/"; INDEX="v4_dolma-v1_7_llama"
CONC=12
OUT='data/processed/corpus_freq_full.tsv'
CACHE='data/interim/infinigram_cache.jsonl'

_lock=threading.Lock()
cache={}
if os.path.exists(CACHE):
    for line in open(CACHE):
        try:
            o=json.loads(line); cache[o['k']]=o['v']
        except Exception: pass
print(f"cache awal: {len(cache)} entri", flush=True)
_fh=open(CACHE,'a')

def api(payload):
    k=json.dumps(payload, sort_keys=True)
    with _lock:
        if k in cache: return cache[k]
    body=json.dumps(payload).encode()
    last=None
    for i in range(4):
        try:
            req=urllib.request.Request(API, data=body, headers={"Content-Type":"application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                d=json.loads(r.read().decode())
            with _lock:
                cache[k]=d; _fh.write(json.dumps({'k':k,'v':d})+'\n')
            return d
        except Exception as e:
            last=f"{type(e).__name__}"; time.sleep(3*(i+1))
    return {"error":last}

def count1(term):
    return api({"index":INDEX,"query_type":"count","query":term})
def count2(a,b):
    return api({"index":INDEX,"query_type":"count","query":f"{a} AND {b}","max_clause_freq":500000})

STOP={'of','the','and','with','due','to','in','a','an','nos','unspecified'}
def variants(name):
    """Bentuk permukaan penyakit, dari spesifik ke umum."""
    v=[name]
    s=re.sub(r'\s*,\s*(unspecified|NOS).*$','',name, flags=re.I)
    if s!=name: v.append(s)
    s2=re.sub(r'\b(stage|grade)\s+[IVX0-9]+\b','',s, flags=re.I).strip(' ,')
    if s2 and s2 not in v: v.append(s2)
    w=[x for x in re.findall(r"[A-Za-z][A-Za-z\-']+", s) if x.lower() not in STOP]
    if len(w)>=2:
        core=' '.join(sorted(w, key=len, reverse=True)[:2])
        if core not in v: v.append(core)
    return v[:4]

# ---- muat pasangan ----
nv={r['drugbank_id']:r['A0_canonical'] for r in csv.DictReader(open('data/processed/name_variants.tsv'), delimiter='\t')}
with open('data/raw/repodb/full.csv', newline='', encoding='utf-8-sig') as fh:
    rows=[r for r in csv.DictReader(fh)]
pairs=[]
for r in rows:
    a0=nv.get(r['drug_id'])
    if a0 and r['ind_name']: pairs.append((r['drug_id'], a0, r['ind_id'], r['ind_name'], r['status'], r['phase']))
print(f"pasangan: {len(pairs)} | obat unik: {len({p[1] for p in pairs})} | penyakit unik: {len({p[3] for p in pairs})}", flush=True)

t0=time.time(); done=0
def work(p):
    global done
    db,drug,cui,dis,status,phase=p
    dn=count1(drug).get('count')
    best=(0,None,None)
    for form in variants(dis):
        d=count2(drug, form)
        c=d.get('count')
        if c is None: continue
        if c>0: best=(c, form, d.get('approx')); break
        if best[1] is None: best=(0, form, d.get('approx'))
    sn=count1(best[1] or dis).get('count')
    with _lock:
        done+=1
        if done%500==0:
            el=time.time()-t0
            print(f"  {done}/{len(pairs)}  {el/60:.0f} mnt  {done/max(el,1):.1f}/dtk  cache={len(cache)}", flush=True)
    return {'drugbank_id':db,'A0':drug,'umls_cui':cui,'disease':dis,
            'status':status,'phase':phase,'drug_count':dn,'disease_count':sn,
            'pair_count':best[0],'disease_form_used':best[1],'pair_approx':best[2]}

res=[]
with ThreadPoolExecutor(max_workers=CONC) as ex:
    futs=[ex.submit(work,p) for p in pairs]
    for f in as_completed(futs):
        try: res.append(f.result())
        except Exception as e: print("  gagal:", e, flush=True)
_fh.close()

with open(OUT,'w',newline='') as fh:
    w=csv.DictWriter(fh, fieldnames=list(res[0].keys()), delimiter='\t')
    w.writeheader(); w.writerows(res)

ok=[r for r in res if isinstance(r['pair_count'],int)]
z=[r for r in ok if r['pair_count']==0]
fb=[r for r in ok if r['disease_form_used']!=r['disease']]
ap=[r for r in ok if r['pair_approx']]
c=sorted(r['pair_count'] for r in ok)
print(f"\n=== SELESAI dalam {(time.time()-t0)/60:.0f} menit ===")
print(f"  berhasil        : {len(ok)}/{len(res)}")
print(f"  pair_count 0    : {len(z)} ({100*len(z)/max(len(ok),1):.0f}%)   [pilot: 46%]")
print(f"  pakai bentuk alt: {len(fb)} ({100*len(fb)/max(len(ok),1):.0f}%)")
print(f"  aproksimasi     : {len(ap)} ({100*len(ap)/max(len(ok),1):.0f}%)")
print(f"  sebaran: min={c[0]} q1={c[len(c)//4]} median={c[len(c)//2]} q3={c[3*len(c)//4]} maks={c[-1]}")
print(f"\nditulis: {OUT}")
