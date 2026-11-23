"""Bangkitkan pseudonim A4 (format kode pengembangan) + verifikasi absen dari Dolma.

Format meniru pola ALPHA-NUM nyata (IMMU-132, LOXO-292) agar in-distribution,
tapi TANPA stem INN (yg membawa makna kelas farmakologis -> kebocoran).
"""
import json, sys, time, urllib.request, random, string, re

API="https://api.infini-gram.io/"
INDEX="v4_dolma-v1_7_llama"

def count(q, tries=3):
    last="?"
    # CATATAN: max_clause_freq hanya valid untuk query CNF (X AND Y).
    # Mengirimnya pada count tunggal -> "Processor.count() got an unexpected keyword argument".
    body=json.dumps({"index":INDEX,"query_type":"count","query":q}).encode()
    req=urllib.request.Request(API, data=body, headers={"Content-Type":"application/json"})
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                d=json.loads(r.read().decode())
            if "count" in d: return d["count"], d.get("approx")
            return None, d.get("error")
        except Exception as e:
            last=f"{type(e).__name__}: {e}"
            time.sleep(2*(i+1))
    return None, last

# prefiks 4-huruf yg sengaja tak lazim (hindari kode perusahaan nyata)
BANNED={'IMMU','INCB','LOXO','MEDI','ASTX','BMS','MOR','TRV','INC','PM','UX','DCC','NS','ABT','GSK','AZD','MK','BI','LY','PF'}
rng=random.Random(20261030)
CONS='BCDFGHJKLMNPQRSTVWXZ'

def gen(n):
    out=[]
    seen=set()
    while len(out)<n:
        p=''.join(rng.choice(CONS) for _ in range(4))
        if p in BANNED or p in seen: continue
        seen.add(p)
        out.append(f"{p}-{rng.randint(100,999)}")
    return out

cands=gen(int(sys.argv[1]) if len(sys.argv)>1 else 8)
print(f"menguji {len(cands)} kandidat pseudonim terhadap Dolma-v1.7\n")
clean, dirty = [], []
for c in cands:
    n, extra = count(c)
    status = "ABSEN" if n==0 else (f"ADA n={n}" if n is not None else f"ERR {extra}")
    print(f"  {c:<12} {status}", flush=True)
    (clean if n==0 else dirty).append((c,n))
    time.sleep(0.2)

print(f"\nabsen total : {len(clean)}/{len(cands)}")
print(f"terpakai    : {[c for c,_ in clean][:12]}")
if dirty:
    print(f"ditolak     : {[(c,n) for c,n in dirty]}")
print("\n-> hanya pseudonim ber-count 0 yg boleh dipakai sebagai A4.")
