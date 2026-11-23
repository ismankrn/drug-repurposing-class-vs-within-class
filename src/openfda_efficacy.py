"""Hitung supplement EFFICACY per tahun dari openFDA Drugs@FDA."""
import json, sys, time, urllib.parse, urllib.request, collections

BASE = "https://api.fda.gov/drug/drugsfda.json"

def get(params, tries=4):
    url = BASE + "?" + urllib.parse.urlencode(params)
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            body = ""
            if hasattr(e, "read"):
                try: body = e.read().decode()[:200]
                except Exception: pass
            print(f"   [retry {i+1}] {type(e).__name__} {e} {body}", file=sys.stderr)
            time.sleep(3 * (i + 1))
    return None

d = get({"search": 'submissions.submission_class_code:"EFFICACY"',
         "count": "submissions.submission_status_date"})
if not d or "results" not in d:
    print("GAGAL:", json.dumps(d)[:300] if d else "tidak ada respons"); sys.exit(1)

by = collections.Counter()
for x in d["results"]:
    key = x.get("time") or x.get("term") or ""
    by[str(key)[:4]] += x.get("count", 0)

print("supplement EFFICACY per tahun:")
total = sum(by.values())
for y in sorted(by):
    if y >= "2015":
        print(f"  {y}: {by[y]}")
print(f"  --- total semua tahun: {total}")
for lo in ("2024", "2025", "2026"):
    print(f"  kumulatif >= {lo}: {sum(v for k, v in by.items() if k >= lo)}")
