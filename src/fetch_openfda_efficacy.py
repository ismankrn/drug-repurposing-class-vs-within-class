"""Cache seluruh aplikasi Drugs@FDA yang punya >=1 supplement EFFICACY."""
import json, sys, time, urllib.parse, urllib.request, os

BASE="https://api.fda.gov/drug/drugsfda.json"
OUT="data/raw/openfda/drugsfda_efficacy_apps.json"

def get(params, tries=5):
    url=BASE+"?"+urllib.parse.urlencode(params)
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            print(f"  [retry {i+1}] {type(e).__name__}", file=sys.stderr); time.sleep(4*(i+1))
    return None

if os.path.exists(OUT):
    print(f"cache sudah ada: {OUT}"); sys.exit(0)

apps, skip, LIMIT = [], 0, 1000
while True:
    d=get({"search":'submissions.submission_class_code:"EFFICACY"',"limit":LIMIT,"skip":skip})
    if not d or "results" not in d: break
    apps.extend(d["results"]); total=d["meta"]["results"]["total"]; skip+=LIMIT
    print(f"  {len(apps)}/{total}", file=sys.stderr)
    if skip>=total or skip>25000: break
    time.sleep(1)

with open(OUT,"w") as fh: json.dump(apps, fh)
print(f"disimpan {len(apps)} aplikasi -> {OUT}")
