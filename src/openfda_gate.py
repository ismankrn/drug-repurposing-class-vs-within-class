"""Hitung submission EFFICACY sisi-klien (hindari jebakan agregasi nested openFDA)."""
import json, sys, time, urllib.parse, urllib.request, collections

BASE = "https://api.fda.gov/drug/drugsfda.json"

def get(params, tries=4):
    url = BASE + "?" + urllib.parse.urlencode(params)
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            print(f"   [retry {i+1}] {type(e).__name__}", file=sys.stderr)
            time.sleep(3 * (i + 1))
    return None

apps, skip, LIMIT = [], 0, 1000
while True:
    d = get({"search": 'submissions.submission_class_code:"EFFICACY"',
             "limit": LIMIT, "skip": skip})
    if not d or "results" not in d:
        break
    apps.extend(d["results"])
    total = d["meta"]["results"]["total"]
    skip += LIMIT
    print(f"  diambil {len(apps)}/{total}", file=sys.stderr)
    if skip >= total or skip > 25000:
        break
    time.sleep(1)

print(f"aplikasi dgn >=1 supplement EFFICACY: {len(apps)}")

eff_by_year = collections.Counter()
all_by_year = collections.Counter()
eff_apps_by_year = collections.defaultdict(set)
n_eff = n_all = 0
for a in apps:
    appl = a.get("application_number")
    for s in a.get("submissions", []):
        n_all += 1
        y = str(s.get("submission_status_date") or "")[:4]
        all_by_year[y] += 1
        if s.get("submission_class_code") == "EFFICACY" and s.get("submission_status") == "AP":
            n_eff += 1
            eff_by_year[y] += 1
            eff_apps_by_year[y].add(appl)

print(f"total submission pada aplikasi tsb : {n_all}")
print(f"di antaranya EFFICACY + disetujui  : {n_eff}")
print(f"rasio (menjelaskan selisih 18x?)   : {n_all/max(n_eff,1):.1f}x\n")

print("EFFICACY disetujui per tahun (submission | aplikasi unik):")
for y in sorted(eff_by_year):
    if y >= "2020":
        print(f"  {y}: {eff_by_year[y]:>4} | {len(eff_apps_by_year[y]):>4}")
for lo in ("2024", "2025", "2026"):
    s = sum(v for k, v in eff_by_year.items() if k >= lo)
    a = len(set().union(*[eff_apps_by_year[k] for k in eff_apps_by_year if k >= lo]) or set())
    print(f"  kumulatif >= {lo}: {s} submission | {a} aplikasi unik")
