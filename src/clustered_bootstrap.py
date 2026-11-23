"""Bootstrap berkluster: CI jujur yang memperhitungkan non-independensi pasangan.

Masalah: satu obat muncul di banyak pasangan, satu penyakit juga. Rumus
Hanley-McNeil mengasumsikan independensi -> CI terlalu sempit.

Perbaikan: resample pada tingkat OBAT (klaster), ambil semua pasangannya,
hitung ulang AUROC. 2.000 iterasi, persentil 2,5/97,5.
Sensitivitas: diulang dengan klaster PENYAKIT.
"""
import json, sys, numpy as np, pandas as pd
from collections import defaultdict

B = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
rng = np.random.default_rng(20261102)

def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = pd.Series(s).rank().values
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def naive_ci(a, n1, n0):
    q1 = a / (2 - a); q2 = 2 * a * a / (1 + a)
    se = ((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0)) ** .5
    return a - 1.96 * se, a + 1.96 * se

def boot(df, cluster_col, score_col='score', B=B):
    """AUROC bootstrap, resample pada tingkat klaster."""
    groups = defaultdict(list)
    for i, c in enumerate(df[cluster_col].values): groups[c].append(i)
    keys = list(groups); idx = {k: np.array(v) for k, v in groups.items()}
    y = df['label'].values; s = df[score_col].values
    out = []
    for _ in range(B):
        pick = rng.choice(len(keys), len(keys), replace=True)
        sel = np.concatenate([idx[keys[p]] for p in pick])
        a = auroc(y[sel], s[sel])
        if not np.isnan(a): out.append(a)
    return np.array(out)

def load(f, within):
    d = pd.DataFrame([json.loads(l) for l in open(f)])
    if within: g = d[d.condition == 'A0']
    else:
        g = pd.concat([d[d.tier == 'pos'], d[d.tier == 'neg_random']])
        g = g[g.condition == 'A0']
    return g.reset_index(drop=True)

E = {'1B':'results/easy_OLMo-2-0425-1B.jsonl','7B':'results/scores_5cond.jsonl',
     '13B':'results/easy_OLMo-2-1124-13B-Instruct.jsonl','32B':'results/easy_OLMo-2-0325-32B-Instruct.jsonl'}
H = {'1B':'results/within_OLMo-2-0425-1B.jsonl','7B':'results/scores_within.jsonl',
     '13B':'results/within_OLMo-2-1124-13B-Instruct.jsonl','32B':'results/within_OLMo-2-0325-32B-Instruct.jsonl'}

print("=" * 86)
print(f"CI NAIF vs BOOTSTRAP BERKLUSTER (klaster = obat, B={B})")
print("=" * 86)
print(f"  {'model':<6} {'tugas':<12} {'AUROC':>7}  {'CI naif':<18} {'CI berkluster (obat)':<22} {'lebar'}")
store = {}
for task, FILES, within in (('kelas', E, False), ('intra-kelas', H, True)):
    for m in ['1B','7B','13B','32B']:
        g = load(FILES[m], within)
        a = auroc(g['label'], g['score'])
        n1 = int(g['label'].sum()); n0 = len(g) - n1
        nlo, nhi = naive_ci(a, n1, n0)
        bs = boot(g, 'drugbank_id')
        blo, bhi = np.percentile(bs, [2.5, 97.5])
        store[(task, m)] = bs
        ratio = (bhi - blo) / (nhi - nlo)
        print(f"  {m:<6} {task:<12} {a:>7.3f}  [{nlo:.3f}-{nhi:.3f}]   [{blo:.3f}-{bhi:.3f}]       {ratio:.1f}x")

print("\n" + "=" * 86)
print("JURANG (kelas - intra-kelas), CI bootstrap berkluster")
print("=" * 86)
for m in ['1B','7B','13B','32B']:
    d = store[('kelas', m)] - store[('intra-kelas', m)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    sig = "ya" if lo > 0 else "TIDAK"
    print(f"  {m:<6} jurang={d.mean():+.3f}  [{lo:+.3f}, {hi:+.3f}]   >0 signifikan: {sig}")

print("\n" + "=" * 86)
print("UJI KUNCI: 32B intra-kelas vs baseline derajat obat")
print("=" * 86)
g = load(H['32B'], True)
mdl = boot(g, 'drugbank_id')
# baseline derajat pada klaster yg sama
gd = g.copy(); gd['score'] = gd['drug_degree'].astype(float)
bas = boot(gd, 'drugbank_id')
d = mdl - bas
lo, hi = np.percentile(d, [2.5, 97.5])
print(f"  model 32B        : {mdl.mean():.3f}  [{np.percentile(mdl,2.5):.3f}-{np.percentile(mdl,97.5):.3f}]")
print(f"  baseline derajat : {bas.mean():.3f}  [{np.percentile(bas,2.5):.3f}-{np.percentile(bas,97.5):.3f}]")
print(f"  SELISIH          : {d.mean():+.3f}  [{lo:+.3f}, {hi:+.3f}]   >0 signifikan: {'ya' if lo>0 else 'TIDAK'}")

print("\n" + "=" * 86)
print("SENSITIVITAS: klaster = PENYAKIT")
print("=" * 86)
for m in ['7B','32B']:
    for task, FILES, within in (('kelas', E, False), ('intra-kelas', H, True)):
        g = load(FILES[m], within)
        bs = boot(g, 'umls_cui')
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"  {m:<5} {task:<12} [{lo:.3f}-{hi:.3f}]")
