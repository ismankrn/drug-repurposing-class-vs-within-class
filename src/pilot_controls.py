"""Kontrol diagnostik: apakah pipeline/model BISA membedakan sama sekali?

Null di pilot utama punya >=3 penjelasan bersaing:
  (a) negatif terlalu sulit  -- kegagalan Phase 3 adalah hard negative
  (b) model 1B base terlalu lemah / tak ikut format yes-no
  (c) skoring log-prob degenerate
Kontrol ini memisahkannya.
"""
import csv, json, random, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL="allenai/OLMo-2-0425-1B"
tok=AutoTokenizer.from_pretrained(MODEL)
model=AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16); model.eval()

TMPL="Question: Is {d} an approved treatment for {s}?\nAnswer:"
def fid(s): return tok.encode(s, add_special_tokens=False)[0]
YES,NO = fid(" Yes"), fid(" No")

@torch.no_grad()
def score(d,s):
    ids=tok(TMPL.format(d=d,s=s), return_tensors='pt')
    lp=torch.log_softmax(model(**ids).logits[0,-1].float(), dim=-1)
    return (lp[YES]-lp[NO]).item()

print("="*62)
print("KONTROL 1: pasangan jelas benar vs jelas absurd")
print("="*62)
obvious=[("aspirin","pain",1),("insulin","diabetes mellitus",1),
         ("metformin","type 2 diabetes mellitus",1),("morphine","severe pain",1),
         ("amoxicillin","bacterial infection",1),
         ("aspirin","bone fracture",0),("insulin","broken leg",0),
         ("metformin","male pattern baldness",0),("morphine","color blindness",0),
         ("amoxicillin","myopia",0)]
for d,s,l in obvious:
    print(f"  [{'BENAR ' if l else 'ABSURD'}] {d:<14} / {s:<28} score={score(d,s):+.3f}")
pos=[score(d,s) for d,s,l in obvious if l]; neg=[score(d,s) for d,s,l in obvious if not l]
print(f"\n  rerata benar={sum(pos)/len(pos):+.3f}  rerata absurd={sum(neg)/len(neg):+.3f}  "
      f"selisih={sum(pos)/len(pos)-sum(neg)/len(neg):+.3f}")
print("  >> bila selisih ~0, model/skoring tidak membedakan APA PUN -> penyebab (b) atau (c)")

print("\n"+"="*62)
print("KONTROL 2: negatif ACAK (mudah) vs negatif Phase 3 (sulit)")
print("="*62)
rows=list(csv.DictReader(open('data/processed/pilot_pairs.tsv'), delimiter='\t'))
P=[r for r in rows if r['label']=='1']
rng=random.Random(5)
drugs=[r['A0'] for r in rows]; dis=[r['disease'] for r in rows]
real={(r['A0'],r['disease']) for r in rows}
rand=[]
while len(rand)<100:
    d,s = rng.choice(drugs), rng.choice(dis)
    if (d,s) not in real: rand.append((d,s))

def auroc(y,s):
    import numpy as np, pandas as pd
    y=np.asarray(y); s=np.asarray(s); r=pd.Series(s).rank().values
    n1=y.sum(); n0=len(y)-n1
    return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)

ps=[score(r['A0'],r['disease']) for r in P]
rs=[score(d,s) for d,s in rand]
y=[1]*len(ps)+[0]*len(rs); sc=ps+rs
print(f"  positif (n={len(ps)}) rerata={sum(ps)/len(ps):+.3f}")
print(f"  negatif ACAK (n={len(rs)}) rerata={sum(rs)/len(rs):+.3f}")
print(f"\n  >> AUROC positif vs negatif ACAK = {auroc(y,sc):.3f}")
print(f"     (bandingkan AUROC vs negatif Phase 3 = 0.479)")
print("     bila jauh >0.5 -> pipeline SEHAT, negatif Phase 3 memang terlalu sulit (a)")
print("     bila tetap ~0.5 -> masalah di model/skoring (b/c)")
