"""Pilot inferensi: OLMo-2 1B, kondisi A0 vs A4, skoring log-probabilitas.

Skoring memakai log-prob token jawaban (bukan parsing teks) -- lihat runbook §2.2:
tidak ada kegagalan parsing, menghasilkan skor kontinu utk AUROC, dan tahan
terhadap perbedaan gaya luaran antar model.

Pakai: python3 src/pilot_inference.py [model_id] [dtype]
"""
import csv, json, os, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = sys.argv[1] if len(sys.argv)>1 else "allenai/OLMo-2-0425-1B"
DTYPE = sys.argv[2] if len(sys.argv)>2 else "bfloat16"
IN    = sys.argv[3] if len(sys.argv)>3 else 'data/processed/pilot_pairs.tsv'
OUT   = sys.argv[4] if len(sys.argv)>4 else 'results/pilot_scores.jsonl'
os.makedirs('results', exist_ok=True)

# 3 template -> varians prompt dilaporkan, bukan hanya yg terbaik (runbook §2.1)
TEMPLATES=[
 ("qa",    "Question: Is {d} an approved treatment for {s}?\nAnswer:", " Yes", " No"),
 ("tf",    "{d} is used to treat {s}.\nThis statement is:", " True", " False"),
 ("cloze", "Medical fact: the drug {d} is indicated for the treatment of {s}.\nCorrect:", " yes", " no"),
]

print(f"memuat {MODEL} dtype={DTYPE}", flush=True)
t0=time.time()
tok=AutoTokenizer.from_pretrained(MODEL)
dt=getattr(torch, DTYPE)
model=AutoModelForCausalLM.from_pretrained(MODEL, dtype=dt)
model.eval()
print(f"  termuat dalam {time.time()-t0:.0f}s | params={sum(p.numel() for p in model.parameters())/1e9:.2f}B", flush=True)

def first_id(s):
    ids=tok.encode(s, add_special_tokens=False)
    return ids[0]

@torch.no_grad()
def score(prompt, pos_tok, neg_tok):
    ids=tok(prompt, return_tensors='pt')
    logits=model(**ids).logits[0,-1]
    lp=torch.log_softmax(logits.float(), dim=-1)
    return (lp[first_id(pos_tok)] - lp[first_id(neg_tok)]).item()

rows=list(csv.DictReader(open(IN), delimiter='\t'))
print(f"pasangan: {len(rows)} | kondisi: A0, A4 | template: {len(TEMPLATES)}", flush=True)
print(f"total forward pass: {len(rows)*2*len(TEMPLATES)}", flush=True)

t0=time.time(); n=0
with open(OUT,'w') as fh:
    for i,r in enumerate(rows,1):
        for cond in ('A0','A4'):
            drug=r[cond]
            for tname, tmpl, pos, neg in TEMPLATES:
                s=score(tmpl.format(d=drug, s=r['disease']), pos, neg)
                fh.write(json.dumps({
                    'drugbank_id':r['drugbank_id'],'umls_cui':r['umls_cui'],
                    'condition':cond,'template':tname,'score':s,
                    'label':int(r['label']),'tier':r.get('tier','?'),
                    'drug_degree':int(r['drug_degree']),
                    'disease_degree':int(r['disease_degree']),'model':MODEL,
                })+'\n')
                n+=1
        if i%20==0:
            el=time.time()-t0
            print(f"  ...{i}/{len(rows)}  ({n} pass, {el:.0f}s, {n/el:.1f} pass/s)", flush=True)
            fh.flush()
print(f"\nselesai: {n} forward pass dalam {time.time()-t0:.0f}s")
print(f"ditulis: {OUT}")
