"""Inferensi GPU berbatch: skoring log-prob Yes/No.

Left-padding dipakai agar token terakhir selalu di posisi -1 untuk setiap
anggota batch, sehingga logits[:, -1] valid tanpa indexing per-contoh.

Pakai: python3 gpu_inference.py <model> <in.tsv> <out.jsonl> [batch] [dtype]
"""
import csv, json, os, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = sys.argv[1]; IN = sys.argv[2]; OUT = sys.argv[3]
BATCH = int(sys.argv[4]) if len(sys.argv)>4 else 32
DTYPE = sys.argv[5] if len(sys.argv)>5 else "bfloat16"
os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)

TEMPLATES=[
 ("qa",    "Question: Is {d} an approved treatment for {s}?\nAnswer:", " Yes", " No"),
 ("tf",    "{d} is used to treat {s}.\nThis statement is:", " True", " False"),
 ("cloze", "Medical fact: the drug {d} is indicated for the treatment of {s}.\nCorrect:", " yes", " no"),
]

print(f"memuat {MODEL}", flush=True); t0=time.time()
tok=AutoTokenizer.from_pretrained(MODEL)
tok.padding_side='left'
if tok.pad_token is None: tok.pad_token = tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL, dtype=getattr(torch,DTYPE), device_map="cuda")
model.eval()
print(f"  {time.time()-t0:.0f}s | params={sum(p.numel() for p in model.parameters())/1e9:.2f}B", flush=True)
print(f"  VRAM terpakai: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

def fid(s): return tok.encode(s, add_special_tokens=False)[0]
TOKS={n:(fid(p),fid(q)) for n,_,p,q in TEMPLATES}

rows=list(csv.DictReader(open(IN), delimiter='\t'))
jobs=[]
for r in rows:
    for cond in ('A0','A4'):
        for tname,tmpl,_,_ in TEMPLATES:
            jobs.append((r, cond, tname, tmpl.format(d=r[cond], s=r['disease'])))
print(f"pasangan={len(rows)} | job={len(jobs)} | batch={BATCH}", flush=True)

@torch.no_grad()
def run(batch):
    prompts=[j[3] for j in batch]
    enc=tok(prompts, return_tensors='pt', padding=True).to('cuda')
    logits=model(**enc).logits[:,-1,:].float()
    lp=torch.log_softmax(logits, dim=-1)
    out=[]
    for i,j in enumerate(batch):
        pos,neg=TOKS[j[2]]
        out.append((lp[i,pos]-lp[i,neg]).item())
    return out

t0=time.time(); n=0
with open(OUT,'w') as fh:
    for i in range(0,len(jobs),BATCH):
        chunk=jobs[i:i+BATCH]
        for j,s in zip(chunk, run(chunk)):
            r,cond,tname,_=j
            fh.write(json.dumps({'drugbank_id':r['drugbank_id'],'umls_cui':r['umls_cui'],
                'condition':cond,'template':tname,'score':s,'label':int(r['label']),
                'tier':r['tier'],'drug_degree':int(r['drug_degree']),
                'disease_degree':int(r['disease_degree']),'model':MODEL})+'\n')
            n+=1
        if (i//BATCH)%20==0:
            el=time.time()-t0
            print(f"  {n}/{len(jobs)}  {el:.0f}s  {n/max(el,1):.0f} pass/s", flush=True); fh.flush()
print(f"\nselesai: {n} pass dalam {time.time()-t0:.0f}s -> {OUT}")
