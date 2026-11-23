"""Run lengkap 5 kondisi + audit adversarial dlm satu sesi GPU.

Kondisi:
  A0  nama kanonik          -- baseline
  A4  pseudonim acak        -- identitas hancur, TANPA informasi apa pun
  A5c kelas farmakologis    -- mekanisme minimal (33% non-identifikasi)
  A5t kelas + target        -- mekanisme menengah (18%)
  A5  kelas + target + ATC  -- mekanisme kaya (6%)

A4 vs A5* lah yg memisahkan "menghafal nama->penyakit" dari "perlu identitas
utk menalar" -- Δ(A0-A4) saja tidak bisa (lihat Temuan W).

Pakai: python3 gpu_run_all.py <model> <in.tsv> <out.jsonl> [batch]
"""
import csv, json, os, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL=sys.argv[1]; IN=sys.argv[2]; OUT=sys.argv[3]
BATCH=int(sys.argv[4]) if len(sys.argv)>4 else 24
os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)

# A0/A4 = nama; A5* = frasa deskriptif -> template berbeda agar gramatikal
T_NAME=[
 ("qa",    "Question: Is {d} an approved treatment for {s}?\nAnswer:", " Yes"," No"),
 ("tf",    "{d} is used to treat {s}.\nThis statement is:", " True"," False"),
 ("cloze", "Medical fact: the drug {d} is indicated for the treatment of {s}.\nCorrect:", " yes"," no"),
]
T_DESC=[
 ("qa",    "Question: Is {d} an approved treatment for {s}?\nAnswer:", " Yes"," No"),
 ("tf",    "There is a drug described as {d}. It is used to treat {s}.\nThis statement is:", " True"," False"),
 ("cloze", "Medical fact: {d} is indicated for the treatment of {s}.\nCorrect:", " yes"," no"),
]
CONDS=[('A0',T_NAME),('A4',T_NAME),('A5c',T_DESC),('A5t',T_DESC),('A5',T_DESC)]

print(f"memuat {MODEL}", flush=True); t0=time.time()
tok=AutoTokenizer.from_pretrained(MODEL); tok.padding_side='left'
if tok.pad_token is None: tok.pad_token=tok.eos_token
# Model >24 GB (13B=27,4 / 32B=64,5) dibagi GPU+CPU lewat accelerate.
# bf16 DIPERTAHANKAN -- offload hanya memindah komputasi, tidak seperti
# kuantisasi yg akan mendistorsi log-prob (jantung pengukuran kita).
import subprocess as _sp
_free=int(_sp.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],
                  capture_output=True).stdout.decode().split()[0])
_cap=f"{max(_free//1024-3,4)}GiB"
model=AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="auto",
    max_memory={0:_cap, "cpu":"70GiB"}).eval()
try:
    dmap=getattr(model,'hf_device_map',{}) or {}
    ncpu=sum(1 for v in dmap.values() if v in ('cpu','disk'))
    print(f"  pemetaan: {len(dmap)} modul, {ncpu} di CPU (cap GPU {_cap})", flush=True)
except Exception: pass
print(f"  {time.time()-t0:.0f}s | VRAM {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)
def fid(s): return tok.encode(s, add_special_tokens=False)[0]

rows=list(csv.DictReader(open(IN), delimiter='\t'))
jobs=[]
for r in rows:
    for cond,tmpls in CONDS:
        d=(r.get(cond) or '').strip()
        if not d: continue                      # lewati bila varian tak tersedia
        for tn,tmpl,pos,neg in tmpls:
            jobs.append((r,cond,tn,tmpl.format(d=d,s=r['disease']),fid(pos),fid(neg)))
print(f"pasangan={len(rows)} | job={len(jobs)}", flush=True)
cov={}
for cond,_ in CONDS: cov[cond]=sum(1 for r in rows if (r.get(cond) or '').strip())
print(f"cakupan per kondisi: {cov}", flush=True)

@torch.no_grad()
def run(batch):
    enc=tok([b[3] for b in batch], return_tensors='pt', padding=True).to('cuda')
    lp=torch.log_softmax(model(**enc).logits[:,-1,:].float(), dim=-1)
    return [(lp[i,b[4]]-lp[i,b[5]]).item() for i,b in enumerate(batch)]

t0=time.time(); n=0
with open(OUT,'w') as fh:
    for i in range(0,len(jobs),BATCH):
        ch=jobs[i:i+BATCH]
        for b,s in zip(ch, run(ch)):
            r,cond,tn=b[0],b[1],b[2]
            fh.write(json.dumps({'drugbank_id':r['drugbank_id'],'umls_cui':r['umls_cui'],
                'condition':cond,'template':tn,'score':s,'label':int(r['label']),
                'tier':r['tier'],'drug_degree':int(r['drug_degree']),
                'disease_degree':int(r['disease_degree']),
                'sig_class_size':int(r.get('sig_class_size') or 0),
                'a5t_class_size':int(r.get('a5t_class_size') or 0),
                'coarse_class_size':int(r.get('coarse_class_size') or 0),
                'model':MODEL})+'\n')
            n+=1
        if (i//BATCH)%40==0:
            el=time.time()-t0; print(f"  {n}/{len(jobs)} {el:.0f}s {n/max(el,1):.0f}/dtk", flush=True); fh.flush()
print(f"\nselesai: {n} pass, {time.time()-t0:.0f}s -> {OUT}")
