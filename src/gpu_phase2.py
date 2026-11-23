"""Runner fase 2: keluarga model kedua + tangga perturbasi 32B.

Beda dari gpu_run_all.py:
  - BISA DILANJUTKAN. Job yang sudah ada di file keluaran dilewati, jadi
    instance spot yang mati di tengah jalan tinggal dijalankan ulang dengan
    perintah yang sama. Ini alasan utama skrip ini ada.
  - Filter kondisi lewat --conds, supaya 32B bisa menjalankan A4/A5c/A5t/A5
    saja tanpa mengulang A0 yang sudah selesai.
  - Batch otomatis menyusut kalau kena OOM, tidak langsung mati.

Pakai:
  python3 src/gpu_phase2.py --model Qwen/Qwen3-8B \
      --in data/processed/pilot_pairs_v5.tsv \
      --out results/cls_qwen3_8b.jsonl --conds A0
"""
import argparse, collections, csv, json, os, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ap=argparse.ArgumentParser()
ap.add_argument('--model',required=True)
ap.add_argument('--in',dest='inp',required=True)
ap.add_argument('--out',required=True)
ap.add_argument('--conds',default='A0,A4,A5c,A5t,A5')
ap.add_argument('--batch',type=int,default=24)
ap.add_argument('--limit',type=int,default=0,help='hanya N job pertama (uji cepat)')
A=ap.parse_args()
WANT=[c.strip() for c in A.conds.split(',') if c.strip()]
os.makedirs(os.path.dirname(A.out) or '.',exist_ok=True)

T_NAME=[("qa","Question: Is {d} an approved treatment for {s}?\nAnswer:"," Yes"," No"),
        ("tf","{d} is used to treat {s}.\nThis statement is:"," True"," False"),
        ("cloze","Medical fact: the drug {d} is indicated for the treatment of {s}.\nCorrect:"," yes"," no")]
T_DESC=[("qa","Question: Is {d} an approved treatment for {s}?\nAnswer:"," Yes"," No"),
        ("tf","There is a drug described as {d}. It is used to treat {s}.\nThis statement is:"," True"," False"),
        ("cloze","Medical fact: {d} is indicated for the treatment of {s}.\nCorrect:"," yes"," no")]
TMPL={'A0':T_NAME,'A4':T_NAME,'A5c':T_DESC,'A5t':T_DESC,'A5':T_DESC}

done=set()
if os.path.exists(A.out):
    with open(A.out) as fh:
        for line in fh:
            try:
                r=json.loads(line); done.add((r['drugbank_id'],r['umls_cui'],r['condition'],r['template']))
            except Exception: pass
    print(f"melanjutkan: {len(done)} job sudah ada di {A.out}",flush=True)

print(f"memuat {A.model}",flush=True); t0=time.time()
tok=AutoTokenizer.from_pretrained(A.model); tok.padding_side='left'
if tok.pad_token is None: tok.pad_token=tok.eos_token
import subprocess as _sp
# SEMUA GPU dipakai, bukan hanya GPU 0. Versi lama hanya membaca kartu pertama,
# sehingga di mesin 4x24 GB model 32B akan menaruh 24 GB di GPU 0 dan sisanya
# di-offload ke CPU -- throughput jatuh ~5x dan bisa kehabisan RAM.
_fr=[int(x) for x in _sp.run(['nvidia-smi','--query-gpu=memory.free',
     '--format=csv,noheader,nounits'],capture_output=True).stdout.decode().split()]
if not _fr: sys.exit("nvidia-smi tidak mengembalikan GPU apa pun.")
maxmem={i:f"{max(f//1024-2,2)}GiB" for i,f in enumerate(_fr)}
maxmem["cpu"]="24GiB"          # kecil & disengaja: offload CPU harus jadi
                               # kejutan yang terlihat, bukan jalan diam-diam
cap=f"{sum(f//1024-2 for f in _fr)}GiB total dari {len(_fr)} GPU"
# bf16 dipertahankan. Kuantisasi akan mendistorsi log-prob, yaitu besaran yang
# justru kita ukur -- offload ke CPU hanya memindah komputasi, bukan mengubah nilai.
model=AutoModelForCausalLM.from_pretrained(A.model,dtype=torch.bfloat16,
        device_map="auto",max_memory=maxmem).eval()
dmap=getattr(model,'hf_device_map',{}) or {}
ncpu=sum(1 for v in dmap.values() if v in ('cpu','disk'))
spread=collections.Counter(str(v) for v in dmap.values())
print(f"  {time.time()-t0:.0f}s | {len(dmap)} modul | {cap}",flush=True)
print(f"  sebaran: {dict(spread)}",flush=True)
if ncpu:
    print(f"  PERINGATAN: {ncpu} modul di CPU/disk. Throughput akan turun ~5x.",flush=True)
    print(f"  VRAM tidak cukup -- tambah GPU, atau jalankan model lebih kecil dulu.",flush=True)
def fid(s): return tok.encode(s,add_special_tokens=False)[0]
# Sanitas tokenizer: tiap keluarga model memecah " Yes"/" No" berbeda. Kalau
# token pertama kedua jawaban sama, selisih log-prob jadi nol dan seluruh run
# tidak ada artinya -- jadi diperiksa sebelum memuat data, bukan sesudahnya.
for _c,_t in TMPL.items():
    for _tn,_,_p,_n in _t:
        _a,_b=fid(_p),fid(_n)
        if _a==_b:
            sys.exit(f"FATAL: template {_tn} memberi token pertama identik untuk "
                     f"{_p!r} dan {_n!r} (id={_a}) pada tokenizer {A.model}.")
print("  token jawaban: "+", ".join(
    f"{_tn}:{fid(_p)}/{fid(_n)}" for _tn,_,_p,_n in TMPL['A0']),flush=True)

rows=list(csv.DictReader(open(A.inp),delimiter='\t'))
# Beberapa pasangan (obat, penyakit) muncul lebih dari sekali, dan segelintir
# dengan label BERTENTANGAN -- akar masalahnya satu drugbank_id bisa memetakan ke
# beberapa struct_id DrugCentral, sehingga penyaring "tidak terindikasi" pada
# sampler negatif meleset. Dampaknya pada hasil <=0,001 (sudah diuji), tapi
# duplikat membuat kunci resume tidak unik, jadi dibersihkan di sini.
_seen={}; _conf=0
for r in rows:
    k=(r['drugbank_id'],r['umls_cui'])
    if k in _seen:
        if _seen[k]['label']!=r['label']: _conf+=1
        continue
    _seen[k]=r
if len(_seen)<len(rows):
    print(f"  dedup: {len(rows)-len(_seen)} baris ganda dibuang "
          f"({_conf} di antaranya berlabel bertentangan)",flush=True)
rows=list(_seen.values())
jobs=[]
for r in rows:
    for cond in WANT:
        d=(r.get(cond) or '').strip()
        if not d: continue
        for tn,tmpl,pos,neg in TMPL[cond]:
            if (r['drugbank_id'],r['umls_cui'],cond,tn) in done: continue
            jobs.append((r,cond,tn,tmpl.format(d=d,s=r['disease']),fid(pos),fid(neg)))
if A.limit: jobs=jobs[:A.limit]
print(f"baris={len(rows)} kondisi={WANT} | job tersisa={len(jobs)}",flush=True)
if not jobs: print("tidak ada yang perlu dijalankan."); sys.exit(0)

@torch.no_grad()
def run(batch):
    enc=tok([b[3] for b in batch],return_tensors='pt',padding=True).to(model.device)
    lp=torch.log_softmax(model(**enc).logits[:,-1,:].float(),dim=-1)
    return [(lp[i,b[4]]-lp[i,b[5]]).item() for i,b in enumerate(batch)]

t0=time.time(); n=0; bs=A.batch; i=0
with open(A.out,'a') as fh:
    while i<len(jobs):
        ch=jobs[i:i+bs]
        try:
            sc=run(ch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs==1: raise
            bs=max(1,bs//2); print(f"  OOM -> batch turun ke {bs}",flush=True); continue
        for b,s in zip(ch,sc):
            r,cond,tn=b[0],b[1],b[2]
            fh.write(json.dumps({'drugbank_id':r['drugbank_id'],'umls_cui':r['umls_cui'],
                'condition':cond,'template':tn,'score':s,'label':int(r['label']),
                'tier':r['tier'],'drug_degree':int(r['drug_degree']),
                'disease_degree':int(r['disease_degree']),
                'sig_class_size':int(r.get('sig_class_size') or 0),
                'a5t_class_size':int(r.get('a5t_class_size') or 0),
                'coarse_class_size':int(r.get('coarse_class_size') or 0),
                'model':A.model})+'\n')
            n+=1
        i+=len(ch)
        if n % (bs*40) < bs:
            el=time.time()-t0; sisa=(len(jobs)-n)/max(n/el,1e-9)
            print(f"  {n}/{len(jobs)} {el:.0f}s {n/max(el,1):.1f}/dtk sisa~{sisa/60:.0f}mnt",flush=True)
            fh.flush()
print(f"\nselesai: {n} pass, {(time.time()-t0)/60:.1f} menit -> {A.out}")
