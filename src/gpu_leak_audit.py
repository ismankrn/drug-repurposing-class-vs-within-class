"""Audit adversarial: bisakah model MENEBAK obatnya dari deskripsi A5?

Memvalidasi apakah ukuran-kelas-tanda-tangan benar memprediksi kebocoran.
Skor = log P(nama obat benar | deskripsi) - log P(nama distraktor | deskripsi),
distraktor diambil dari obat lain pd stratum ukuran kelas yg sama.
"""
import csv, json, os, random, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL=sys.argv[1]; IN=sys.argv[2]; OUT=sys.argv[3]
VARIANT=sys.argv[4] if len(sys.argv)>4 else 'A5'
SIZEF={'A5':'sig_class_size','A5t':'a5t_class_size','A5c':'coarse_class_size'}[VARIANT]
rng=random.Random(7)

tok=AutoTokenizer.from_pretrained(MODEL); tok.padding_side='left'
if tok.pad_token is None: tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()

rows=[r for r in csv.DictReader(open(IN), delimiter='\t') if (r.get(VARIANT) or '').strip()]
seen={}; 
for r in rows: seen.setdefault(r['drugbank_id'], r)   # satu entri per obat
drugs=list(seen.values())
print(f"{VARIANT}: {len(drugs)} obat unik diaudit", flush=True)

PROMPT="A drug is described as: {desc}.\nThe name of this drug is"

@torch.no_grad()
def logp_name(desc, name):
    p=PROMPT.format(desc=desc)
    ids=tok(p, return_tensors='pt').to('cuda')
    nid=tok(" "+name, add_special_tokens=False)['input_ids']
    full=torch.cat([ids['input_ids'], torch.tensor([nid]).to('cuda')], dim=1)
    out=model(full).logits[0].float()
    lp=torch.log_softmax(out[:-1], dim=-1)
    tgt=full[0,1:]
    start=ids['input_ids'].shape[1]-1
    return lp[start:, :].gather(1, tgt[start:].unsqueeze(1)).sum().item()/max(len(nid),1)

res=[]; t0=time.time()
for i,r in enumerate(drugs,1):
    desc=r[VARIANT]; true=r['A0']
    pool=[d['A0'] for d in drugs if d['drugbank_id']!=r['drugbank_id']]
    distr=rng.sample(pool, min(4,len(pool)))
    lt=logp_name(desc, true)
    ld=[logp_name(desc, x) for x in distr]
    rank=1+sum(1 for x in ld if x>lt)
    res.append({'drugbank_id':r['drugbank_id'],'A0':true,'variant':VARIANT,
                'class_size':int(r.get(SIZEF) or 0),'logp_true':lt,
                'logp_distractor_mean':sum(ld)/len(ld),'rank_of_true':rank,
                'n_candidates':1+len(distr)})
    if i%100==0: print(f"  {i}/{len(drugs)} {time.time()-t0:.0f}s", flush=True)

with open(OUT,'w') as fh:
    for x in res: fh.write(json.dumps(x)+'\n')
top1=sum(1 for x in res if x['rank_of_true']==1)
print(f"\nidentifikasi benar (rank 1 dari 5): {top1}/{len(res)} ({100*top1/len(res):.0f}%)")
print("  kebetulan = 20%. Jauh di atas itu = deskripsi membocorkan identitas.")
print(f"ditulis: {OUT}")
