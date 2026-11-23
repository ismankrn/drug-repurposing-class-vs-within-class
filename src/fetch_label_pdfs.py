"""Unduh PDF label FDA utk diff. URL openFDA TIDAK ter-encode -> spasi jadi %20 (Temuan T)."""
import json, os, subprocess, sys, time

PD='data/raw/fda_labels'; os.makedirs(PD, exist_ok=True)
pairs=json.load(open('data/processed/label_diff_pairs.json'))
urls={}
for p in pairs:
    urls[p['new_url']]=f"{p['appl']}_s{p['suppl']}_new.pdf"
    urls[p['old_url']]=f"{p['appl']}_s{p['old_suppl']}_old.pdf"
print(f"PDF unik: {len(urls)}", flush=True)

ok=skip=fail=0
for i,(u,fn) in enumerate(urls.items(),1):
    path=os.path.join(PD,fn)
    if os.path.exists(path) and os.path.getsize(path)>5000:
        skip+=1
    else:
        enc=u.replace(' ','%20')
        r=subprocess.run(['curl','-sSL','-f','--retry','2','--retry-delay','2',
                          '--max-time','120','-o',path,enc], capture_output=True)
        good = r.returncode==0 and os.path.exists(path) and os.path.getsize(path)>5000
        if good:
            with open(path,'rb') as fh: good = fh.read(5)==b'%PDF-'
        if good: ok+=1
        else: fail+=1; os.path.exists(path) and os.remove(path)
        time.sleep(0.25)
    if i%100==0:
        print(f"  {i}/{len(urls)}  baru={ok} cache={skip} gagal={fail}", flush=True)
print(f"\nselesai: baru={ok} cache={skip} gagal={fail}")
print(f"ukuran total: {sum(os.path.getsize(os.path.join(PD,f)) for f in os.listdir(PD))/1e9:.2f} GB")
