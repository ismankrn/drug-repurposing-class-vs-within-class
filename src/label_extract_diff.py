"""Ekstrak bagian INDICATIONS AND USAGE dari PDF label FDA, lalu diff lama vs baru.

Struktur label FDA baku:
  HIGHLIGHTS ... INDICATIONS AND USAGE  (ringkas, di atas)
  FULL PRESCRIBING INFORMATION
    1 INDICATIONS AND USAGE   <- yg kita ambil
    1.1 ...
    2 DOSAGE AND ADMINISTRATION  <- batas akhir

Kalimat dianggap BARU bila kemiripan terbaiknya thd seluruh kalimat label lama
di bawah ambang -- toleran terhadap perubahan redaksional kecil.
"""
import json, os, re, subprocess, sys, difflib, collections

PD='data/raw/fda_labels'; OUT='data/processed'
THRESH=float(sys.argv[1]) if len(sys.argv)>1 else 0.88

START=re.compile(r'\n\s*1\s+INDICATIONS\s+AND\s+USAGE', re.I)
END  =re.compile(r'\n\s*2\s+DOSAGE\s+AND\s+ADMINISTRATION|\n\s*2\s+DOSAGE', re.I)
HL   =re.compile(r'INDICATIONS\s+AND\s+USAGE(.{50,3000}?)(?:DOSAGE\s+AND\s+ADMINISTRATION|CONTRAINDICATIONS)', re.I|re.S)

def pdftext(path):
    try:
        r=subprocess.run(['pdftotext','-layout',path,'-'],capture_output=True,timeout=180)
        return r.stdout.decode('utf-8','replace')
    except Exception:
        return ''

def indications(path):
    t=pdftext(path)
    if len(t.strip())<300: return None, 'teks kosong (scan?)'
    m=START.search(t)
    if m:
        rest=t[m.end():]
        e=END.search(rest)
        sec=rest[:e.start()] if e else rest[:6000]
        if len(sec.strip())>80: return sec, 'full_PI'
    m=HL.search(t)
    if m and len(m.group(1).strip())>80: return m.group(1), 'highlights'
    return None, 'bagian tidak ditemukan'

BULLET=re.compile(r'[•●·]|\n\s*[-*]\s')
def sentences(sec):
    s=BULLET.sub('. ', sec)
    s=re.sub(r'\(\s*\d+(\.\d+)?\s*\)','',s)          # rujukan silang "(1.2)"
    s=re.sub(r'\s+',' ',s)
    out=[]
    for p in re.split(r'(?<=[.;])\s+', s):
        p=p.strip(' .;')
        if len(p)<30 or len(p)>400: continue
        if re.fullmatch(r'[\d\s.,()%-]+', p): continue
        out.append(p)
    return out

def norm(x): return re.sub(r'[^a-z0-9 ]','', x.lower())

def new_sentences(old, new):
    o=[norm(x) for x in old]
    res=[]
    for s in new:
        n=norm(s)
        best=max((difflib.SequenceMatcher(None,n,x).ratio() for x in o), default=0.0)
        if best < THRESH: res.append((s, round(best,2)))
    return res

if __name__=='__main__':
    pairs=json.load(open(f'{OUT}/label_diff_pairs.json'))
    stat=collections.Counter(); rows=[]
    for i,p in enumerate(pairs,1):
        fn_new=os.path.join(PD, f"{p['appl']}_s{p['suppl']}_new.pdf")
        fn_old=os.path.join(PD, f"{p['appl']}_s{p['old_suppl']}_old.pdf")
        if not (os.path.exists(fn_new) and os.path.exists(fn_old)):
            stat['PDF belum ada']+=1; continue
        sn,srcn=indications(fn_new); so,srco=indications(fn_old)
        if sn is None or so is None:
            stat[f'ekstraksi gagal ({srcn if sn is None else srco})']+=1; continue
        new=new_sentences(sentences(so), sentences(sn))
        stat[f'sumber: {srcn}']+=1
        if new:
            stat['ADA kalimat baru']+=1
            rows.append({**p,'n_new':len(new),'src':srcn,
                         'new_text':' | '.join(s for s,_ in new[:5])[:600]})
        else:
            stat['tidak ada perubahan indikasi']+=1
        if i%100==0: print(f"  {i}/{len(pairs)}", flush=True)

    print(f"\n=== RINGKASAN (ambang kemiripan {THRESH}) ===")
    for k,v in stat.most_common(): print(f"  {k:<36} {v}")
    with open(f'{OUT}/label_diff_results.tsv','w') as fh:
        fh.write('appl\tsuppl\tdate\told_suppl\told_date\tn_new\tsrc\tnew_text\n')
        for r in rows:
            fh.write(f"{r['appl']}\t{r['suppl']}\t{r['date']}\t{r['old_suppl']}\t{r['old_date']}\t{r['n_new']}\t{r['src']}\t{r['new_text']}\n")
    print(f"\nditulis: {OUT}/label_diff_results.tsv ({len(rows)} supplement dgn indikasi baru)")
