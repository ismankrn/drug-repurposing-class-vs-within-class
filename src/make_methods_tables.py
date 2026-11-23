"""Dua tabel untuk bab Methodology: sumber data dan definisi kedua tugas.

Dipisah dari rebuild_outputs.py karena isinya properti BAHAN, bukan hasil --
tidak perlu bootstrap, jalan dalam hitungan detik, dan tidak ikut berubah saat
analisis diulang. Tetap dihasilkan skrip supaya angkanya tidak bisa menyimpang
dari data.
"""
import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from pgcopy import read_copy

DC='data/interim/drugcentral/'; OUT='paper/tables'
os.makedirs(OUT,exist_ok=True)
def w(fn,t): open(f'{OUT}/{fn}','w').write(t+"\n")

_,om=read_copy(DC+'omop_relationship.sql')
ind=[r for r in om if r['relationship_name']=='indication' and r['umls_cui']]
pairs={(r['struct_id'],r['umls_cui']) for r in ind}
_,idt=read_copy(DC+'identifier.sql'); db=[r for r in idt if r['id_type']=='DRUGBANK_ID']
_,pc=read_copy(DC+'pharma_class.sql')
epc=[r for r in pc if r['type']=='EPC']; moa=[r for r in pc if r['type']=='MoA']
_,act=read_copy(DC+'act_table_full.sql'); acc=[r for r in act if r.get('accession')]
_,s2a=read_copy(DC+'struct2atc.sql')

# ---------- Tabel: sumber data ----------
rows=[("Approved indications","\\texttt{omop\\_relationship}",f"{len(ind):,}",
       f"{len(pairs):,} unique pairs over {len({a for a,_ in pairs}):,} drugs and {len({b for _,b in pairs}):,} diseases"),
      ("Drug identity","\\texttt{identifier}",f"{len(db):,}",
       "DrugBank accessions used to normalise drug identity"),
      ("Pharmacologic class","\\texttt{pharma\\_class}",f"{len(epc):,}",
       f"Established Pharmacologic Class (EPC), {len({r['name'] for r in epc}):,} distinct classes"),
      ("Mechanism of action","\\texttt{pharma\\_class}",f"{len(moa):,}","used in the mechanism descriptions"),
      ("Molecular targets","\\texttt{act\\_table\\_full}",f"{len(acc):,}",
       f"UniProt accessions, {len({r['accession'] for r in acc}):,} distinct targets"),
      ("ATC category","\\texttt{struct2atc}",f"{len(s2a):,}","Anatomical Therapeutic Chemical codes")]
t=["\\begin{table}[!tbp]","\\centering","\\small",
   "\\caption{Data taken from the DrugCentral database.}",
   "\\label{tab:sources}","\\begin{tabular}{llrp{0.42\\linewidth}}","\\toprule",
   "Item & Source table & Records & Role \\\\","\\midrule"]
for a,b,c,d in rows: t.append(f"{a} & {b} & {c} & {d} \\\\")
t+=["\\bottomrule","\\end{tabular}","\\end{table}"]
w('tab-sources.tex',"\n".join(t))

# ---------- Tabel: definisi kedua tugas ----------
CLS=pd.read_csv('data/processed/pilot_pairs_v5.tsv',sep='\t')
CLS=CLS[CLS.tier.isin(['pos','neg_random'])].drop_duplicates(['drugbank_id','umls_cui'])
WIT=pd.read_csv('data/processed/within_class_pairs.tsv',sep='\t').drop_duplicates(['drugbank_id','umls_cui'])
gw=WIT.groupby(['umls_cui','shared_class']).label.nunique()
gc=CLS.groupby('drugbank_id').label.nunique()   # kelompok tugas tingkat-kelas = obat
spec=[("Positive instance","An approved drug--disease pair","An approved drug--disease pair"),
      ("Negative instance","A pair that is not an indication",
       "A drug of the same class not approved for that disease"),
      ("Held fixed between the two instances","Nothing","Disease and pharmacologic class"),
      ("Matched between the two instances","Drug degree and disease degree","Number of indications"),
      ("Pairs",f"{len(CLS):,}",f"{len(WIT):,}"),
      ("Positive pairs",f"{int(CLS.label.sum()):,}",f"{int(WIT.label.sum()):,}"),
      # Negatif ditulis eksplisit: pembaca tidak boleh disuruh mengurangkan
      # sendiri, dan keseimbangannya yang membuat 0,5 berarti kebetulan.
      ("Negative pairs",f"{len(CLS)-int(CLS.label.sum()):,}",
                       f"{len(WIT)-int(WIT.label.sum()):,}"),
      ("Distinct drugs",f"{CLS.drugbank_id.nunique():,}",f"{WIT.drugbank_id.nunique():,}"),
      ("Distinct diseases",f"{CLS.umls_cui.nunique():,}",f"{WIT.umls_cui.nunique():,}"),
      ("Pharmacologic classes","--",f"{WIT.shared_class.nunique():,}"),
      ("Grouping unit of the evaluation","Drug","Disease $\\times$ EPC class"),
      ("Groups",f"{len(gc):,}",f"{len(gw):,}"),
      ("Evaluable groups",f"{int((gc==2).sum()):,}",f"{int((gw==2).sum()):,}"),
      ("Expected \\auroc{} of a class-only predictor","Above chance","$0.5$ by construction")]
t=["\\begin{table}[!tbp]","\\centering","\\small",
   "\\caption{Construction and composition of the two tasks.}",
   "\\label{tab:tasks}","\\begin{tabular}{@{}p{0.30\\linewidth}p{0.31\\linewidth}p{0.33\\linewidth}@{}}","\\toprule",
   "& Class-level task & Within-class task \\\\","\\midrule"]
for a,b,c in spec: t.append(f"{a} & {b} & {c} \\\\")
t+=["\\bottomrule","\\end{tabular}","\\end{table}"]
w('tab-tasks.tex',"\n".join(t))
print("ditulis: paper/tables/tab-sources.tex, tab-tasks.tex")


# ---------- Tabel: model ----------
# Jumlah parameter dari metadata safetensors di Hugging Face (diverifikasi).
MODELS=[("OLMo-2","allenai/OLMo-2-0425-1B-Instruct","1.48","Instruct"),
        ("OLMo-2","allenai/OLMo-2-1124-7B-Instruct","7.30","Instruct"),
        ("OLMo-2","allenai/OLMo-2-1124-13B-Instruct","13.72","Instruct"),
        ("OLMo-2","allenai/OLMo-2-0325-32B-Instruct","32.23","Instruct"),
        ("Qwen3","Qwen/Qwen3-1.7B","2.03","Instruct"),
        ("Qwen3","Qwen/Qwen3-8B","8.19","Instruct"),
        ("Qwen3","Qwen/Qwen3-14B","14.77","Instruct"),
        ("Qwen3","Qwen/Qwen3-32B","32.76","Instruct")]
t=["\\begin{table}[!tbp]","\\centering","\\small",
   "\\caption{The eight checkpoints scored in this study.}","\\label{tab:models}",
   "\\begin{tabular}{lllr}","\\toprule",
   "Family & Checkpoint & Type & Parameters (B) \\\\","\\midrule"]
prev=None
for fam,ckpt,par,typ in MODELS:
    if prev and fam!=prev: t.append("\\addlinespace")
    t.append(f"{fam} & \\texttt{{{ckpt.split('/')[1]}}} & {typ} & {par} \\\\"); prev=fam
t+=["\\bottomrule","\\end{tabular}","\\end{table}"]
w('tab-models.tex',"\n".join(t))

# ---------- Tabel: template prompt ----------
# Disalin verbatim dari src/gpu_phase2.py; {d} obat atau deskripsinya, {s} penyakit.
TPL=[("\\texttt{qa}","Question: Is \\{d\\} an approved treatment for \\{s\\}?",
      "Answer:","\\texttt{\\ Yes}","\\texttt{\\ No}","same"),
     ("\\texttt{tf}","\\{d\\} is used to treat \\{s\\}.","This statement is:",
      "\\texttt{\\ True}","\\texttt{\\ False}","There is a drug described as \\{d\\}. It is used to treat \\{s\\}."),
     ("\\texttt{cloze}","Medical fact: the drug \\{d\\} is indicated for the treatment of \\{s\\}.",
      "Correct:","\\texttt{\\ yes}","\\texttt{\\ no}","Medical fact: \\{d\\} is indicated for the treatment of \\{s\\}.")]
t=["\\begin{table}[!tbp]","\\centering","\\small",
   "\\caption{The three prompt templates and their answer tokens.}","\\label{tab:templates}",
   "\\begin{tabular}{@{}lp{0.52\\linewidth}ll@{}}","\\toprule",
   "Template & Text & Affirmative & Negative \\\\","\\midrule"]
for nm,body,tail,pos,neg,alt in TPL:
    t.append(f"{nm} & {body} \\newline {tail} & {pos} & {neg} \\\\")
    if alt!="same":
        t.append(f" & \\textit{{mechanism variant:}} {alt} \\newline {tail} & & \\\\")
    t.append("\\addlinespace")
t=t[:-1]+["\\bottomrule","\\end{tabular}","\\end{table}"]
w('tab-templates.tex',"\n".join(t))
print("ditulis juga: tab-models.tex, tab-templates.tex")

# ---------- Tabel: tangga perturbasi ----------
# Contoh diambil LANGSUNG dari berkas data, bukan diketik ulang, supaya teks di
# tabel tidak bisa menyimpang dari string yang benar-benar dikirim ke model.
import csv as _csv
_rows=list(_csv.DictReader(open('data/processed/pilot_pairs_v5.tsv'),delimiter='\t'))
EX=next(r for r in _rows if r['A0']=='tobramycin' and r['A5'].strip())
def _tex(s):
    s=s.replace('\\','').replace('&','\\&').replace('%','\\%').replace('_','\\_')
    out=[];q=0
    for ch in s:
        if ch=='"': out.append('``' if q%2==0 else "''"); q+=1
        else: out.append(ch)
    return ''.join(out)
def _inc(long,short):
    assert long.startswith(short), "kondisi A5 tidak bersarang seperti yang diasumsikan"
    return long[len(short):].lstrip('; ').strip()
# Label P1-P5 berurutan; kunci internal A0/A4/A5c/A5t/A5 tidak diubah.
LAD=[("P1","Canonical name",_tex(EX['A0'])),
     ("P2","None (format control)",_tex(EX['A4'])),
     ("P3","EPC class",_tex(EX['A5c'])),
     ("P4","\\quad $+$ molecular targets","\\dots\\ ; "+_tex(_inc(EX['A5t'],EX['A5c']))),
     ("P5","\\quad $+$ ATC category","\\dots\\ ; "+_tex(_inc(EX['A5'],EX['A5t'])))]
t=["\\begin{table}[!tbp]","\\centering","\\small",
   "\\caption{The five conditions of the perturbation ladder.}","\\label{tab:ladderdef}",
   "\\begin{tabular}{@{}llp{0.50\\linewidth}@{}}","\\toprule",
   "Condition & Information given & Text substituted for the drug name \\\\","\\midrule"]
for c,i,x in LAD: t.append(f"{c} & {i} & {x} \\\\")
t+=["\\bottomrule","\\end{tabular}","\\end{table}"]
w('tab-ladderdef.tex',"\n".join(t))
print("ditulis juga: tab-ladderdef.tex")

# ---------- Tabel: komponen kedua baseline non-LLM ----------
# Cacahnya dihitung ulang di sini supaya tidak bisa menyimpang dari
# src/baseline_vs_llm.py dan src/kg_baseline.py.
import pandas as _pd
_ind={(r['struct_id'],r['umls_cui']) for r in ind}
_cls=_pd.read_csv('data/processed/pilot_pairs_v5.tsv',sep='\t')
_cls=_cls[_cls.tier.isin(['pos','neg_random'])].drop_duplicates(['drugbank_id','umls_cui'])
_wit=_pd.read_csv('data/processed/within_class_pairs.tsv',sep='\t').drop_duplicates(
    ['drugbank_id','umls_cui'])
_test={str(s) for s in _cls.struct_id}|{str(s) for s in _wit.struct_id}
_uni={str(s) for s,_ in _ind}
_nref=len(_uni-_test)           # identik dgn rebuild_outputs.py: dedup DULU, baru TEST
_drop={(str(r.struct_id),r.umls_cui) for r in _cls.itertuples() if r.label==1}
_drop|={(str(r.struct_id),r.umls_cui) for r in _wit.itertuples() if r.label==1}
_estr={(str(a),b) for a,b in _ind}
_ndrop=len(_estr&_drop); _ntrain=len(_estr)-_ndrop
# Obat BER-TARGET yang ada DI GRAF (2.779 simpul), bukan seluruh DrugCentral.
_nodes=_uni|{str(x) for x in _cls.struct_id}|{str(x) for x in _wit.struct_id}
_ntgt=len(_nodes & {str(r['struct_id']) for r in acc})
def _n(x): return f"{x:,}".replace(',','{,}')

SEC1=(f"\\textit{{Feature lookup: priors estimated from {_n(_nref)} reference drugs, "
      f"disjoint from both test sets}}")
SEC2=(f"\\textit{{Knowledge graph: {_n(_ntrain)} training edges, after the {_n(_ndrop)} "
      f"test edges are removed}}")
ROWS=[(SEC1,None,None),
 ("EPC class prior","reference drugs of the same pharmacologic class","$\\alpha=5$"),
 ("MoA prior","reference drugs of the same mechanism of action","$\\alpha=5$"),
 ("ATC level-4 prior","reference drugs sharing the ATC level-4 code","$\\alpha=5$"),
 ("ATC level-3 prior","reference drugs sharing the ATC level-3 code","$\\alpha=5$"),
 ("Shared-target prior","reference drugs sharing a molecular target","$\\alpha=5$"),
 ("Disease prior","the disease alone","---"),
 ("Drug degree","log number of indications of the drug","---"),
 (SEC2,None,None),
 ("Adamic--Adar","diseases shared by two drugs, weighted by rarity","---"),
 ("Katz index","weighted count of paths between drug and disease","$\\beta=0.05$, length $\\leq5$"),
 ("Truncated SVD","low-rank reconstruction of the incidence matrix","rank 64"),
 ("Target propagation","molecular target similarity spread to diseases",f"{_n(_ntgt)} drugs"),
 ("\\textit{Combination, used in both baselines}",None,None),
 ("Logistic regression","the components listed above","5 folds, grouped by drug")]
t=["\\begin{table}[!tbp]","\\centering","\\small",
   "\\caption{Components of the two non-LLM baselines.}","\\label{tab:baselines}",
   "\\begin{tabular}{@{}lp{0.40\\linewidth}l@{}}","\\toprule",
   "Component & Computed over & Setting \\\\","\\midrule"]
for a,b,c in ROWS:
    if b is None:
        t+=["\\addlinespace",f"\\multicolumn{{3}}{{@{{}}l}}{{{a}}} \\\\"]
    else:
        t.append(f"\\quad {a} & {b} & {c} \\\\")
t+=["\\bottomrule","\\end{tabular}","\\end{table}"]
w('tab-baselines.tex',"\n".join(t))
print(f"ditulis juga: tab-baselines.tex  (ref={_nref}, dibuang={_ndrop}, latih={_ntrain}, target={_ntgt})")
