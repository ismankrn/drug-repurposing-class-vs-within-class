# Class-level discrimination and within-class resolution in drug–indication prediction

Code and data for a study that separates two abilities which a conventional
drug–indication benchmark joins:

- **class-level discrimination** — whether a *class* of drugs is used for a
  disease of a given kind;
- **within-class resolution** — which *member* of that class is the one
  approved for the disease at hand.

Two tasks are built over the same approved indications and differ only in the
identity of the negative instance. In the **class-level task** the negative is a
degree-matched pair that is not an indication. In the **within-class task** the
negative is a drug of the same pharmacologic class that is not approved for the
same disease, so that class-level knowledge is uninformative by construction.

Eleven checkpoints of two independent families (OLMo-2 and Qwen3, 1B–32B,
including three base checkpoints) are scored without fine-tuning under three
prompt templates, and performance is summarised by the macro within-group
AUROC with a 95 % confidence interval from a bootstrap over groups.

## Benchmark files

| File | Task | Pairs |
|---|---|---|
| `data/processed/pilot_pairs_v5.tsv` | class-level | 2,536 |
| `data/processed/within_class_pairs.tsv` | within-class | 2,763 |

Positives are approved indications taken from the DrugCentral
`omop_relationship` table; drug identity is normalised through DrugBank
accessions, and the pharmacologic class is the FDA Established Pharmacologic
Class as recorded by DrugCentral. The remaining files in `data/processed/` are
intermediate products of the benchmark build and of an earlier line of work
that the study did not pursue; the two files above are the ones the results
rest on.

## Score files

`results/*.jsonl` hold the per-pair scores of every checkpoint under every
prompt template. A score is the difference between the log-probability of the
affirmative answer token and that of the negative one. These files are what
make the results reproducible **without a GPU**.

## Reproducing the numbers

```bash
python3 -I src/rebuild_outputs.py   # main tables, from results/*.jsonl
python3 -I src/robustness.py        # stratification and calibration tables
python3 -I src/offlabel_check.py    # sensitivity to off-label negatives
```

Both of the first two scripts fix the seed of every resampling, so the
intervals they print are reproducible. A caution for anyone modifying them:
each script draws from a single global `numpy` generator, so inserting a new
bootstrap in the middle of a script shifts every interval computed after it.
New blocks belong at the end, or in a script with a generator of its own.

Scoring the checkpoints again from scratch needs a GPU and the phase-2 scripts
in `src/`; it is not required to reproduce any table.

## Rebuilding the benchmark from source

`data/raw/` and `data/interim/` are not distributed here: they hold a 2.9 GB
copy of the DrugCentral database, and three of the files exceed the GitHub
size limit. Download a DrugCentral release from <https://drugcentral.org/>
into `data/raw/drugcentral/`, then run the build scripts in `src/`.

## Licence

Code in `src/` is MIT (see `LICENSE`). The files in `data/` and `results/` are
derived from DrugCentral and are released under CC BY-SA 4.0, with the
attribution stated in `data/LICENSE`.
