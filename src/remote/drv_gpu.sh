#!/bin/bash
# A3, jalur GPU. Skor P1 (A0) saja, pada KEDUA tugas, untuk tiga checkpoint
# BASE OLMo-2. Tujuannya memisahkan efek skala dari efek instruction tuning:
# naskah saat ini hanya punya satu titik base (1B), jadi klaim apa pun tentang
# instruction tuning bersandar pada satu perbandingan.
#
# gpu_phase2.py DIPAKAI, bukan gpu_run_all.py: yang terakhir menyetel
# max_memory={0:...} sehingga hanya kartu pertama terbaca, dan model 32B akan
# di-offload ke CPU dengan throughput jatuh ~5x.
set -u
cd /workspace/dr
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home
export HF_XET_HIGH_PERFORMANCE=1
log(){ echo "[gpu $(date +%H:%M:%S)] $*"; }

skor(){            # $1=repo  $2=nama-pendek
  local repo="$1" nm="$2"
  log "menunggu penanda ready_$nm.ok"
  for i in $(seq 1 540); do [ -f "/workspace/dr/ready_$nm.ok" ] && break; sleep 10; done
  if [ ! -f "/workspace/dr/ready_$nm.ok" ]; then log "BATAL: $nm tidak pernah siap"; return 1; fi

  log "=== $repo : tugas tingkat-kelas ==="
  python src/gpu_phase2.py --model "$repo" \
    --in data/processed/pilot_pairs_v5.tsv \
    --out "results/easy_$nm.jsonl" --conds A0 || { log "GAGAL kelas $nm"; return 1; }

  log "=== $repo : tugas intra-kelas ==="
  python src/gpu_phase2.py --model "$repo" \
    --in data/processed/within_class_pairs.tsv \
    --out "results/within_$nm.jsonl" --conds A0 || { log "GAGAL intra $nm"; return 1; }

  log "$nm tuntas: $(wc -l < results/easy_$nm.jsonl) + $(wc -l < results/within_$nm.jsonl) baris"
  df -h / | tail -1
}

skor allenai/OLMo-2-1124-7B   OLMo-2-1124-7B   || exit 1
skor allenai/OLMo-2-1124-13B  OLMo-2-1124-13B  || exit 1
skor allenai/OLMo-2-0325-32B  OLMo-2-0325-32B  || exit 1
log "########## A3 SELESAI ##########"
