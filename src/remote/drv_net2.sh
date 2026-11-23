#!/bin/bash
# A3, jalur JARINGAN. Berpasangan dengan drv_gpu.sh yang memegang GPU.
#
# Dipisah supaya unduhan 120 GiB untuk 32B base menumpang dengan scoring 7B dan
# 13B. Kalau berurutan, GPU menganggur selama unduhan dan jam sewa tetap jalan;
# itu yang sudah memakan ~1 jam percuma sebelum ini.
#
# Setiap model yang tuntas menulis penanda ready_*.ok; drv_gpu.sh menunggu
# penanda itu, bukan menebak dari ukuran direktori (du tidak bisa dipercaya
# selama unduhan Xet -- lihat komentar di fetch.sh).
set -u
cd /workspace/dr
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home
log(){ echo "[net $(date +%H:%M:%S)] $*"; }

log "menunggu unduhan 13B yang sedang jalan selesai"
for i in $(seq 1 360); do
  [ -f /workspace/dr/fetch.pid ] || break
  sleep 10
done
if ! grep -q "OK: allenai/OLMo-2-1124-13B" /workspace/dr/fetch_A3a.log; then
  log "BATAL: 13B tidak selesai"; tail -5 /workspace/dr/fetch_A3a.log; exit 1
fi
touch /workspace/dr/ready_OLMo-2-1124-13B.ok
log "13B siap"

# Bersihkan dulu, baru ukur. Prune aman SEKARANG karena tidak ada unduhan yang
# berjalan -- kalau dijalankan saat 13B masih mengunduh, berkas .incomplete
# miliknya ikut terhapus dan progresnya hilang.
log "memangkas unduhan tak lengkap yang menggantung"
hf cache prune -y 2>&1 | tail -3

# 32B base butuh ~129 GB. Bobot Qwen3 (81,9 GB) sudah tidak diperlukan: hasil
# A1 sudah ditarik ke laptop dan terverifikasi. Dry-run dulu supaya yang
# terhapus tercatat di log sebelum benar-benar hilang.
log "pratinjau penghapusan bobot Qwen3"
hf cache rm model/Qwen/Qwen3-8B model/Qwen/Qwen3-32B --dry-run 2>&1 | tail -6
log "menghapus bobot Qwen3"
hf cache rm model/Qwen/Qwen3-8B model/Qwen/Qwen3-32B -y 2>&1 | tail -4
df -h / | tail -1

log "mengunduh 32B base (120,1 GiB fp32)"
# Keluaran dialihkan ke berkasnya sendiri: tanpa ini ia bercampur dengan log
# driver dan pemeriksaan "OK:" di bawah jatuh ke hf cache ls, yang cocok juga
# untuk unduhan separuh -- penanda ready akan jadi dusta.
AMBANG=2000000 DIAM_MAKS=4 /workspace/dr/fetch2.sh allenai/OLMo-2-0325-32B > /workspace/dr/fetch_A3b.log 2>&1
if grep -q "OK: allenai/OLMo-2-0325-32B" /workspace/dr/fetch_A3b.log 2>/dev/null; then
  touch /workspace/dr/ready_OLMo-2-0325-32B.ok
  log "32B base siap"
else
  log "GAGAL: 32B base tidak tuntas"; exit 1
fi
log "jalur jaringan selesai"
