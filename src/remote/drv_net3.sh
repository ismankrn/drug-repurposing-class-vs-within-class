#!/bin/bash
# A3, jalur JARINGAN -- versi dengan pengawas kemandekan untuk SEMUA unduhan.
#
# Versi sebelumnya hanya memasang pengawas pada 32B, sementara 13B masih jalan
# di bawah batas-waktu-buta 2.700 s. 13B lalu mandek dan akan menganggur 17
# menit sebelum dibunuh. Di sini kedua unduhan memakai fetch2.sh, jadi setiap
# kemandekan dipotong dalam 60 detik.
#
# Menghentikan dan memulai ulang unduhan itu murah: HF melanjutkan dari berkas
# .incomplete, jadi byte yang sudah turun tidak hilang.
set -u
cd /workspace/dr
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home
log(){ echo "[net $(date +%H:%M:%S)] $*"; }

log "melanjutkan unduhan 13B di bawah pengawas"
/workspace/dr/fetch2.sh allenai/OLMo-2-1124-13B > /workspace/dr/fetch_13b.log 2>&1
if ! grep -q "OK: allenai/OLMo-2-1124-13B" /workspace/dr/fetch_13b.log; then
  log "BATAL: 13B tidak tuntas"; tail -5 /workspace/dr/fetch_13b.log; exit 1
fi
touch /workspace/dr/ready_OLMo-2-1124-13B.ok
log "13B siap -- jalur GPU boleh mulai"

# Pangkas SETELAH 13B tuntas, bukan sebelum: kalau dijalankan saat unduhan masih
# berjalan, berkas .incomplete miliknya ikut terhapus dan progresnya hilang.
log "memangkas unduhan tak lengkap yang menggantung"
hf cache prune -y 2>&1 | tail -3

# 32B base butuh ~129 GB. Bobot Qwen3 (81,9 GB) sudah tidak diperlukan: hasil A1
# sudah ditarik ke laptop dan terverifikasi. Bobotnya TIDAK ada di direktori
# per-model -- itu hanya symlink ke hub/blobs bersama -- jadi "rm -rf" pada
# direktori model tidak membebaskan ruang sama sekali. Harus lewat hf cache rm.
# Dry-run dulu supaya yang terhapus tercatat sebelum benar-benar hilang.
log "pratinjau penghapusan bobot Qwen3"
hf cache rm model/Qwen/Qwen3-8B model/Qwen/Qwen3-32B --dry-run 2>&1 | tail -6
log "menghapus bobot Qwen3"
hf cache rm model/Qwen/Qwen3-8B model/Qwen/Qwen3-32B -y 2>&1 | tail -4
df -h / | tail -1

log "mengunduh 32B base (120,1 GiB fp32) di bawah pengawas"
/workspace/dr/fetch2.sh allenai/OLMo-2-0325-32B > /workspace/dr/fetch_A3b.log 2>&1
if ! grep -q "OK: allenai/OLMo-2-0325-32B" /workspace/dr/fetch_A3b.log; then
  log "BATAL: 32B base tidak tuntas"; tail -5 /workspace/dr/fetch_A3b.log; exit 1
fi
touch /workspace/dr/ready_OLMo-2-0325-32B.ok
log "32B base siap -- jalur jaringan selesai"
