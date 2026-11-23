#!/bin/bash
# Pra-unduh bobot model sebelum scoring, di instance GPU sewaan.
#
# KENAPA ADA "timeout" DI SETIAP PERCOBAAN: kegagalan nyata di mesin ini bukan
# galat melainkan GANTUNG. "hf download" berhenti memindahkan byte (terukur
# 0 Mbps selama puluhan menit, satu blob .incomplete membeku) tetapi tidak
# pernah kembali, sehingga lingkaran ulang-coba tanpa timeout menunggu selamanya
# sementara GPU menganggur dan jam sewa jalan. Terjadi dua kali, dua kali pada
# BERKAS TERAKHIR. HF_HUB_DOWNLOAD_TIMEOUT tidak mencakup jalur Xet. Dengan
# timeout, gantung menjadi gagal, dan percobaan berikutnya melanjutkan dari
# .incomplete yang sudah ada -- jadi mengulang itu murah.
#
# Xet DIPAKAI, jangan dimatikan: di huggingface_hub 2.x hf_transfer sudah tidak
# dipakai lagi dan HF_HUB_DISABLE_XET=1 menjatuhkan unduhan ke HTTP satu aliran
# (terukur 6 Mbps, lawan ~500 Mbps lewat Xet).
#
# TANPA --exclude: "hf download REPO [FILENAMES...]" memperlakukan argumen
# posisional sebagai daftar berkas yang DIUNDUH, jadi pola tambahan di belakang
# --exclude membuatnya mengunduh 0 berkas lalu melaporkan "OK" secara palsu.
set -u
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_DOWNLOAD_TIMEOUT=30
BATAS=${BATAS:-1800}
echo $$ > /workspace/dr/fetch.pid
trap 'rm -f /workspace/dr/fetch.pid' EXIT
for M in "$@"; do
  echo "########## $M $(date +%H:%M:%S) ##########"
  OK=0
  for try in $(seq 1 10); do
    echo "--- percobaan $try $(date +%H:%M:%S) (batas ${BATAS}s) ---"
    timeout --kill-after=30 "$BATAS" hf download "$M"
    rc=$?
    if [ "$rc" -eq 0 ]; then echo "OK: $M"; OK=1; break; fi
    if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
      echo "  -> kena batas waktu (gantung); melanjutkan dari .incomplete"
    else
      echo "  -> keluar dengan kode $rc"
    fi
    sleep 10
  done
  if [ "$OK" -ne 1 ]; then echo "MENYERAH: $M"; exit 1; fi
  du -sh /workspace/.hf_home; df -h / | tail -1
done
echo "########## UNDUHAN SELESAI $(date +%H:%M:%S) ##########"
