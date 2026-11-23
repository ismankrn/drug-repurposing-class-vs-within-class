#!/bin/bash
# Pra-unduh bobot, versi dengan PENGAWAS KEMANDEKAN.
#
# Kenapa bukan "timeout" biasa seperti fetch.sh: batas waktu buta harus disetel
# selebar berkas terbesar yang wajar (2.700 s untuk 120 GiB), sehingga gantung
# baru dibunuh setelah 45 menit menganggur. Pengawas ini menonton lalu lintas
# nyata dan membunuh begitu aliran berhenti -- 60 detik, bukan 45 menit --
# tanpa pernah membunuh unduhan besar yang sehat tapi lambat.
#
# Diukur dari /sys/class/net/eth0/statistics/rx_bytes, bukan dari du: berkas
# .incomplete dialokasikan di muka dan du pernah menunjukkan 0 MiB/60 s
# sementara delapan shard mengalir pada 501 Mbps.
#
# BATAS: penghitung rx bersifat SE-MESIN, jadi lalu lintas lain bisa menutupi
# kemandekan. Ambang 2 MB/15 s dipilih jauh di atas derau SSH/pemantauan
# (kilobyte) dan jauh di bawah unduhan sehati terlambat (10 Mbps = 18 MB/15 s).
set -u
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_DOWNLOAD_TIMEOUT=30
AMBANG=${AMBANG:-2000000}      # byte per cuplikan 15 s
DIAM_MAKS=${DIAM_MAKS:-4}      # cuplikan sepi berturut-turut sebelum dibunuh
rx(){ cat /sys/class/net/eth0/statistics/rx_bytes; }
echo $$ > /workspace/dr/fetch.pid
trap 'rm -f /workspace/dr/fetch.pid' EXIT

unduh(){
  local repo="$1" diam=0 prev now delta pid
  hf download "$repo" & pid=$!
  prev=$(rx)
  while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    now=$(rx); delta=$(( now - prev )); prev=$now
    if [ "$delta" -lt "$AMBANG" ]; then
      diam=$(( diam + 1 ))
      echo "    [pengawas] sepi $diam/$DIAM_MAKS ($(( delta/1024 )) KiB/15s)"
      if [ "$diam" -ge "$DIAM_MAKS" ]; then
        echo "    [pengawas] MANDEK -> dibunuh setelah $(( DIAM_MAKS*15 ))s tanpa aliran"
        kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
        return 124
      fi
    else
      diam=0
    fi
  done
  wait "$pid"; return $?
}

for M in "$@"; do
  echo "########## $M $(date +%H:%M:%S) ##########"
  OK=0
  for try in $(seq 1 12); do
    echo "--- percobaan $try $(date +%H:%M:%S) ---"
    unduh "$M"; rc=$?
    if [ "$rc" -eq 0 ]; then echo "OK: $M"; OK=1; break; fi
    if [ "$rc" -eq 124 ]; then
      echo "  -> mandek; melanjutkan dari .incomplete"
    else
      echo "  -> keluar dengan kode $rc"
    fi
    sleep 10
  done
  if [ "$OK" -ne 1 ]; then echo "MENYERAH: $M"; exit 1; fi
  du -sh /workspace/.hf_home; df -h / | tail -1
done
echo "########## UNDUHAN SELESAI $(date +%H:%M:%S) ##########"
