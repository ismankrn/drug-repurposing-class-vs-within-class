"""Cek sebelum sewa: model terjangkau? berapa VRAM & disk? berapa lama?

Hanya memakai urllib (tanpa huggingface_hub), jadi bisa dijalankan di laptop
apa adanya. Tidak mengunduh bobot sama sekali.

  python3 src/preflight.py Qwen/Qwen3-1.7B Qwen/Qwen3-8B Qwen/Qwen3-14B Qwen/Qwen3-32B
"""
import sys, json, urllib.request, urllib.error

JOBS=[('Run 1  Qwen3 x4, A0, dua tugas',71316),
      ('Run 2  OLMo-2 32B, A4+A5c+A5t+A5',31491),
      ('Run 3  Qwen3-32B, A4+A5c+A5t+A5',31491)]

print(f"{'model':<34} {'param':>8} {'VRAM bf16':>10} {'unduh':>7}  status")
print("-"*78)
total=0; bad=0; sizes=[]
for m in sys.argv[1:]:
    try:
        with urllib.request.urlopen(f"https://huggingface.co/api/models/{m}?blobs=true",timeout=25) as r:
            d=json.load(r)
    except urllib.error.HTTPError as e:
        print(f"{m:<34} {'-':>8} {'-':>10} {'-':>7}  TIDAK ADA (HTTP {e.code})"); bad+=1; continue
    except Exception as e:
        print(f"{m:<34} {'-':>8} {'-':>10} {'-':>7}  GAGAL ({type(e).__name__})"); bad+=1; continue
    p=(d.get('safetensors') or {}).get('total') or 0
    disk=sum(s.get('size') or 0 for s in d.get('siblings',[])
             if s.get('rfilename','').endswith('.safetensors'))/1e9
    g=d.get('gated')
    note="GATED — perlu persetujuan di HF dulu" if g and g!=False else "ok"
    if g and g!=False: bad+=1
    total+=disk; sizes.append(disk)
    print(f"{m:<34} {p/1e9:>7.1f}B {p*2/1e9:>9.0f}G {disk:>6.0f}G  {note}")
print(f"\nTotal unduh bila semua disimpan sekaligus: {total:.0f} GB")
print(f"Disk minimum bila tiap model dihapus setelah dipakai: {max(sizes or [0])+20:.0f} GB "
      f"(model terbesar {max(sizes or [0]):.0f} GB + ~20 GB ruang kerja)")
print("\nPerkiraan waktu komputasi (di luar waktu unduh):")
print(f"{'skenario':<36} {'pass':>8} {'@20/dtk':>9} {'@3/dtk':>9}")
print("-"*66)
for k,v in JOBS:
    print(f"{k:<36} {v:>8} {v/20/60:>8.0f}m {v/3/60:>8.0f}m")
print("\n20/dtk ~ model muat penuh di VRAM; 3/dtk ~ sebagian modul di-offload ke CPU.")
print("Aturan praktis bf16: VRAM >= 2 x parameter, plus ~4 GB aktivasi.")
if bad: print(f"\n{bad} model bermasalah — selesaikan sebelum menyewa instance.")
else:   print("\nSemua model siap.")
