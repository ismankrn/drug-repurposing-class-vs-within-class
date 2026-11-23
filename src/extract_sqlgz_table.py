"""Ekstrak satu blok COPY dari dump .sql.gz secara streaming (hemat memori).

Pakai: python3 -I src/extract_sqlgz_table.py <dump.sql.gz> <nama_tabel> <keluaran.sql>
"""
import gzip, sys

def main(path, table, out):
    needle = f"COPY public.{table} ("
    alt    = f"COPY {table} ("
    n = 0
    with gzip.open(path, 'rt', encoding='utf-8', errors='replace') as fh, \
         open(out, 'w', encoding='utf-8') as wh:
        capturing = False
        for line in fh:
            if not capturing:
                if line.startswith(needle) or line.startswith(alt):
                    wh.write(line); capturing = True
                continue
            if line.startswith('\\.'):
                wh.write(line); break
            wh.write(line); n += 1
    print(f"{table}: {n} baris data -> {out}")
    return n

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(__doc__); sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
