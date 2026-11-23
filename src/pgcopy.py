"""Parser minimal untuk file COPY ... FROM stdin hasil pg_restore --data-only."""
import csv, sys

def read_copy(path):
    """Yield dict per baris. Mengembalikan (columns, rows)."""
    cols, rows = None, []
    with open(path, encoding='utf-8', errors='replace') as fh:
        in_data = False
        for line in fh:
            if not in_data:
                if line.startswith('COPY '):
                    inner = line[line.index('(')+1:line.index(')')]
                    cols = [c.strip() for c in inner.split(',')]
                    in_data = True
                continue
            if line.startswith('\\.'):
                break
            vals = line.rstrip('\n').split('\t')
            if len(vals) != len(cols):
                continue
            rows.append({c: (None if v == '\\N' else v) for c, v in zip(cols, vals)})
    return cols, rows
