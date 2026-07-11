"""
Aggregate probe CSVs from multiple seeds into a mean +/- std table.

Usage:
    python diagnostics/aggregate.py checkpoints/compositional-run4/version_*/probe_r2_test.csv
    python diagnostics/aggregate.py --label run6 checkpoints/compositional/version_6/probe_r2_test.csv \
        checkpoints/compositional/version_7/probe_r2_test.csv
"""
import argparse
import csv
from collections import defaultdict

import numpy as np


def read_csv(path):
    values = {}
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            target = row[0]
            for col, cell in zip(header[1:], row[1:]):
                if cell.strip():
                    values[(target, col)] = float(cell)
    return values


def main(paths, label):
    per_key = defaultdict(list)
    for path in paths:
        for key, value in read_csv(path).items():
            per_key[key].append(value)

    n_files = len(paths)
    print(f'\nAggregated over {n_files} file(s)' + (f' [{label}]' if label else '') + ':')
    for path in paths:
        print(f'  {path}')

    targets, cols = [], []
    for target, col in per_key:
        if target not in targets:
            targets.append(target)
        if col not in cols:
            cols.append(col)

    header = f'\n{"target":<22}' + ''.join(f'{c:>18}' for c in cols)
    print(header)
    print('-' * len(header))
    for target in targets:
        line = f'{target:<22}'
        for col in cols:
            vals = per_key.get((target, col))
            if not vals:
                line += f'{"":>18}'
            elif len(vals) == 1:
                line += f'{vals[0]:>18.3f}'
            else:
                line += f'{np.mean(vals):>10.3f} ± {np.std(vals):<5.3f}'
        print(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aggregate probe CSVs across seeds.')
    parser.add_argument('paths', nargs='+', help='probe_r2_*.csv files (glob is fine)')
    parser.add_argument('--label', type=str, default='')
    args = parser.parse_args()
    main(args.paths, args.label)
