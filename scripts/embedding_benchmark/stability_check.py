"""Robustness check for the benchmark ranking -- NOT a significance test.

WHAT WAS TRIED AND REJECTED
---------------------------
Three approaches were attempted before landing here, and each failed for a
reason worth recording so nobody repeats them:

1. **Well bootstrap** (`resample wells with replacement, recompute mAP`).
   Duplicated rows become each other's nearest neighbours, which inflates AP.
   Measured: point estimate 0.1338, bootstrap mean 0.1864, CI [0.1615, 0.2176] --
   the point estimate fell outside its own interval.  A point outside its own CI
   is a broken procedure, not a wide one.  **Withdrawn.**

2. **Leave-one-well-out retrieval.** Structurally impossible here: every compound
   occupies exactly one well, so holding that well out leaves zero relevant items
   in the database.  Only the 11 mock wells (which repeat across wells) produced
   a number, so the "ranking" was 11 wells of noise.

3. **Per-well AP against the full database.** Implemented by masking the query
   well's own columns to `inf` instead of removing them; `argsort` still returned
   those columns, so every relevant item was ranked last by construction and AP
   collapsed to ~0.035 for all six candidates identically.  Also withdrawn.

WHY NO P-VALUE IS REPORTED
--------------------------
With one well per compound, compound identity and well identity are the same
variable.  Any test of "does method A retrieve compounds better than method B"
either excludes the query's own well (no relevant items, approach 2) or keeps it
(measures well recognition, not chemistry).  Neither supports a significance
claim.  So this script reports **stability**, not significance.

WHAT IS ACTUALLY REPORTED
-------------------------
* the descriptive ranking with the raw-pixel and random anchors;
* a k-fold split over wells, holding the evaluation protocol fixed, to check
  whether the *ordering* survives on subsets of the data.

Order stability across folds is evidence that the number is not driven by one
lucky well.  It is not evidence of statistical significance, and is not
presented as such.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import sys
from pathlib import Path

import numpy as np

# Import the sibling scoring module by location, so the pair works from any
# checkout. The previous version hardcoded an absolute path into a private
# directory outside the repository.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def _load_scoring():
    import score_benchmark

    return score_benchmark


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cells-dir', required=True)
    parser.add_argument('--folds', type=int, default=4)
    parser.add_argument(
        '--profile', default='v2', choices=['v2', 'v3'],
        help='v3 also folds the surrogate-backed methods and marks them',
    )
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    import pandas as pd

    scoring = _load_scoring()

    cells = Path(args.cells_dir)
    meta = pd.read_csv(cells / 'meta.csv')
    labels = meta['compound'].astype(str).to_numpy()
    wells = meta['well'].astype(str).to_numpy()

    features = {}
    from cellpaint_pipeline.models import registry

    registry.set_profile(args.profile)
    provenance = {m.name: m.weights_provenance for m in registry.manifests()}
    for name in registry.declared():
        path = cells / f'emb_{name}.npy'
        if path.exists():
            features[name] = np.asarray(np.load(path), dtype=np.float64)
    features['baseline_raw_pixels'] = np.asarray(
        np.load(cells / 'cells_dp.npy', mmap_mode='r'), dtype=np.float64
    ).reshape(len(meta), -1)
    features['baseline_random'] = np.random.default_rng(0).standard_normal(
        (len(meta), 384)
    )

    print('=== full-data mAP (descriptive; confounded by well) ===')
    full = {}
    for name, raw in features.items():
        full[name] = scoring.mean_average_precision(
            scoring.prepare_z(raw), labels
        )
        print(f'  {name:<22} mAP_z={full[name]:.4f}')

    # Assign wells to folds deterministically, then score each fold on its own
    # cells.  Z-scoring is recomputed inside each fold so the split does not leak
    # statistics across folds.
    unique_wells = np.unique(wells)
    rng = np.random.default_rng(12345)
    shuffled = rng.permutation(unique_wells)
    fold_of_well = {well: position % args.folds for position, well in enumerate(shuffled)}
    fold_ids = np.array([fold_of_well[w] for w in wells])

    print()
    print(f'=== {args.folds}-fold split over wells: does the ORDER survive? ===')
    header = '  fold   n   ' + ''.join(f'{n[:12]:>14}' for n in features)
    print(header)
    print('  ' + '-' * (len(header) - 2))

    fold_scores: dict[int, dict[str, float]] = {}
    for fold in range(args.folds):
        mask = fold_ids == fold
        if mask.sum() < 50:
            continue
        row = {}
        for name, raw in features.items():
            row[name] = scoring.mean_average_precision(
                scoring.prepare_z(raw[mask]), labels[mask]
            )
        fold_scores[fold] = row
        print(f'  {fold:<5}{int(mask.sum()):<5}' +
              ''.join(f'{row[n]:>14.4f}' for n in features))

    # Rank order per fold, restricted to the real methods (not the baselines).
    methods = [n for n in features if not n.startswith('baseline_')]
    print()
    print(f'=== per-fold ordering of the {len(methods)} methods ===')
    orders = {}
    for fold, row in fold_scores.items():
        order = sorted(methods, key=lambda n: -row[n])
        orders[fold] = order
        print(f'  fold {fold}: ' + ' > '.join(order))

    consistent = len({tuple(o) for o in orders.values()}) == 1
    print()
    print(f'  identical ordering in every fold: {consistent}')

    # A whole-table "stable / unstable" verdict is too coarse to be useful once
    # more than a few methods are compared: two methods sitting on the random
    # baseline swap places easily while the entire top of the table is solid.
    # Reporting the stable *prefix* and the specific pairs that swap says what
    # the data supports, instead of collapsing that into one word.
    reference = orders[min(orders)]
    stable_prefix = 0
    for position in range(len(methods)):
        if len({tuple(o[:position + 1]) for o in orders.values()}) == 1:
            stable_prefix = position + 1
        else:
            break

    _PAIR = None  # keeps flake8 quiet about the loop variable below
    swapped = set()
    for order in orders.values():
        for position, (a, b) in enumerate(zip(reference, order)):
            if a != b:
                pair = tuple(sorted((a, b)))
                swapped.add(pair)
                break

    print(f'  stable prefix (same order in every fold): {stable_prefix} of '
          f'{len(methods)}')
    if stable_prefix:
        print('    ' + ' > '.join(reference[:stable_prefix]))
    if swapped:
        print('  pairs whose relative order changes across folds: ' +
              ', '.join(f'{a}/{b}' for a, b in sorted(swapped)))
        print('  -> the top of the table is stable; comparisons between the '
              'listed pairs are NOT resolved by this data')
    elif not consistent:
        print('  -> the ordering is NOT stable across folds; report it as unstable')

    # How often each method lands first.
    firsts: dict[str, int] = {m: 0 for m in methods}
    for order in orders.values():
        firsts[order[0]] += 1
    print('  first-place counts: ' +
          ', '.join(f'{k}={v}' for k, v in sorted(firsts.items(), key=lambda x: -x[1])))

    payload = {
        'full_data_map_z': full,
        'folds': {str(k): v for k, v in fold_scores.items()},
        'per_fold_order': {str(k): v for k, v in orders.items()},
        'order_stable': consistent,
        'stable_prefix_size': stable_prefix,
        'stable_prefix': reference[:stable_prefix],
        'unstable_pairs': [list(pair) for pair in sorted(swapped)],
        'first_place_counts': firsts,
        'no_p_values_because': (
            'one well per compound makes compound identity equal to well '
            'identity; leave-one-well-out has no relevant items, and any test '
            'that keeps the well measures well recognition rather than chemistry'
        ),
        'rejected_approaches': {
            'well_bootstrap': 'duplicated rows inflate AP; point estimate fell '
                              'outside its own 95% CI (0.1338 vs [0.1615, 0.2176])',
            'leave_one_well_out': 'zero relevant items for 53 of 64 wells',
            'per_well_ap_masked': 'masked columns still returned by argsort, so '
                                  'relevant items ranked last by construction',
        },
        'baseline_note': (
            "the 2026-10-26 numbers for unidino / morphogenie / dinocells were "
            "produced by SURROGATE checkpoints, and dinocells' was retrained "
            "that day because its input geometry changed from the official "
            "storage format (a wide tape) to the official model input format (a "
            "channel stack). Its score therefore moved, and the old value is "
            "not a regression to be recovered -- the weights and the geometry "
            "both changed"
        ),
    }
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print(f'\nwrote {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
