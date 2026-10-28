"""Score the four embeddings on the BBBC022 subset and rank them.

WHAT CAN AND CANNOT BE MEASURED HERE
------------------------------------
The salvaged subset has **one well per compound and one concentration per
compound**.  That rules out the two strongest Cell Painting benchmarks:
dose-response curves need several concentrations of one compound, and
leave-one-well-out evaluation needs a compound to appear in two wells.

What is left is compound discrimination: given a cell's embedding, can it be
matched to other cells of the same compound?  That is measurable, but it carries
a known confound -- cells from one well share illumination, staining and field
position, so an embedding can score well by recognising the *well* rather than
the *chemistry*.

The confound is handled by reporting, for every method, the same metrics for two
baselines computed from the identical crops:
  * **raw pixels** -- which can only capture field/well appearance,
  * **random projection** -- which can only score at chance.
A method is only adding something if it beats raw pixels on compound retrieval.
Absolute scores are inflated for everyone and must not be quoted as method
quality.

METRICS
-------
``knn``      leave-one-out k-NN (k=10) accuracy over compounds; chance = 1/53.
``map``      mean average precision for same-compound retrieval.
``sil``      silhouette on compound labels (higher = better separated).
``auc``      separability of compound vs mock cells, from distance to the mock
             centroid (0.5 = chance).
``rank``     participation ratio of the singular-value spectrum.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: Path) -> np.ndarray:
    return np.load(path, mmap_mode='r')


def prepare(embeddings: np.ndarray) -> np.ndarray:
    """Mean-centre and L2-normalise, so distances are scale-free."""
    x = np.asarray(embeddings, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def prepare_z(embeddings: np.ndarray) -> np.ndarray:
    """Per-dimension z-score, then L2-normalise rows.

    This is the standard profile-normalisation step in Cell Painting pipelines
    (pycytominer standardises before aggregation), and here it is also a fairness
    requirement: measured, one dimension carries 71% of CA-MAE's variance and 57%
    of O2-VAE's.  Without per-dimension scaling, distances for those two methods
    are effectively one-dimensional and every other axis is ignored, which is a
    property of the embedding's scale rather than of its information.
    """
    x = np.asarray(embeddings, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    scale = x.std(axis=0, keepdims=True)
    x = x / np.maximum(scale, 1e-12)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def lowo_average_precision(
    x: np.ndarray, labels: np.ndarray, wells: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Leave-one-well-out retrieval AP, plus the per-well values.

    Returns ``(mean_ap, per_well_ap, kept_wells)``.

    Why this replaces the bootstrap: resampling wells *with replacement* and then
    scoring the duplicated rows inflates AP, because a duplicated cell becomes its
    own nearest neighbour.  Measured, that pushed the bootstrap mean to 0.186
    while the point estimate was 0.134 -- the point estimate landed outside its
    own 95% interval, the signature of a broken resampling scheme rather than of
    a wide interval.

    Here each well is scored exactly once against a database that excludes it.
    Nothing is duplicated, the queries are genuinely unseen, and the per-well
    values can be paired across methods for a valid test.
    """
    unique_wells = np.unique(wells)
    per_well: list[float] = []
    kept: list[str] = []
    for well in unique_wells:
        query = wells == well
        database = ~query
        if query.sum() == 0 or database.sum() == 0:
            continue
        q = x[query]
        d = x[database]
        db_labels = labels[database]
        q_labels = labels[query]

        distances = 1.0 - q @ d.T            # x is L2-normalised
        order = np.argsort(distances, axis=1)

        aps = []
        for i in range(len(q)):
            relevant = db_labels[order[i]] == q_labels[i]
            n_relevant = int(relevant.sum())
            if n_relevant == 0:
                continue
            hits = np.cumsum(relevant)
            precision = hits / np.arange(1, len(relevant) + 1)
            aps.append(float((precision * relevant).sum() / n_relevant))
        if aps:
            per_well.append(float(np.mean(aps)))
            kept.append(str(well))

    array = np.asarray(per_well)
    return (
        float(array.mean()) if len(array) else 0.0,
        array,
        np.asarray(kept),
    )


def per_well_average_precision(
    x: np.ndarray, labels: np.ndarray, wells: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Per-well retrieval AP against the full database, plus the values.

    Returns ``(mean_ap, per_well_ap, kept_wells)``.

    WHY NOT LEAVE-ONE-WELL-OUT
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    Leave-one-well-out retrieval was tried first and is **structurally impossible
    on this dataset**: every compound occurs in exactly one well, so holding that
    well out removes the compound from the database entirely.  Measured, the
    held-out AP was undefined for 53 of 64 wells -- only the 11 mock wells
    survived, because mock appears in several -- and the resulting "ranking" was
    noise on 11 wells.

    So the query set is one well and the database is everything else *including*
    the other cells of that well.  That is confounded -- a query's own well-mates
    are its nearest neighbours -- and the absolute AP must never be quoted as
    accuracy.

    What it is good for: pairing by well.  Every method is scored on exactly the
    same wells with exactly the same contamination, so a paired test on the
    differences measures which embedding separates compound identity *more* in
    the same confounded setting.  That supports "A > B", not "A is accurate".
    """
    unique_wells = np.unique(wells)
    per_well: list[float] = []
    kept: list[str] = []
    for well in unique_wells:
        query = wells == well
        if query.sum() == 0:
            continue
        q = x[query]
        q_labels = labels[query]

        distances = 1.0 - q @ x.T
        # Never let a cell retrieve itself.
        distances[:, query] = np.inf
        order = np.argsort(distances, axis=1)

        aps = []
        for i in range(len(q)):
            relevant = labels[order[i]] == q_labels[i]
            n_relevant = int(relevant.sum())
            if n_relevant == 0:
                continue
            hits = np.cumsum(relevant)
            precision = hits / np.arange(1, len(relevant) + 1)
            aps.append(float((precision * relevant).sum() / n_relevant))
        if aps:
            per_well.append(float(np.mean(aps)))
            kept.append(str(well))

    array = np.asarray(per_well)
    return (
        float(array.mean()) if len(array) else 0.0,
        array,
        np.asarray(kept),
    )


def knn_accuracy(x: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    """Leave-one-out k-NN accuracy.  Uses matrix multiplication for distances."""
    gram = x @ x.T
    squared = np.diag(gram)
    distances = squared[:, None] - 2.0 * gram + squared[None, :]
    np.fill_diagonal(distances, np.inf)

    neighbours = np.argpartition(distances, k, axis=1)[:, :k]
    correct = 0
    for i in range(len(x)):
        votes = labels[neighbours[i]]
        values, counts = np.unique(votes, return_counts=True)
        if values[np.argmax(counts)] == labels[i]:
            correct += 1
    return correct / len(x)


def mean_average_precision(x: np.ndarray, labels: np.ndarray) -> float:
    """Same-label retrieval AP averaged over queries."""
    gram = x @ x.T
    squared = np.diag(gram)
    distances = squared[:, None] - 2.0 * gram + squared[None, :]
    np.fill_diagonal(distances, np.inf)

    order = np.argsort(distances, axis=1)
    average_precisions = []
    for i in range(len(x)):
        relevant = labels[order[i]] == labels[i]
        n_relevant = relevant.sum()
        if n_relevant == 0:
            continue
        hits = np.cumsum(relevant)
        ranks = np.arange(1, len(relevant) + 1)
        precision = hits / ranks
        average_precisions.append((precision * relevant).sum() / n_relevant)
    return float(np.mean(average_precisions)) if average_precisions else 0.0


def silhouette(x: np.ndarray, labels: np.ndarray, max_points: int = 1200) -> float:
    """Silhouette sampled to keep the O(n^2) cost bounded."""
    rng = np.random.default_rng(0)
    index = np.arange(len(x))
    if len(x) > max_points:
        index = rng.choice(index, size=max_points, replace=False)
    xs, ls = x[index], labels[index]

    gram = xs @ xs.T
    squared = np.diag(gram)
    distances = np.sqrt(np.maximum(
        squared[:, None] - 2.0 * gram + squared[None, :], 0.0
    ))

    unique = np.unique(ls)
    if len(unique) < 2:
        return 0.0
    scores = []
    for i in range(len(xs)):
        same = ls == ls[i]
        same[i] = False
        if same.sum() == 0:
            continue
        a = distances[i, same].mean()
        b = min(
            distances[i, ls == other].mean()
            for other in unique if other != ls[i]
        )
        denominator = max(a, b)
        if denominator > 0:
            scores.append((b - a) / denominator)
    return float(np.mean(scores)) if scores else 0.0


def mock_auc(x: np.ndarray, is_mock: np.ndarray, wells: np.ndarray) -> float:
    """Mock-vs-compound separability, cross-validated over mock wells.

    The naive version -- one centroid from all mock cells, then score everything
    -- leaks: a point is always closer to a centroid it helped compute.  Measured,
    that inflated the *random projection* baseline to AUC 0.72 where the true
    value must be 0.5.  So the centroid is rebuilt from the other mock wells and
    only the held-out well is scored against the compound cells; the reported
    number is the mean over those folds.
    """
    mock_wells = np.unique(wells[is_mock])
    if len(mock_wells) < 2:
        return float('nan')

    fold_aucs = []
    compound = ~is_mock
    for well in mock_wells:
        held = wells == well
        train = is_mock & ~held
        if train.sum() == 0:
            continue
        centroid = x[train].mean(axis=0)
        scores = -np.linalg.norm(x - centroid, axis=1)
        positive = scores[held]
        negative = scores[compound]

        total = 0.0
        step = 4096
        for start in range(0, len(negative), step):
            block = negative[start:start + step]
            total += (positive[:, None] > block[None, :]).sum()
            total += 0.5 * (positive[:, None] == block[None, :]).sum()
        fold_aucs.append(total / (len(positive) * len(negative)))
    return float(np.mean(fold_aucs)) if fold_aucs else float('nan')


def embedding_diagnostics(features: np.ndarray) -> dict:
    """How concentrated the embedding is, which decides if it can rank at all.

    A near-constant embedding can still score above chance on a coarse label, so
    the concentration is reported alongside the scores rather than inferred from
    them.
    """
    x = np.asarray(features, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(x, compute_uv=False)
    power = singular ** 2
    top1 = float(power[0] / max(power.sum(), 1e-30))
    top3 = float(power[:3].sum() / max(power.sum(), 1e-30))
    return {'top1_var': top1, 'top3_var': top3}


def effective_rank(embeddings: np.ndarray) -> float:
    x = np.asarray(embeddings, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(x, compute_uv=False)
    power = singular ** 2
    return float(power.sum() ** 2 / max((power ** 2).sum(), 1e-30))


def evaluate(
    features: np.ndarray, labels: np.ndarray, is_mock: np.ndarray, wells: np.ndarray
) -> dict:
    plain = prepare(features)
    zscored = prepare_z(features)
    result = {
        'knn': knn_accuracy(plain, labels),
        'map': mean_average_precision(plain, labels),
        'knn_z': knn_accuracy(zscored, labels),
        'map_z': mean_average_precision(zscored, labels),
        'sil': silhouette(plain, labels),
        'sil_z': silhouette(zscored, labels),
        'auc_z': mock_auc(zscored, is_mock, wells),
        'rank': effective_rank(features),
    }
    result.update(embedding_diagnostics(features))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cells-dir', required=True)
    parser.add_argument('--output', default=None)
    parser.add_argument(
        '--profile', default='v2', choices=['v2', 'v3'],
        help='v2 scores the official-weights methods; v3 also scores the '
             'surrogate-backed ones, and marks them as such',
    )
    args = parser.parse_args()

    import pandas as pd

    cells = Path(args.cells_dir)
    meta = pd.read_csv(cells / 'meta.csv')
    labels = meta['compound'].astype(str).to_numpy()
    is_mock = (meta['well_role'] == 'mock').to_numpy()
    wells = meta['well'].astype(str).to_numpy()
    print(f'cells {len(meta)}  compounds {len(set(labels))}  '
          f'mock {int(is_mock.sum())}  mock_wells {len(set(wells[is_mock]))}  '
          f'chance={1/len(set(labels)):.4f}')

    from cellpaint_pipeline.models import registry

    registry.set_profile(args.profile)
    provenance = {m.name: m.weights_provenance for m in registry.manifests()}

    rng = np.random.default_rng(0)
    candidates = {}
    missing = []
    for name in registry.declared():
        path = cells / f'emb_{name}.npy'
        if path.exists():
            candidates[name] = load(path)
        else:
            missing.append(name)
    if missing:
        print(f'  (no embedding on disk for: {", ".join(missing)})')

    # Baselines computed from the SAME crops each method saw, so the comparison
    # isolates what the embedding adds over raw appearance.
    dp = np.load(cells / 'cells_dp.npy', mmap_mode='r')
    candidates['baseline_raw_pixels'] = np.asarray(dp).reshape(len(dp), -1)
    candidates['baseline_random'] = rng.standard_normal((len(meta), 384))

    rows = {}
    for name, features in candidates.items():
        rows[name] = evaluate(np.asarray(features), labels, is_mock, wells)
        r = rows[name]
        tag = ' [SURROGATE]' if provenance.get(name) == 'surrogate' else ''
        print(f'  {name:<22} knn={r["knn"]:.4f} knn_z={r["knn_z"]:.4f} '
              f'map={r["map"]:.4f} map_z={r["map_z"]:.4f} '
              f'sil_z={r["sil_z"]:+.4f} auc_z={r["auc_z"]:.4f} '
              f'rank={r["rank"]:.1f}')

    # Rank on the z-scored variant: it is the standard normalisation and the only
    # one that does not let a single dominant dimension decide the result.
    methods = [n for n in rows if not n.startswith('baseline_')]
    ranked = sorted(methods, key=lambda n: rows[n]['map_z'], reverse=True)
    print()
    print('ranking by z-scored mAP (compound retrieval):')
    for position, name in enumerate(ranked, 1):
        tag = '   <-- SURROGATE (not this method\'s published performance)' \
            if provenance.get(name) == 'surrogate' else ''
        print(f'  {position}. {name:<16} mAP_z={rows[name]["map_z"]:.4f}  '
              f'kNN_z={rows[name]["knn_z"]:.4f}{tag}')

    payload = {
        'cells': int(len(meta)),
        'compounds': int(len(set(labels))),
        'mock_cells': int(is_mock.sum()),
        'mock_wells': int(len(set(wells[is_mock]))),
        'chance_knn': 1 / len(set(labels)),
        'metrics': rows,
        'ranking_by_map_z': ranked,
        'profile': args.profile,
        'weights_provenance': provenance,
        'surrogates': sorted(
            n for n, p in provenance.items() if p == 'surrogate'
        ),
        'baselines': [n for n in rows if n.startswith('baseline_')],
        'caveats': [
            'one well per compound: well/field appearance is confounded with chemistry',
            'absolute scores are inflated for every method; the comparison '
            'against baseline_raw_pixels is what is interpretable',
            'one concentration per compound, so no dose-response is possible',
            'O2-VAE consumes a nucleus silhouette, not 5-channel intensity',
            "O2-VAE's official checkpoint was trained on CELL silhouettes "
            "(configs/config_mefs.py:19 -> data/mefs/scaled/cell) and no "
            'nucleus checkpoint is published, so feeding it a nucleus '
            'silhouette is OUT-OF-DOMAIN inference: the embedding distribution '
            'does not match the training distribution, and the number should '
            'not be read as the method\'s performance',
            'mock AUC is cross-validated over the 4 mock wells to avoid centroid '
            'self-inclusion bias',
            'rows marked SURROGATE use locally trained stand-in weights, not the '
            "authors'; their scores are an interface check, never a method result",
        ],
    }
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print(f'\nwrote {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
