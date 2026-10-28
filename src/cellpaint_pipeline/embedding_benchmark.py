"""Run several embedding methods over one dataset and compare them.

WHY THIS IS A SEPARATE MODULE
-----------------------------
The embedding layer answers "what is this method's contract"; this module
answers "did the methods agree on anything".  Keeping the two apart matters
because the comparison has requirements the layer itself must not grow:

* **One cell must be the same cell in every method's table.**  Without a shared
  row identity there is nothing to compare, so alignment happens here, loudly.
* **A stochastic method is not a point estimate.**  The layer records that fact;
  this module decides what to do about it via the layer's repeat-and-aggregate.
* **A missing method must not destroy the run.**  Weights for several methods are
  unavailable on this machine, so the runner collects failures and reports them
  rather than aborting.

Nothing here imports a deep-learning framework: the heavy imports stay inside the
adapters, reached through ``models.embed``.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not score "which method is better".  That depends on the scientific
question, and a module that guessed would produce a number people would quote.
It produces aligned, provenance-carrying matrices plus a stability and
agreement summary, and stops there.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from cellpaint_pipeline.models import registry
from cellpaint_pipeline.models import embedding as embedding_layer

__all__ = [
    'BenchmarkPlan',
    'BenchmarkRunResult',
    'available_benchmark_methods',
    'build_benchmark_plan',
    'run_benchmark',
    'benchmark_result_to_dict',
]


#: Licence classes a caller may publish under.  Recorded per method because the
#: methods differ sharply and a published comparison must not silently mix them.
_LICENCE_GATE = {
    'permissive': 'code and weights both permit redistribution and commercial use',
    'non-commercial': 'usable for research; commercial use restricted',
    'no-derivatives': 'redistribution of modified forms restricted',
    'copyleft': 'derivative works must carry the same licence',
    'unknown': 'upstream does not state terms; confirm before publishing',
}


def classify_licence(licence: str) -> str:
    """Bucket a manifest's licence string for reporting.

    String matching, deliberately: the upstream strings are prose (and two of
    them contradict themselves), so a structured licence model would encode
    guesses.  The bucket is a prompt to check, not a legal conclusion.
    """
    text = licence.lower()
    if 'no licence' in text or 'unknown' in text:
        return 'unknown'
    if 'gpl' in text:
        return 'copyleft'
    if '-nd' in text or 'nd-' in text or 'noderiv' in text:
        return 'no-derivatives'
    if 'nc' in text or 'non-commercial' in text or 'noncommercial' in text:
        return 'non-commercial'
    return 'permissive'


@dataclass(frozen=True)
class BenchmarkPlan:
    """What to run, and where the results go."""

    methods: tuple[str, ...]
    output_dir: Path
    channels: tuple[str, ...] | None = None
    #: Declared input kind.  Either one value shared by every method, or a
    #: ``{method: unit}`` mapping.  The mapping form is required to benchmark
    #: methods with different input kinds together: CellCLIP consumes channel
    #: embeddings while the others consume pixel crops, so a single scalar cannot
    #: describe the batch and those runs used to be impossible in one plan.
    unit: str | dict[str, str] | None = None
    #: Column carrying the per-row identity used to align methods.  ``None``
    #: disables alignment, which is only appropriate for a single method.
    align_on: str | None = None
    #: ``True`` to keep going when a method cannot run.
    skip_unavailable: bool = True
    #: Device for torch-backed methods (``'cpu'``/``'cuda'``); ``None`` lets each
    #: adapter choose.  Exposed because a benchmark that silently runs one method
    #: on CPU and another on GPU is not comparing the same thing twice.
    device: str | None = None
    #: Maximum rows per forward pass.  ``None`` means one pass over the whole
    #: payload, which is only safe for small runs: 3,857 CA-MAE crops are 1.26 GB
    #: of uint8 and become a ~5 GB float32 tensor in a single pass, exhausting an
    #: 8 GiB card.  A benchmark over a real plate must set this.
    chunk_size: int | None = None
    #: How each payload should be read (``'raw_crop'`` or ``None``), as one value
    #: or a ``{method: assume}`` mapping.  Required because the methods disagree:
    #: a raw acquisition crop has to go through the declared preprocessing, while
    #: a method whose centring happens upstream (MorphoGenie) must be handed
    #: already-prepared values.  Without this the plan could not express a
    #: seven-method batch, because one choice would be wrong for one of them.
    assume: str | dict[str, str] | None = None
    invariants: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.methods:
            raise ValueError('BenchmarkPlan.methods must not be empty')
        unknown = [m for m in self.methods if m not in registry.available()]
        if unknown:
            raise ValueError(
                f'unknown method(s) {unknown}; available: {registry.available()}'
            )


@dataclass
class BenchmarkRunResult:
    """Outcome of one benchmark run."""

    plan: BenchmarkPlan
    #: ``{method: EmbeddingResult}`` for methods that produced embeddings.
    results: dict[str, Any] = field(default_factory=dict)
    #: ``{method: reason}`` for methods that could not run.
    errors: dict[str, str] = field(default_factory=dict)
    #: Row identity shared by every method that ran, in order.
    aligned_ids: list[Any] = field(default_factory=list)
    #: ``{method: (n_rows, n_dims)}``.
    shapes: dict[str, tuple[int, int]] = field(default_factory=dict)
    #: ``{method: spread}`` for stochastic methods, from the repeat runs.
    stability: dict[str, float] = field(default_factory=dict)
    #: ``{(a, b): mean cosine similarity}`` on the aligned rows, mean-centred.
    agreement: dict[tuple[str, str], float] = field(default_factory=dict)
    licence_gate: dict[str, str] = field(default_factory=dict)
    written: list[str] = field(default_factory=list)
    started_at: str = ''
    finished_at: str = ''


def available_benchmark_methods(*, require_weights: bool = True) -> list[str]:
    """Methods this machine can actually run.

    ``require_weights`` is True by default because a benchmark over methods
    without weights is not a benchmark.  Pass False to list every declaration.
    """
    if not require_weights:
        return list(registry.available())
    # `doctor` lives on the facade (it aggregates probe() over the registry), so
    # the package is imported here rather than the embedding module alone.
    from cellpaint_pipeline import models

    runnable = []
    for report in models.doctor():
        if report.get('runnable'):
            runnable.append(report['model'])
    return runnable


def build_benchmark_plan(
    methods: Sequence[str] | None = None,
    *,
    output_dir: Path | str,
    channels: Sequence[str] | None = None,
    unit: str | dict[str, str] | None = None,
    align_on: str | None = None,
    skip_unavailable: bool = True,
    device: str | None = None,
    chunk_size: int | None = None,
    assume: str | dict[str, str] | None = None,
) -> BenchmarkPlan:
    """Assemble a plan, defaulting to every runnable method.

    The invariants are computed rather than documented, so a printed plan states
    what the run actually guarantees instead of what the author intended.
    """
    if methods is None:
        methods = available_benchmark_methods()
    invariants = [
        'every method reports weights_sha256, verified against the file',
        'stochastic methods are repeated and their spread is recorded',
        'all methods are aligned on one shared row identity before comparison',
    ]
    # A mapping must survive as a mapping: collapsing it to a tuple of keys is
    # how every method ended up being told its channels were ('morphogenie',).
    if channels is None:
        resolved_channels: object = None
    elif isinstance(channels, Mapping):
        resolved_channels = dict(channels)
    else:
        resolved_channels = tuple(channels)
    return BenchmarkPlan(
        methods=tuple(methods),
        output_dir=Path(output_dir),
        channels=resolved_channels,
        unit=unit,
        align_on=align_on,
        skip_unavailable=skip_unavailable,
        device=device,
        chunk_size=chunk_size,
        assume=dict(assume) if isinstance(assume, Mapping) else assume,
        invariants=tuple(invariants),
    )


def _validate_row_counts(
    plan: BenchmarkPlan,
    inputs: Mapping[str, Any] | Any,
    sample_metadata: dict | None,
) -> None:
    """Fail before loading a checkpoint if the rows cannot line up.

    This used to surface from inside ``_align`` *after* every model had loaded and
    run: a mismatch between one method's output and ``sample_metadata`` cost a
    full pipeline (measured at 14.79s) and then wrote nothing, because the
    exception came before ``_write_artefacts``.  A scheduling layer should reject
    a mis-sized batch up front, where the only cost is the check itself.

    Only *knowable* counts are compared -- a payload without a ``shape`` is left
    to the contract layer.
    """
    problems: list[str] = []
    is_mapping = isinstance(inputs, Mapping)

    expected: int | None = None
    if sample_metadata:
        lengths = {
            column: len(values)
            for column, values in sample_metadata.items()
            if hasattr(values, '__len__')
        }
        distinct = sorted(set(lengths.values()))
        if len(distinct) > 1:
            problems.append(
                f'sample_metadata columns disagree on row count: {lengths}'
            )
        if distinct:
            expected = distinct[-1]
        if plan.align_on and plan.align_on not in sample_metadata:
            problems.append(
                f"align_on={plan.align_on!r} is not a sample_metadata column "
                f'(has {sorted(sample_metadata)})'
            )
    elif plan.align_on:
        problems.append(
            f"align_on={plan.align_on!r} requires sample_metadata, but none was given"
        )

    if expected is not None:
        for name in plan.methods:
            payload = inputs.get(name) if is_mapping else inputs
            if payload is None:
                continue
            shape = getattr(payload, 'shape', None)
            if not shape:
                continue
            rows = int(shape[0])
            if rows != expected:
                problems.append(
                    f"input for '{name}' has {rows} rows but sample_metadata has "
                    f'{expected}'
                )

    if problems:
        raise ValueError(
            'the benchmark cannot line up its rows: ' + '; '.join(problems)
        )


def run_benchmark(
    plan: BenchmarkPlan,
    inputs: Mapping[str, Any] | Any,
    *,
    sample_metadata: dict | None = None,
) -> BenchmarkRunResult:
    """Execute a plan and write its artefacts.

    Args:
        plan: what to run and where to write.
        inputs: ``{method: array}`` or one array for every method whose declared
            input unit matches ``plan.unit``.  The mapping form exists because
            the methods do not share an input type -- CellCLIP consumes
            per-channel embeddings while the rest consume pixel crops -- so a
            single payload would have to be wrong for one of them.
        sample_metadata: per-row annotations; also the source of alignment ids.
    """
    started = datetime.now(timezone.utc).isoformat()
    outcome = BenchmarkRunResult(plan=plan, started_at=started)

    _validate_row_counts(plan, inputs, sample_metadata)

    ran = embedding_layer.embed_all(
        plan.methods,
        inputs,
        unit=plan.unit,
        channels=plan.channels,
        assume=plan.assume,
        sample_metadata=sample_metadata,
        device=plan.device,
        chunk_size=plan.chunk_size,
        on_error='collect' if plan.skip_unavailable else 'raise',
    )
    outcome.errors = dict(ran.errors)
    outcome.results = dict(ran.results)

    for name, result in outcome.results.items():
        outcome.shapes[name] = (int(result.shape[0]), int(result.dim))
        spread = result.metadata.get('stochastic_max_spread')
        if spread is not None:
            outcome.stability[name] = float(spread)
        manifest_ = registry.manifest(name)
        outcome.licence_gate[name] = classify_licence(manifest_.license)

    outcome.aligned_ids = _align(outcome, plan, sample_metadata)
    outcome.agreement = _pairwise_agreement(outcome, plan, sample_metadata)

    plan.output_dir.mkdir(parents=True, exist_ok=True)

    # Order matters: `_write_artefacts` serialises this object, so any field set
    # *after* it is written is permanently empty in summary.json.  `finished_at`
    # and the written-path list were both being assigned afterwards, which meant
    # every summary recorded `finished_at: ""` and `written: []`.
    outcome.finished_at = datetime.now(timezone.utc).isoformat()
    expected_paths = [
        str(plan.output_dir / 'embeddings' / f'{name}.npz')
        for name in outcome.results
    ] + [str(plan.output_dir / 'summary.json')]
    outcome.written = expected_paths

    _write_artefacts(outcome)
    return outcome


# --------------------------------------------------------------- internals ---

def _row_ids(plan: BenchmarkPlan, sample_metadata: dict | None) -> list[Any] | None:
    if not plan.align_on or not sample_metadata:
        return None
    values = sample_metadata.get(plan.align_on)
    if values is None:
        raise ValueError(
            f'align_on={plan.align_on!r} is not present in sample_metadata '
            f'(has {sorted(sample_metadata)})'
        )
    return list(values)


def _align(
    outcome: BenchmarkRunResult, plan: BenchmarkPlan, sample_metadata: dict | None
) -> list[Any]:
    """Return the row identity shared by every method that ran.

    Intersection, not union: a method that saw fewer cells cannot be padded, and
    silently comparing different cell sets is the failure this guards against.
    """
    ids = _row_ids(plan, sample_metadata)
    if ids is None:
        # Without an alignment key the only safe claim is per-method row counts.
        return []
    if not outcome.results:
        return []

    common: set[Any] | None = None
    for name, result in outcome.results.items():
        n_rows = int(result.shape[0])
        if n_rows != len(ids):
            raise ValueError(
                f'{name!r} produced {n_rows} rows but sample_metadata has '
                f'{len(ids)} for {plan.align_on!r}; they must correspond'
            )
        common = set(ids) if common is None else (common & set(ids))

    ordered = [value for value in ids if value in (common or set())]
    # Deduplicate while preserving first-seen order.
    seen: set[Any] = set()
    unique: list[Any] = []
    for value in ordered:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def _pairwise_agreement(
    outcome: BenchmarkRunResult, plan: BenchmarkPlan, sample_metadata: dict | None
) -> dict[tuple[str, str], float]:
    """How similarly two methods arrange the *same cells*, as a 0..1 number.

    Different methods have different widths (384 vs 512 vs 672), so a per-row
    cosine between two embeddings is not defined -- the first version of this
    function tried it and failed on the broadcast.  What is defined is whether
    the two methods agree on the *geometry* of the cell set, which is what the RV
    coefficient measures: the normalised cross-covariance of two centred
    matrices, a multivariate generalisation of R^2.  1 means identical
    configuration up to rotation and scale, 0 means unrelated.

    Mean-centring is part of the definition, not a preprocessing choice: without
    it every embedding is dominated by one dataset-level direction and every
    pair scores near 1.

    This is a "did the run produce sane matrices that relate to each other"
    check, not a quality metric.  It says nothing about which method is better.
    """
    import numpy as np

    if len(outcome.results) < 2:
        return {}

    ids = _row_ids(plan, sample_metadata)
    aligned = set(outcome.aligned_ids) if outcome.aligned_ids else None

    matrices: dict[str, Any] = {}
    for name, result in outcome.results.items():
        values = np.asarray(result.as_numpy(), dtype=np.float64)
        if ids is not None and aligned is not None:
            keep = [i for i, value in enumerate(ids) if value in aligned]
            values = values[keep]
        matrices[name] = values

    names = sorted(matrices)
    agreement: dict[tuple[str, str], float] = {}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            a, b = matrices[left], matrices[right]
            if a.shape[0] != b.shape[0] or a.shape[0] < 2:
                # Cannot relate matrices with no shared rows.
                continue
            value = _rv_coefficient(a, b)
            if value is not None:
                agreement[(left, right)] = value
    return agreement


def _rv_coefficient(a: Any, b: Any) -> float | None:
    """RV coefficient of two centred matrices with (possibly) different widths.

    Computed from the Gram (linear kernel) matrices, which is what makes it work
    across widths::

        K_a = A_c A_c^T      (n x n, independent of the feature width)
        K_b = B_c B_c^T
        RV  = <K_a, K_b> / sqrt(<K_a, K_a> <K_b, K_b>)

    for mean-centred ``A_c``/``B_c``.  The ``n x n`` route matters: the naive form
    ``tr(S_ab S_ba)`` needs ``A_c B_c^T``, which is undefined when the widths
    differ (384 vs 512), so the covariance route cannot be used here at all.

    1 means the two methods place the cells in the same configuration up to an
    orthogonal transformation (rotation/reflection/permutation) and a global
    scale; 0 means unrelated.  It is *not* invariant under a general invertible
    linear map: rescaling the axes differently is enough to lower it, which was
    measured at ~0.72 for a 5-axis rescaling rather than the 1.0 an earlier
    version of this docstring claimed.  That is a property of the coefficient,
    not a defect in the computation -- but it means a low RV is evidence of a
    different geometry, while a high RV does not imply the methods are
    interchangeable.

    Returns ``None`` when either matrix is degenerate (every row identical),
    where the ratio is undefined rather than zero.
    """
    import numpy as np

    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)

    ka = a @ a.T
    kb = b @ b.T

    numerator = float((ka * kb).sum())
    denominator = float(np.sqrt((ka * ka).sum() * (kb * kb).sum()))
    if denominator <= 0.0:
        return None
    return numerator / denominator


def _write_artefacts(outcome: BenchmarkRunResult) -> list[str]:
    """Persist embeddings and the summary, returning the paths actually written.

    Verification rather than bookkeeping: a path is only reported after the file
    exists, so `written` cannot claim an artefact that a failed write left behind.
    """
    directory = outcome.plan.output_dir
    written: list[str] = []

    for name, result in outcome.results.items():
        target = directory / 'embeddings' / f'{name}.npz'
        result.save_npz(target)
        if target.is_file():
            written.append(str(target))

    summary_path = directory / 'summary.json'
    summary_path.write_text(
        json.dumps(benchmark_result_to_dict(outcome), indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    if summary_path.is_file():
        written.append(str(summary_path))
    return written


def benchmark_result_to_dict(outcome: BenchmarkRunResult) -> dict:
    """JSON-serialisable summary.  Embedding arrays are described, not inlined."""
    return {
        'started_at': outcome.started_at,
        'finished_at': outcome.finished_at,
        'output_dir': str(outcome.plan.output_dir),
        'methods_requested': list(outcome.plan.methods),
        'methods_run': sorted(outcome.results),
        'errors': dict(outcome.errors),
        'shapes': {k: list(v) for k, v in outcome.shapes.items()},
        'stability': dict(outcome.stability),
        'licence_gate': dict(outcome.licence_gate),
        'licence_gate_legend': _LICENCE_GATE,
        'agreement_rv_coefficient': {
            f'{a}|{b}': value for (a, b), value in outcome.agreement.items()
        },
        'agreement_note': (
            'RV coefficient of mean-centred embeddings: a 0..1 measure of whether '
            'two methods arrange the same cells similarly. It is defined across '
            'different widths, unlike a row-wise cosine. It is a sanity check, '
            'not a quality ranking.'
        ),
        'aligned_row_count': len(outcome.aligned_ids),
        'align_on': outcome.plan.align_on,
        'invariants': list(outcome.plan.invariants),
        'provenance': {
            name: {
                'weights_sha256': result.weights_sha256,
                'weights_path': result.weights_path,
                'license': result.license,
                'determinism': result.determinism,
                'feature_layer': result.feature_layer,
                'dim': result.dim,
            }
            for name, result in outcome.results.items()
        },
        'written': outcome.written,
    }
