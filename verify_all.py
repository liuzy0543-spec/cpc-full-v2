"""End-to-end verification for this release tree.

Three things must work before the tree can be called complete:

1. **the 19 documented skills** -- every one is executed for real, in dependency
   order, with each step's returned paths threaded into the next;
2. **the embedding layer** -- every method this release declares is run through
   ``models.embed`` on the shipped demo crops;
3. **the benchmark dispatcher** -- ``models`` + ``embedding_benchmark`` drive all
   declared methods through one call shape and write comparable artefacts.

Usage
-----
    python verify_all.py                # all three, in order
    python verify_all.py --skills       # skills only
    python verify_all.py --embeddings   # embedding methods only
    python verify_all.py --benchmark    # benchmark dispatch only

Exit code 0 only when every requested check passed.  A JSON report is written to
``verify/verify_report.json`` next to this file, so the result can be diffed
between runs instead of re-read from the console.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# torch bundles libiomp5md while the MKL numpy in this environment loads
# libomp; both register as the OpenMP runtime and abort the process unless the
# duplicate check is relaxed.  It must be set before numpy/torch are imported.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEMO = HERE / 'demo'


# --------------------------------------------------------------------------- #
# 1. skills
# --------------------------------------------------------------------------- #

def run_skills(config_path: Path, work_root: Path) -> dict:
    """Execute the 19 documented skills, chaining returned paths."""
    from cellpaint_pipeline import skills as skills_api
    from cellpaint_pipeline.config import ProjectConfig

    config = ProjectConfig.from_json(config_path)
    backend = DEMO / 'backend'
    prof = backend / 'profiling_backend'
    cp_tables = prof / 'outputs' / 'cellprofiler'

    records: dict[str, dict] = {}

    def step(key: str, name: str, out: str, **kwargs) -> dict:
        target = work_root / out
        target.mkdir(parents=True, exist_ok=True)
        started = time.time()
        try:
            result = skills_api.run_pipeline_skill(config, name, output_dir=target, **kwargs)
            payload = {
                'ok': bool(result.ok),
                'outputs': {k: str(v) if v is not None else None
                            for k, v in (result.primary_outputs or {}).items()},
                'mode': (result.details or {}).get('mode')
                        or (result.details or {}).get('export_mode'),
                'error': None,
            }
        except Exception as exc:  # noqa: BLE001 - a failure is data here
            payload = {
                'ok': False,
                'outputs': {},
                'mode': None,
                'error': f'{type(exc).__name__}: {exc}',
                'traceback': traceback.format_exc().strip().splitlines()[-6:],
            }
        payload['skill'] = name
        payload['elapsed_s'] = round(time.time() - started, 2)
        records[key] = payload
        flag = 'OK  ' if payload['ok'] else 'FAIL'
        print(f'  [{flag}] {name:<34} {payload["elapsed_s"]:>6.2f}s'
              + (f'  mode={payload["mode"]}' if payload['mode'] else ''))
        if not payload['ok'] and payload['error']:
            print(f'         {payload["error"][:170]}')
        return payload

    # data access ---------------------------------------------------------
    step('01', 'data-inspect-availability', '01-data-inspect')
    step('02', 'data-plan-download', '02-data-plan')
    from cellpaint_pipeline.data_access import build_data_request

    # A bare ``max_files`` is a *request* field, so it has to be built into a
    # DataRequest before the skill sees it.  Two objects keep the demo honest:
    # a real (small) download instead of a dry run that proves nothing.
    request = build_data_request(
        mode='gallery-source',
        dataset_id=config.data_access.default_dataset_id,
        source_id=config.data_access.default_source_id,
        max_files=2,
        dry_run=False,
        output_dir=work_root / '03-data-download' / 'downloads',
    )
    step('03', 'data-download', '03-data-download', data_request=request)

    # classical profiling -------------------------------------------------
    meas = step('04', 'cp-extract-measurements', '04-measurements')
    image_csv = meas['outputs'].get('image_table_path') or str(cp_tables / 'Image.csv')
    cells_csv = meas['outputs'].get('cells_table_path') or str(cp_tables / 'Cells.csv')
    single = step('05', 'cp-build-single-cell-table', '05-single-cell',
                  image_csv_path=Path(image_csv), object_table_path=Path(cells_csv),
                  object_table='Cells')
    agg = step('06', 'cyto-aggregate-profiles', '06-aggregate',
               single_cell_path=Path(single['outputs']['single_cell_path']))
    ann = step('07', 'cyto-annotate-profiles', '07-annotate',
               aggregated_path=Path(agg['outputs']['aggregated_path']))
    norm = step('08', 'cyto-normalize-profiles', '08-normalize',
                annotated_path=Path(ann['outputs']['annotated_path']))
    sel = step('09', 'cyto-select-profile-features', '09-feature-select',
               normalized_path=Path(norm['outputs']['normalized_path']))
    step('10', 'cyto-summarize-classical-profiles', '10-classical-summary',
         feature_selected_path=Path(sel['outputs']['feature_selected_path']),
         manifest_path=prof / 'data' / 'metadata' / 'image_manifest.csv')

    # segmentation --------------------------------------------------------
    prep = step('11', 'cp-prepare-segmentation-inputs', '11-seg-prepare')
    seg = step('12', 'cp-extract-segmentation-artifacts', '12-seg-artifacts')
    seg_root = work_root / '12-seg-artifacts'
    step('13', 'cp-generate-segmentation-previews', '13-seg-previews',
         workflow_root=seg_root)
    step('14', 'crop-export-single-cell-crops', '14-crops', workflow_root=seg_root)
    step('15', 'dp-export-deep-feature-inputs', '15-dp-export',
         workflow_root=seg_root,
         image_csv_path=Path(seg['outputs']['image_table_path']),
         nuclei_csv_path=Path(seg['outputs']['nuclei_table_path']),
         load_data_csv_path=Path(prep['outputs'].get('load_data_path')
                                 or prep['outputs'].get('load_data_csv_path')))
    step('16', 'dp-build-deep-feature-project', '16-dp-project',
         workflow_root=seg_root, export_root=work_root / '15-dp-export')
    run = step('17', 'dp-run-deep-feature-model', '17-dp-run',
               project_root=work_root / '16-dp-project')
    collect = step('18', 'dp-collect-deep-features', '18-dp-collect',
                   project_root=work_root / '16-dp-project')
    step('19', 'dp-summarize-deep-features', '19-dp-summary',
         single_cell_parquet_path=Path(collect['outputs']['single_cell_parquet_path']),
         well_aggregated_parquet_path=Path(collect['outputs']['well_aggregated_parquet_path']),
         manifest_path=Path(collect['outputs']['feature_manifest_path']))

    ok = sum(1 for r in records.values() if r['ok'])
    print(f'\n  skills: {ok}/{len(records)} ok')
    return {'ok': ok == len(records), 'passed': ok, 'total': len(records),
            'skills': records}


# --------------------------------------------------------------------------- #
# 2. embeddings
# --------------------------------------------------------------------------- #

def _embeddings_worker() -> dict:
    """Run every declared method through ``models.embed`` on the demo crops."""
    import numpy as np

    from cellpaint_pipeline import models
    from cellpaint_pipeline.models import registry

    sample = np.load(DEMO / 'data' / 'sample_inputs.npz')
    ca, dp = sample['ca'], sample['dp']
    o2, clip = sample['o2'], sample['cellclip']

    payloads = {
        'deepprofiler': (dp, {}),
        'ca_mae': (np.moveaxis(ca, 1, -1).astype('uint16') * 257, {'assume': 'raw_crop'}),
        'cellclip': (clip, {}),
        'o2vae': (o2, {}),
        'unidino': (np.moveaxis(ca, 1, -1).astype('uint16') * 257, {'assume': 'raw_crop'}),
        'dinocells': (np.moveaxis(ca, 1, -1).astype('uint16') * 257, {'assume': 'raw_crop'}),
        'morphogenie': (ca[:, :3].astype('float32') / 255.0,
                        {'channels': ('C0', 'C1', 'C2')}),
    }

    print(f'  profile: {registry.active_profile()}  '
          f'declared: {", ".join(models.declared())}')
    records = {}
    for name in models.declared():
        started = time.time()
        try:
            block, kwargs = payloads[name]
            result = models.embed(name, block, **kwargs)
            array = result.as_numpy()
            expected = registry.manifest(name).output_dim
            ok = array.shape[0] == len(block) and array.shape[1] == expected
            records[name] = {
                'ok': ok,
                'shape': list(array.shape),
                'expected_dim': expected,
                'provenance': result.metadata.get('weights_provenance'),
                'elapsed_s': round(time.time() - started, 2),
                'error': None if ok else f'dim {array.shape[1]} != {expected}',
            }
        except Exception as exc:  # noqa: BLE001
            records[name] = {'ok': False, 'error': f'{type(exc).__name__}: {exc}',
                             'elapsed_s': round(time.time() - started, 2)}
        r = records[name]
        flag = 'OK  ' if r['ok'] else 'FAIL'
        extra = f" dim={r['shape'][1]} {r['provenance']}" if r['ok'] else f" {r['error']}"
        print(f'  [{flag}] {name:<14} {r["elapsed_s"]:>6.2f}s{extra}')

    ok = sum(1 for r in records.values() if r['ok'])
    print(f'\n  embeddings: {ok}/{len(records)} ok')
    return {'ok': ok == len(records), 'passed': ok, 'total': len(records),
            'profile': registry.active_profile(), 'methods': records}


# --------------------------------------------------------------------------- #
# 3. benchmark dispatch
# --------------------------------------------------------------------------- #

def _benchmark_worker(work_root: Path) -> dict:
    """Drive every declared method through the one benchmark call shape."""
    import numpy as np

    from cellpaint_pipeline import embedding_benchmark as bench
    from cellpaint_pipeline import models

    sample = np.load(DEMO / 'data' / 'sample_inputs.npz')
    ca, dp = sample['ca'], sample['dp']
    o2, clip = sample['o2'], sample['cellclip']
    pixels = np.moveaxis(ca, 1, -1).astype('uint16') * 257

    inputs = {
        'deepprofiler': dp,
        'ca_mae': pixels,
        'cellclip': clip,
        'o2vae': o2,
        'unidino': pixels,
        'dinocells': pixels,
        'morphogenie': ca[:, :3].astype('float32') / 255.0,
    }
    units = {'cellclip': 'channel_embeddings'}
    channels = {'morphogenie': ('C0', 'C1', 'C2')}
    # Only the three methods that take an acquisition crop need the declared
    # pipeline; deepprofiler/o2vae are handed their model input, morphogenie is
    # pre-centred upstream, and cellclip takes vectors.
    assume = {
        'deepprofiler': None,
        'ca_mae': 'raw_crop',
        'cellclip': None,
        'o2vae': None,
        'unidino': 'raw_crop',
        'dinocells': 'raw_crop',
        'morphogenie': None,
    }

    out_dir = work_root / 'benchmark'
    plan = bench.build_benchmark_plan(
        models.declared(),
        output_dir=out_dir,
        unit=units,
        channels=channels,
        assume=assume,
        skip_unavailable=True,
    )
    metadata = {'row_ids': [str(x) for x in sample['cell_ids']]}
    started = time.time()
    result = bench.run_benchmark(plan, inputs, sample_metadata=metadata)
    elapsed = time.time() - started

    payload = bench.benchmark_result_to_dict(result)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'benchmark_report.json').write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding='utf-8')

    ran = sorted(payload.get('shapes', {}))
    errors = payload.get('errors', {}) or {}
    print(f'  benchmark ran: {", ".join(ran)}')
    if errors:
        for name, err in errors.items():
            print(f'    [FAIL] {name}: {str(err)[:150]}')

    ok = not errors and len(ran) == len(models.declared())
    print(f'\n  benchmark: {len(ran)}/{len(models.declared())} methods, '
          f'{len(errors)} error(s), {elapsed:.1f}s')
    return {'ok': ok, 'methods': ran, 'errors': errors,
            'output_dir': str(out_dir), 'elapsed_s': round(elapsed, 2)}


# --------------------------------------------------------------------------- #

def _run_child(kind: str, work_root: Path) -> dict:
    """Run a phase in a fresh interpreter.

    TensorFlow and torch both bring their own OpenMP/MKL runtime; importing both
    in one process makes ``torch/lib/fbgemm.dll`` fail to bind (WinError 127).
    A phase per process is also how these run in practice, so isolating them is
    the honest shape rather than a workaround specific to verification.
    """
    import subprocess

    command = [sys.executable, str(Path(__file__).resolve()), '--_child', kind,
               '--work-root', str(work_root)]
    proc = subprocess.run(command, capture_output=True, text=True,
                          encoding='utf-8', errors='replace')
    marker = '__VERIFY_JSON__'
    payload = None
    for line in proc.stdout.splitlines():
        if line.startswith(marker):
            payload = json.loads(line[len(marker):])
    if payload is None:
        print(proc.stdout[-1500:])
        print(proc.stderr[-1500:])
        return {'ok': False, 'error': f'child {kind} produced no report '
                                      f'(exit {proc.returncode})'}
    return payload


def _echo_child(payload: dict, kind: str) -> dict:
    if kind == 'embeddings':
        for name, record in (payload.get('methods') or {}).items():
            flag = 'OK  ' if record.get('ok') else 'FAIL'
            detail = (f"dim={record['shape'][1]} {record['provenance']}"
                      if record.get('ok') else str(record.get('error'))[:120])
            print(f"  [{flag}] {name:<14} {detail}")
        print(f"\n  embeddings: {payload.get('passed')}/{payload.get('total')} ok")
    elif kind == 'benchmark':
        print(f"  benchmark ran: {', '.join(payload.get('methods') or [])}")
        for name, err in (payload.get('errors') or {}).items():
            print(f"    [FAIL] {name}: {str(err)[:150]}")
        print(f"\n  benchmark: {len(payload.get('methods') or [])} methods, "
              f"{len(payload.get('errors') or {})} error(s)")
    if payload.get('error'):
        print(f"  [FAIL] {payload['error'][:200]}")
    return payload


def run_embeddings() -> dict:
    return _echo_child(_run_child('embeddings', HERE / 'verify' / 'run'), 'embeddings')


def run_benchmark_check(work_root: Path) -> dict:
    return _echo_child(_run_child('benchmark', work_root), 'benchmark')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--_child', default=None, help=argparse.SUPPRESS)
    parser.add_argument('--skills', action='store_true')
    parser.add_argument('--embeddings', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--config', default=str(HERE / 'configs' / 'project_config.demo.json'))
    parser.add_argument('--work-root', default=str(HERE / 'verify' / 'run'))
    args = parser.parse_args()

    work_root = Path(args.work_root).resolve()

    if args._child:
        # Child mode: compute one phase, print it, exit.  Nothing else runs, so
        # the parent's console stays a clean pass/fail table.
        if args._child == 'embeddings':
            payload = _embeddings_worker()
        elif args._child == 'benchmark':
            payload = _benchmark_worker(work_root)
        else:
            payload = {'ok': False, 'error': f'unknown child phase {args._child}'}
        print('__VERIFY_JSON__' + json.dumps(payload, default=str))
        return 0 if payload.get('ok') else 1

    everything = not (args.skills or args.embeddings or args.benchmark)
    work_root.mkdir(parents=True, exist_ok=True)

    report: dict = {'started_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                    'tree': str(HERE), 'checks': {}}
    failures = 0

    if everything or args.skills:
        print('== 19 documented skills ==')
        report['checks']['skills'] = run_skills(Path(args.config), work_root)
        failures += 0 if report['checks']['skills']['ok'] else 1
        print()

    if everything or args.embeddings:
        print('== embedding methods ==')
        report['checks']['embeddings'] = run_embeddings()
        failures += 0 if report['checks']['embeddings']['ok'] else 1
        print()

    if everything or args.benchmark:
        print('== benchmark dispatch ==')
        report['checks']['benchmark'] = run_benchmark_check(work_root)
        failures += 0 if report['checks']['benchmark']['ok'] else 1
        print()

    report['ok'] = failures == 0
    report_path = HERE / 'verify' / 'verify_report.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str),
                           encoding='utf-8')
    print(('ALL CHECKS PASSED' if report['ok'] else f'{failures} CHECK(S) FAILED')
          + f' -> {report_path}')
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
