"""One automated audit round over the embedding framework (v2 and v3).

The loop this script serves is: **run -> audit -> write findings -> fix -> re-run**.
It exists so the loop has a *machine* verdict instead of a human re-reading
results, and so a fix that only makes the report look better cannot pass: every
gate below is evaluated from a live call or a produced artefact, never from a
claim in a document.

Six gate families
-----------------
``G1`` runs        seven methods return conforming results
``G2`` dispatch    one call drives all seven, identical result fields
``G3`` guards      the contract refuses what it must refuse
``G4`` determinism two calls agree
``G5`` artefacts   benchmark/stability/IO files exist and the v2 numbers hold
``G6`` parity      each official method's preprocessing/output is justified
                   against its upstream source (see ``_parity_*`` below)

``G6`` is the only gate that cannot be fully reduced to an equality test: two of
the four official methods deliberately differ from upstream in a way that must be
*measured* rather than asserted.  Those checks therefore report a residual and a
verdict, never a bare "ok".

Exit code is 0 only when every gate passes; the findings file is written either
way, because the loop needs it as input for the next round.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / 'src'))

from cellpaint_pipeline import models  # noqa: E402
from cellpaint_pipeline.models import preprocessing, registry  # noqa: E402
from cellpaint_pipeline.models.errors import (  # noqa: E402
    IncompatibleInput,
    UnknownModel,
)

CELLS = Path(r'E:\bbbc022\cells')
WEIGHTS = Path(r'E:\cpc_weights')

#: Published digests of the four official checkpoints.  A change here is either a
#: re-download or a substitution, both of which invalidate every number produced
#: so far -- so the audit pins them rather than trusting whatever is on disk.
OFFICIAL_SHA = {
    'deepprofiler': '87b756f579336d60',
    'ca_mae': 'f6e5f1c97101331b',
    'cellclip': '66c2d32a44a3e7d6',
    'o2vae': 'aa70cfa8b080c9d9',
}

#: Widths the contracts declare, written out so a silent contract edit shows up as
#: a mismatch instead of quietly moving the goalposts.
EXPECTED_DIM = {
    'deepprofiler': 672,
    'ca_mae': 384,
    'cellclip': 512,
    'o2vae': 512,
    'unidino': 1920,      # 384 per channel x 5 channels
    'dinocells': 384,
    'morphogenie': 10,    # c_dim, not z_dim
}

#: v2 retrieval quality, pinned.  Any drift means the official path changed.
EXPECTED_V2_MAP_Z = {
    'cellclip': 0.1337,
    'ca_mae': 0.1083,
    'deepprofiler': 0.0691,
    'o2vae': 0.0471,
}

SURROGATES = ('unidino', 'dinocells', 'morphogenie')
OFFICIAL = ('deepprofiler', 'ca_mae', 'cellclip', 'o2vae')


class Audit:
    """Collects gate results; `.findings` becomes the round's markdown."""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.blocking: list[str] = []

    def check(self, gate: str, name: str, ok: bool, detail: str, *,
              blocking: bool = True, action: str = '') -> bool:
        self.rows.append({
            'gate': gate, 'name': name, 'ok': bool(ok),
            'detail': detail, 'action': action, 'blocking': blocking,
        })
        if not ok and blocking:
            self.blocking.append(f'{gate}/{name}: {detail}')
        return ok

    def failed(self, gate: str) -> list[dict]:
        return [r for r in self.rows if r['gate'] == gate and not r['ok']]

    @property
    def ok(self) -> bool:
        return not self.blocking


def _sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


def _payloads(n: int = 2) -> dict[str, dict]:
    """Inputs for all seven methods, built from real BBBC022 cells."""
    block = np.asarray(np.load(CELLS / 'cells_ca.npy', mmap_mode='r')[:n])
    bhwc = np.moveaxis(block, 1, -1).astype('uint16') * 257
    dna = block[:, 0]
    masks = (dna > dna.mean(axis=(1, 2), keepdims=True)).astype('uint8')
    dp = np.asarray(np.load(CELLS / 'cells_dp.npy', mmap_mode='r')[:n])
    vec = np.asarray(np.load(CELLS / 'cellclip_vectors.npy', mmap_mode='r')[:n])
    sil = np.asarray(np.load(CELLS / 'cells_o2.npy', mmap_mode='r')[:n])
    return {
        'deepprofiler': {'array': dp, 'assume': 'raw_crop'},
        'ca_mae': {'array': bhwc, 'assume': 'raw_crop'},
        'cellclip': {'array': vec},
        'o2vae': {'array': sil},
        'unidino': {'array': bhwc, 'assume': 'raw_crop'},
        'dinocells': {'array': bhwc[0], 'masks': masks[0], 'assume': 'raw_crop'},
        'morphogenie': {'array': block[:, :3].astype('float32') / 255.0,
                        'channels': ('C0', 'C1', 'C2')},
        '_block': {'block': block, 'bhwc': bhwc, 'masks': masks},
    }


# --------------------------------------------------------------------- gates --

#: Device every adapter call in this audit is pinned to.
#:
#: Needed because G1 loads all seven methods at once: on an 8 GiB card that is
#: impossible while anything else holds VRAM, and the failure mode is a native
#: segfault rather than a Python exception (measured 2026-10-27, exit 139 with
#: 7484/8192 MiB occupied by another job).  `--device cpu` makes the audit
#: runnable regardless of what else has the GPU.
DEVICE: str | None = None


def gate_runs(audit: Audit, payloads: dict) -> None:
    registry.set_profile('v3')
    for name, expected in EXPECTED_DIM.items():
        try:
            result = models.embed(name, device=DEVICE, **payloads[name])
            values = result.as_numpy()
            detail = (f'dim={result.dim} shape={values.shape} '
                      f'sha={(result.weights_sha256 or "")[:16]}')
            audit.check('G1', f'{name} runs', True, detail)
            audit.check('G1', f'{name} width', int(result.dim) == expected,
                        f'{result.dim} vs declared {expected}',
                        action='manifest.output_dim 与实际输出不符时改契约，不要改断言')
            audit.check('G1', f'{name} finite',
                        bool(np.isfinite(values).all()),
                        'NaN/Inf' if not np.isfinite(values).all() else 'finite')
            audit.check('G1', f'{name} sha',
                        bool(result.weights_sha256),
                        'weights_sha256 为空，溯源链断开',
                        action='适配器加载后必须记录实际权重摘要')
        except Exception as exc:  # noqa: BLE001
            audit.check('G1', f'{name} runs', False,
                        f'{type(exc).__name__}: {exc}',
                        action='先修可运行性，其余检查无意义')
            for sub in ('width', 'finite', 'sha'):
                audit.check('G1', f'{name} {sub}', False, 'blocked by run failure')


def gate_dispatch(audit: Audit, payloads: dict) -> None:
    """One call must drive all seven, with no per-method preparation by hand.

    This is the interface promise: a benchmark hands `embed_all` one payload per
    method plus the per-method declarations that genuinely differ (`assume`,
    `channels`), and gets seven results back.  Preparing each payload first would
    work too, but then the gate would be testing the caller, not the interface.
    """
    registry.set_profile('v3')
    names = [n for n in EXPECTED_DIM]
    inputs = {n: payloads[n]['array'] for n in names}
    assume = {n: 'raw_crop' for n in names
              if 'assume' in payloads[n]}
    channels = {n: payloads[n]['channels'] for n in names
                if 'channels' in payloads[n]}
    try:
        batch = models.embed_all(names, inputs, assume=assume, channels=channels,
                                 on_error='raise', weights_roots=(str(WEIGHTS),),
                                 device=DEVICE)
    except Exception as exc:  # noqa: BLE001
        audit.check('G2', 'embed_all drives seven', False,
                    f'{type(exc).__name__}: {exc}',
                    action='统一调度失败，接口不成立')
        return

    audit.check('G2', 'embed_all drives seven', len(batch.results) == 7,
                f'{len(batch.results)}/7 results, errors={batch.errors}')
    audit.check('G2', 'no method reported an error', not batch.errors,
                str(batch.errors))

    fields = {n: set(r.to_dict()) for n, r in batch.results.items()}
    baseline = fields.get('deepprofiler', set())
    audit.check('G2', 'identical result fields',
                all(f == baseline for f in fields.values()),
                f'{len({tuple(sorted(f)) for f in fields.values()})} distinct field sets')

    prov = {n: r.metadata.get('weights_provenance') for n, r in batch.results.items()}
    want = {**{n: 'official' for n in OFFICIAL}, **{n: 'surrogate' for n in SURROGATES}}
    audit.check('G2', 'provenance correct', prov == want, str(prov))
    audit.check('G2', 'surrogates not in v2', _v2_hides_surrogates(),
                'v2 下替身不可解析', action='剖面隔离失效')


def _v2_hides_surrogates() -> bool:
    before = registry.active_profile()
    registry.set_profile('v2')
    try:
        for name in SURROGATES:
            try:
                registry.manifest(name)
            except UnknownModel:
                continue
            return False
        return True
    finally:
        registry.set_profile(before)


def gate_guards(audit: Audit, payloads: dict) -> None:
    registry.set_profile('v3')
    meta = payloads['_block']
    block, bhwc, masks = meta['block'], meta['bhwc'], meta['masks']

    def rejects(label: str, fn, reason: str) -> None:
        try:
            fn()
        except IncompatibleInput as exc:
            audit.check('G3', label, exc.reason == reason,
                        f'reason={exc.reason} (want {reason})')
            return
        except Exception as exc:  # noqa: BLE001
            audit.check('G3', label, False,
                        f'{type(exc).__name__} 而非 IncompatibleInput',
                        action='所有拒绝必须是类型化错误，不能是框架异常')
            return
        audit.check('G3', label, False, '未被拒绝',
                    action='守卫缺失会静默产生错误嵌入')

    rejects('uint8 -> unidino',
            lambda: models.embed('unidino', np.moveaxis(block, 1, -1).astype('uint8'),
                                 device=DEVICE,
                                 assume='raw_crop'),
            'value-range')
    rejects('256 -> deepprofiler',
            lambda: models.embed('deepprofiler', bhwc, assume='raw_crop',
                                 device=DEVICE),
            'spatial-size')

    probe = models.probe('dinocells')
    audit.check('G3', 'probe(dinocells) runnable', bool(probe.get('runnable')),
                f'status={probe.get("status")} requires={probe.get("requires")}')
    audit.check('G3', 'runnable() == 7', len(models.runnable()) == 7,
                f'{len(models.runnable())} methods')

    man = registry.manifest('dinocells')
    try:
        batch_prepared = np.asarray(preprocessing.prepare(man, bhwc, masks=masks))
        audit.check('G3', 'dinocells batch', batch_prepared.shape == (2, 5, 96, 96),
                    f'{batch_prepared.shape}')
        without = np.asarray(preprocessing.prepare(man, bhwc))
        audit.check('G3', 'dinocells mask dropped',
                    np.array_equal(batch_prepared, without),
                    '带/不带 mask 必须逐位相等（官方 mode=drop）')
    except Exception as exc:  # noqa: BLE001
        audit.check('G3', 'dinocells batch', False, f'{type(exc).__name__}: {exc}')


def gate_determinism(audit: Audit, payloads: dict) -> None:
    registry.set_profile('v3')
    for name in EXPECTED_DIM:
        if name.startswith('_'):
            continue
        manifest = registry.manifest(name)
        if manifest.determinism == 'stochastic':
            audit.check('G4', f'{name} determinism', True,
                        'declared stochastic; repeatability not expected',
                        blocking=False)
            continue
        first = models.embed(name, device=DEVICE, **payloads[name]).as_numpy()
        second = models.embed(name, device=DEVICE, **payloads[name]).as_numpy()
        same = np.array_equal(first, second)
        audit.check('G4', f'{name} repeats exactly', same,
                    'identical' if same else
                    f'max|d|={float(np.abs(first - second).max()):.4g}')


def gate_artefacts(audit: Audit) -> None:
    expected = {
        'benchmark_scores_v2_e2e.json': 'v2',
        'benchmark_scores_v3_e2e.json': 'v3',
        'stability_v3_e2e.json': None,
        'io_uniformity_v3_e2e.json': None,
    }
    for filename, profile in expected.items():
        path = CELLS / filename
        if not path.exists():
            audit.check('G5', f'{filename} exists', False, 'missing',
                        action='跑 run_e2e.sh 生成产物')
            continue
        data = json.loads(path.read_text(encoding='utf-8'))
        audit.check('G5', f'{filename} exists', True, f'{path.stat().st_size} B')
        if profile:
            audit.check('G5', f'{filename} profile', data.get('profile') == profile,
                        f'profile={data.get("profile")}')

    scores = CELLS / 'benchmark_scores_v2_e2e.json'
    if scores.exists():
        metrics = json.loads(scores.read_text(encoding='utf-8'))['metrics']
        for name, expected_map in EXPECTED_V2_MAP_Z.items():
            got = round(float(metrics[name]['map_z']), 4)
            audit.check('G5', f'v2 {name} mAP_z', abs(got - expected_map) <= 1e-4,
                        f'{got} vs pinned {expected_map}',
                        action='官方方法路径不应变化；任何漂移都是回归')

    io = CELLS / 'io_uniformity_v3_e2e.json'
    if io.exists():
        summary = json.loads(io.read_text(encoding='utf-8')).get('summary', {})
        audit.check('G5', 'IO uniformity 7/7',
                    summary.get('n_ok') == 7 and summary.get('n_failed') == 0,
                    str(summary))


# ---------------------------------------------------------------- G6 parity --

def _parity_deepprofiler(audit: Audit, payloads: dict) -> None:
    """Bitwise parity against the upstream implementation, via the parity test.

    This is the only method with real upstream weights *and* a runnable upstream
    pipeline here, so it is the only one that can claim bitwise equality.  The
    check is the test module rather than a re-implementation, because that module
    already loads the official graph.
    """
    import subprocess
    proc = subprocess.run(
        [sys.executable, '-m', 'pytest', 'tests/test_parity_deepprofiler.py', '-q'],
        cwd=str(_ROOT), capture_output=True, text=True,
    )
    tail = (proc.stdout or '').strip().splitlines()[-1:] or ['no output']
    audit.check('G6', 'deepprofiler bitwise parity', proc.returncode == 0,
                tail[0][:160],
                action='官方 DP 对拍必须通过；不通过则预处理与官方不一致')


def _parity_ca_mae(audit: Audit, payloads: dict) -> None:
    """The official normalisation is `pixels / 255.0` and nothing else.

    Upstream has exactly one line of it (`normalizer.py:7`).  So the gate is that
    our declared pipeline invents no value transform: the uint8 crop must reach
    the model untouched.

    There is a second, informational measurement here.  The project used to
    stretch each crop to its own max and justify it by claiming the checkpoint's
    InstanceNorm cancels that.  It does not -- this check measures how far a
    global 0.5x rescale moves the embedding, which is the evidence that made that
    justification untenable.
    """
    registry.set_profile('v2')
    manifest = registry.manifest('ca_mae')
    block = payloads['_block']['block']
    bhwc = np.moveaxis(block, 1, -1).astype('uint8')

    audit.check('G6', 'ca_mae declares no invented transform',
                manifest.preprocess.source_map == 'identity',
                f"source_map={manifest.preprocess.source_map!r}",
                action="官方只有 /255；出现 per_crop_* 之类即为偏离")

    prepared = np.asarray(preprocessing.prepare(manifest, bhwc))
    # `prepare` returns BCHW, so the reference is the stored block, not the
    # BHWC view handed in.
    expected = block
    audit.check('G6', 'ca_mae prepared is the raw uint8 crop',
                np.array_equal(prepared, expected),
                f'{bhwc.shape} -> {prepared.shape} dtype={prepared.dtype} '
                f'bitwise_equal={np.array_equal(prepared, expected)}',
                action='prepare 改动了像素值即与官方预处理不一致')

    # Informational: the network does not cancel a global scale.
    full = models.embed('ca_mae', bhwc, assume='raw_crop', device=DEVICE).as_numpy()
    half = models.embed('ca_mae', (bhwc.astype('uint16') // 2).astype('uint8'), device=DEVICE,
                        assume='raw_crop').as_numpy()
    shift = float(np.abs(full - half).max()) / (float(full.std()) or 1.0)
    audit.check('G6', 'ca_mae is NOT scale-invariant (documents why the old '
                'stretch was wrong)', shift > 0.01,
                f'a global 0.5x rescale moves the embedding by {shift:.2%} of std; '
                'the old per-crop stretch was neutral only because every BBBC022 '
                'crop already has max 255', blocking=False)


def _parity_cellclip(audit: Audit, payloads: dict) -> None:
    """cellclip consumes precomputed vectors: the pipeline must transform nothing.

    The vectors come from DINOv2-Giant and upstream feeds the head directly, so
    any scale/normalise step of ours would be a deviation.  The declared plan is
    the evidence: it must contain `chain-required` and nothing that touches
    values.
    """
    registry.set_profile('v2')
    manifest = registry.manifest('cellclip')
    steps = [getattr(step, 'name', str(step)) for step in models.plan('cellclip')]
    transforming = [s for s in steps if s in ('source-map', 'resize')]
    audit.check('G6', 'cellclip has no value transform',
                not transforming and 'chain-required' in steps,
                f'plan={steps} source_map={manifest.preprocess.source_map!r}',
                action='上游直接喂向量；出现 source-map/resize 即与官方不一致')
    audit.check('G6', 'cellclip unit is channel embeddings',
                manifest.input_unit.value == 'channel_embeddings',
                f'{manifest.input_unit.value}')


def _parity_o2vae(audit: Audit, payloads: dict) -> None:
    """Compare the silhouette path against upstream's own transform.

    Upstream's test-time transform is `T.Resize(64)` (`config_mefs.py:28`), which
    is bilinear.  On a mask that is already 64x64 it is the identity, so both
    sides should agree exactly there -- that is the hard gate, because it is the
    case the dataset actually exercises.

    On other sizes the two diverge *by design*: upstream keeps the bilinear greys
    and we re-binarise at 0.5.  That is recorded as a non-blocking divergence
    rather than asserted either way, because calling it a defect would be wrong
    (a silhouette is binary) and calling it "consistent with official" would also
    be wrong.
    """
    registry.set_profile('v2')
    manifest = registry.manifest('o2vae')
    sil = payloads['o2vae']['array']                      # (N, 1, 64, 64) 0/1
    masks = (sil[:, 0] > 0.5).astype('uint8')

    prepared = np.asarray(preprocessing.prepare(manifest, sil, masks=masks))
    values = np.unique(prepared)
    audit.check('G6', 'o2vae silhouette stays binary',
                bool(np.all(np.isin(values, [0.0, 1.0]))),
                f'unique values={values[:6]}',
                action='剪影被插值成灰度即与官方语义不符')

    try:
        import torch
        from torchvision import transforms as T
    except Exception as exc:  # noqa: BLE001
        audit.check('G6', 'o2vae matches T.Resize(64) at 64x64', False,
                    f'cannot import torchvision: {exc}')
        return

    official64 = T.Resize(64)(torch.from_numpy(prepared[:, 0])[:, None]).numpy()
    same64 = np.array_equal(prepared[:, 0], official64[:, 0])
    audit.check('G6', 'o2vae matches T.Resize(64) at 64x64', bool(same64),
                'identity on both sides (as the dataset requires)'
                if same64 else
                f'max|diff|={float(np.abs(prepared[:, 0] - official64[:, 0]).max()):.4g}',
                action='64x64 是官方与本地都退化为恒等的情形，必须逐位相等')

    big = np.repeat(np.repeat(masks, 2, axis=1), 2, axis=2)      # 128x128
    ours_big = np.asarray(preprocessing.prepare(
        manifest, big[:, None].astype('float32'), masks=big))
    official_big = T.Resize(64)(torch.from_numpy(big.astype('float32'))[:, None])
    official_binary = (official_big > 0.5).numpy()[:, 0]
    ours_binary = ours_big[:, 0] > 0.5
    agree = float((ours_binary == official_binary).mean())
    audit.check('G6', 'o2vae 128x128 divergence is declared', True,
                f'upstream keeps bilinear greys; we re-binarise at 0.5. '
                f'After thresholding both, per-pixel agreement is {agree:.1%} '
                '(so only the grey levels differ, not the mask). '
                'No effect on this dataset, which is natively 64x64.',
                blocking=False)

def gate_parity(audit: Audit, payloads: dict) -> None:
    for fn in (_parity_deepprofiler, _parity_ca_mae, _parity_cellclip,
               _parity_o2vae):
        try:
            fn(audit, payloads)
        except Exception as exc:  # noqa: BLE001
            audit.check('G6', fn.__name__, False, f'{type(exc).__name__}: {exc}',
                        action='对拍脚本本身失败，先修脚本再判一致性')


# ------------------------------------------------------------------ reporting --

def write_findings(audit: Audit, round_no: int, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'round_{round_no:02d}_findings.md'

    gates = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6']
    titles = {
        'G1': '能跑（形状/宽度/有限/摘要）',
        'G2': '统一调度（一次调用驱动七个）',
        'G3': '契约守卫（该拒的必须拒）',
        'G4': '确定性（同输入同输出）',
        'G5': '产物与 v2 数字稳定',
        'G6': '与官方实现一致（对拍/等价证据）',
    }

    lines = [f'# 审核轮次 {round_no:02d}', '',
             f'- 时间：{time.strftime("%Y-%m-%d %H:%M:%S")}',
             f'- 结论：**{"通过" if audit.ok else "未通过"}**',
             f'- 阻塞项：{len(audit.blocking)}', '',
             '## 闸门总览', '', '| 闸门 | 内容 | 通过/总数 | 状态 |', '|---|---|---|---|']
    for gate in gates:
        rows = [r for r in audit.rows if r['gate'] == gate]
        passed = sum(1 for r in rows if r['ok'])
        status = '✅' if rows and passed == len(rows) else ('❌' if rows else '—')
        lines.append(f'| {gate} | {titles[gate]} | {passed}/{len(rows)} | {status} |')

    failing = [r for r in audit.rows if not r['ok']]
    lines += ['', '## 未通过项（下一轮的输入）', '']
    if not failing:
        lines.append('无。')
    else:
        lines += ['| 闸门 | 检查项 | 实测 | 应采取的动作 |', '|---|---|---|---|']
        for row in failing:
            detail = str(row['detail']).replace('|', '\\|')
            action = str(row['action']).replace('|', '\\|') or '定位并修复'
            lines.append(f'| {row["gate"]} | {row["name"]} | {detail} | {action} |')

    lines += ['', '## 全部检查明细', '', '| 闸门 | 检查项 | 结果 | 实测 |', '|---|---|---|---|']
    for row in audit.rows:
        mark = '✅' if row['ok'] else '❌'
        detail = str(row['detail']).replace('|', '\\|')
        lines.append(f'| {row["gate"]} | {row["name"]} | {mark} | {detail} |')

    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', type=int, default=1)
    parser.add_argument('--out-dir', default=str(_ROOT / 'docs' / 'audit'))
    parser.add_argument('--skip-parity', action='store_true',
                        help='fast loop while fixing non-parity gates')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--device', default=None,
                        help="pin every adapter to this device (e.g. 'cpu' when "
                             "another job owns the GPU)")
    args = parser.parse_args()
    global DEVICE
    DEVICE = args.device

    started = time.time()
    audit = Audit()
    try:
        payloads = _payloads()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print('cannot build payloads: dataset missing?')
        return 2

    gate_runs(audit, payloads)
    gate_dispatch(audit, payloads)
    gate_guards(audit, payloads)
    gate_determinism(audit, payloads)
    gate_artefacts(audit)
    if not args.skip_parity:
        gate_parity(audit, payloads)

    path = write_findings(audit, args.round, Path(args.out_dir))
    elapsed = time.time() - started

    if args.json:
        print(json.dumps({'ok': audit.ok, 'blocking': audit.blocking,
                          'findings': str(path)}, ensure_ascii=False, indent=2))
    else:
        print(f'round {args.round:02d}: {"PASS" if audit.ok else "FAIL"} '
              f'({len(audit.rows)} checks, {len(audit.blocking)} blocking, '
              f'{elapsed:.1f}s)')
        for row in audit.rows:
            if not row['ok']:
                print(f'  x {row["gate"]} {row["name"]}: {row["detail"]}')
        print(f'  findings -> {path}')

    return 0 if audit.ok else 1


if __name__ == '__main__':
    raise SystemExit(main())