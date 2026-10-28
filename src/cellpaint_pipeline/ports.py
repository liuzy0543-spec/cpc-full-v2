"""Infrastructure ports for side effects the pipeline needs to perform.

WHY THIS MODULE EXISTS
-----------------------
Three concrete side effects used to be performed inline, which coupled the
high-level workflow code to process/OS details:

1. ``subprocess.Popen`` in :mod:`cellpaint_pipeline.runner`
2. ``os.environ`` manipulation in
   :mod:`cellpaint_pipeline.adapters.deepprofiler_project`
3. resolving the ``PathName_*`` columns of a CellProfiler load-data table in
   :mod:`cellpaint_pipeline.segmentation_native`

Each of those is now expressed as a *port* (a small protocol plus a default
implementation) so the decision is made in exactly one place and can be
replaced - for example with an ASCII-staging file locator - without touching
the domain code.

DESIGN RULES FOR THIS MODULE
----------------------------
* The default implementations must reproduce the historical behaviour
  **byte for byte**.  Anything that changes an existing result belongs in a
  migration, not here.
* Ports are protocols plus pure helpers; they never import the domain layer.
* No module in the package may grow a new direct ``subprocess`` import.
"""
from __future__ import annotations

import os
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # pandas is imported lazily at runtime, like everywhere else.
    import pandas as pd


@dataclass(frozen=True)
class CommandSpec:
    """A fully resolved external command, ready to be executed.

    ``argv`` keeps the historical ``list[str]`` shape because it is stored
    verbatim inside :class:`cellpaint_pipeline.runner.ExecutionResult` and
    therefore ends up in skill manifests.
    """

    argv: list[str]
    cwd: Path | None = None
    env: dict[str, str] | None = None
    log_dir: Path | None = None
    label: str | None = None


@dataclass
class CommandOutcome:
    """Result of running a :class:`CommandSpec`.

    ``returncode`` is ``None`` when the process could not be started at all,
    which mirrors the historical ``OSError`` handling in the runner.
    """

    returncode: int | None
    log_path: Path | None = None
    output_tail: list[str] = field(default_factory=list)
    start_error: str | None = None

    @property
    def started(self) -> bool:
        return self.start_error is None


@runtime_checkable
class CommandRunnerPort(Protocol):
    """Executes a :class:`CommandSpec` and reports what happened."""

    def run(self, spec: CommandSpec, *, on_line=None) -> CommandOutcome:
        """Run ``spec`` and return its outcome.

        Implementations must stream stdout, honour ``spec.cwd`` and merge
        ``spec.env`` on top of :data:`os.environ` (never replace it), and
        must not raise for a non-zero return code - deciding what a non-zero
        code means is the caller's job.
        """


def utc_timestamp() -> str:
    """Return the log-file timestamp used by the runner."""
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def build_log_path(log_dir: Path, label: str) -> Path:
    """Build the ``<UTC ts>_<sanitised label>.log`` path used by the runner."""
    return log_dir / f'{utc_timestamp()}_{label}.log'


def merge_child_env(overrides: Mapping[str, str] | None) -> dict[str, str] | None:
    """Merge ``overrides`` on top of the current environment.

    ``None`` is returned when there is nothing to add so that callers can hand
    the value straight to :mod:`subprocess`, which then lets the child inherit
    the parent environment unchanged.
    """
    if not overrides:
        return None
    return {**os.environ, **overrides}


def build_pythonpath_env(package_root: Path, *, key: str = 'PYTHONPATH') -> dict[str, str]:
    """Prepend ``package_root`` to the ``PYTHONPATH`` of the child process.

    The existing value (if any) is kept but pushed behind ``package_root``,
    which is what the DeepProfiler adapter relies on to win over an
    incompatible copy of a third-party package.
    """
    existing = os.environ.get(key, '').strip()
    value = f'{package_root}{os.pathsep}{existing}' if existing else str(package_root)
    return {key: value}


def iter_load_data_path_pairs(dataframe: pd.DataFrame) -> Iterable[tuple[str, str]]:
    """Yield the ``(PathName, FileName)`` pairs referenced by a load-data table.

    Only ``FileName_*`` columns that have a matching ``PathName_*`` column and
    that actually carry values are considered, and duplicates are collapsed -
    this is the exact selection rule the segmentation layer has always used.
    """
    file_columns = [column for column in dataframe.columns if column.startswith('FileName_')]
    for file_column in file_columns:
        path_column = file_column.replace('FileName_', 'PathName_')
        if path_column not in dataframe.columns:
            continue
        pairs = dataframe[[file_column, path_column]].dropna().drop_duplicates()
        for filename, pathname in pairs.itertuples(index=False):
            yield str(pathname), str(filename)


@runtime_checkable
class FileLocatorPort(Protocol):
    """Turns the relative ``PathName_*`` entries of a load-data table into paths."""

    def resolve(self, pathname: str, filename: str) -> Path:
        """Return the concrete path of one referenced file."""


class CwdRelativeFileLocator:
    """Resolves load-data entries against the process working directory.

    This is the historical behaviour: ``PathName_*`` values in the demo
    tables are repository-relative, so they are only correct when the process
    runs from the repository root.  It stays the default to keep every
    existing run reproducible.
    """

    def resolve(self, pathname: str, filename: str) -> Path:
        return Path(pathname) / filename


class BaseDirFileLocator:
    """Resolves load-data entries against an explicit base directory.

    Pass this to :func:`validate_local_files` to make the implicit working
    directory dependency explicit instead of removing it silently.
    """

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = Path(base_dir)

    def resolve(self, pathname: str, filename: str) -> Path:
        candidate = Path(pathname)
        if candidate.is_absolute():
            return candidate / filename
        return self._base_dir / candidate / filename


def validate_local_files(dataframe, *, locator: FileLocatorPort | None = None,
                         error_type: type[Exception] = FileNotFoundError,
                         error_message: str = 'Missing local files referenced by source LoadData. Examples:\n') -> None:
    """Raise when the load-data table points at files that do not exist.

    The traversal order, the truncation to ten examples and the message
    wording are preserved exactly; only the resolution strategy is injectable.
    """
    resolved_locator = locator or CwdRelativeFileLocator()
    missing: list[str] = []
    for pathname, filename in iter_load_data_path_pairs(dataframe):
        path = resolved_locator.resolve(pathname, filename)
        if not path.exists():
            missing.append(str(path))
    if missing:
        preview = '\n'.join(missing[:10])
        raise error_type(error_message + preview)


def build_argv(program: Sequence[str], args: Sequence[str] | None = None) -> list[str]:
    """Concatenate a program and its arguments into a fresh argv list."""
    argv = [str(item) for item in program]
    if args:
        argv.extend(str(item) for item in args)
    return argv


class SubprocessCommandRunner:
    """Default :class:`CommandRunnerPort` backed by :mod:`subprocess`.

    It reproduces the historical runner semantics: stdout and stderr are
    merged, output is streamed line by line, the last twenty lines are kept
    for error reporting, and a log file is written when ``spec.log_dir`` is
    given.  Failures are reported through :class:`CommandOutcome` instead of
    being raised, so the error policy stays with the caller.
    """

    def __init__(self, *, tail_size: int = 20, echo: bool = True) -> None:
        self._tail_size = tail_size
        self._echo = echo

    def run(self, spec: CommandSpec, *, on_line=None) -> CommandOutcome:
        import re
        import shlex

        execution_label = spec.label or Path(spec.argv[0]).stem or 'run'
        log_path: Path | None = None
        log_handle = None
        if spec.log_dir is not None:
            spec.log_dir.mkdir(parents=True, exist_ok=True)
            safe_label = re.sub(r'[^A-Za-z0-9._-]+', '_', execution_label).strip('_') or 'run'
            log_path = build_log_path(spec.log_dir, safe_label)
            log_handle = log_path.open('w', encoding='utf-8')

        if self._echo:
            print(f"[cellpaint_pipeline] running: {shlex.join(spec.argv)}")
            if spec.cwd is not None:
                print(f"[cellpaint_pipeline] cwd: {spec.cwd}")
            if log_path is not None:
                print(f"[cellpaint_pipeline] log: {log_path}")

        tail: list[str] = []
        try:
            try:
                process = subprocess.Popen(
                    spec.argv,
                    cwd=str(spec.cwd) if spec.cwd else None,
                    env=merge_child_env(spec.env),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
            except OSError as exc:
                return CommandOutcome(returncode=None, log_path=log_path,
                                      output_tail=list(tail), start_error=str(exc))

            assert process.stdout is not None
            for line in process.stdout:
                if self._echo:
                    print(line, end='')
                if on_line is not None:
                    on_line(line)
                tail.append(line.rstrip())
                if len(tail) > self._tail_size:
                    del tail[0 : len(tail) - self._tail_size]
                if log_handle is not None:
                    log_handle.write(line)
            returncode = process.wait()
        finally:
            if log_handle is not None:
                log_handle.close()
        return CommandOutcome(returncode=returncode, log_path=log_path, output_tail=list(tail))


DEFAULT_COMMAND_RUNNER: SubprocessCommandRunner = SubprocessCommandRunner()
DEFAULT_FILE_LOCATOR: CwdRelativeFileLocator = CwdRelativeFileLocator()
