from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

from cellpaint_pipeline.errors import PipelineStepError
from cellpaint_pipeline.ports import CommandRunnerPort, CommandSpec, SubprocessCommandRunner

__all__ = [
    'CommandExecutionError',
    'DEFAULT_COMMAND_RUNNER',
    'ExecutionResult',
    'run_command',
    'run_python_script',
    'set_default_command_runner',
]


@dataclass(frozen=True)
class ExecutionResult:
    label: str
    command: list[str]
    cwd: Path | None
    log_path: Path | None
    returncode: int


class CommandExecutionError(PipelineStepError):
    """Raised when a subprocess-backed pipeline step cannot complete successfully.

    The constructor signature and the rendered message are unchanged; the class
    now also inherits :class:`~cellpaint_pipeline.errors.PipelineError` so a
    caller can catch every pipeline failure with one ``except`` clause.
    """

    def __init__(
        self,
        *,
        label: str,
        command: list[str],
        cwd: Path | None,
        log_path: Path | None,
        returncode: int | None,
        output_tail: list[str] | None = None,
        reason: str | None = None,
    ) -> None:
        self.label = label
        self.command = list(command)
        self.cwd = cwd
        self.log_path = log_path
        self.returncode = returncode
        self.output_tail = list(output_tail or [])
        self.reason = reason
        PipelineStepError.__init__(self, self._build_message(), step_label=label,
                                   reason=reason, details=list(self.output_tail))

    def _build_message(self) -> str:
        lines = []
        if self.reason:
            lines.append(f"Step '{self.label}' could not start: {self.reason}")
        else:
            lines.append(f"Step '{self.label}' failed with exit code {self.returncode}.")
        lines.append(f"Command: {shlex.join(self.command)}")
        if self.cwd is not None:
            lines.append(f"CWD: {self.cwd}")
        if self.log_path is not None:
            lines.append(f"Log: {self.log_path}")
        if self.output_tail:
            lines.append('Output tail:')
            lines.extend(self.output_tail)
        return '\n'.join(lines)


# The process-execution port.  Swapping this out is the supported way to run
# pipeline steps somewhere other than a local subprocess (for example inside a
# container or a remote worker) without touching any workflow code.
DEFAULT_COMMAND_RUNNER: CommandRunnerPort = SubprocessCommandRunner()


def set_default_command_runner(runner: CommandRunnerPort) -> CommandRunnerPort:
    """Install ``runner`` as the process-execution port and return the previous one."""
    global DEFAULT_COMMAND_RUNNER
    previous = DEFAULT_COMMAND_RUNNER
    DEFAULT_COMMAND_RUNNER = runner
    return previous


def run_python_script(
    python_executable: str,
    script_path: Path,
    *,
    cwd: Path | None = None,
    extra_args: list[str] | None = None,
    log_dir: Path | None = None,
    label: str | None = None,
    env: dict[str, str] | None = None,
) -> ExecutionResult:
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    cmd = [python_executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)
    return run_command(cmd, cwd=cwd, log_dir=log_dir, label=label or script_path.stem, env=env)


def run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    log_dir: Path | None = None,
    label: str | None = None,
    env: dict[str, str] | None = None,
) -> ExecutionResult:
    if not command:
        raise ValueError('Command must not be empty.')
    if cwd is not None and not cwd.exists():
        raise FileNotFoundError(f"Working directory not found: {cwd}")

    spec = CommandSpec(
        argv=list(command),
        cwd=cwd,
        env=dict(env) if env else None,
        log_dir=log_dir,
        label=label,
    )
    outcome = DEFAULT_COMMAND_RUNNER.run(spec)

    if not outcome.started:
        raise CommandExecutionError(
            label=label or Path(command[0]).stem or 'run',
            command=command,
            cwd=cwd,
            log_path=outcome.log_path,
            returncode=None,
            reason=outcome.start_error,
        )
    if outcome.returncode != 0:
        raise CommandExecutionError(
            label=label or Path(command[0]).stem or 'run',
            command=command,
            cwd=cwd,
            log_path=outcome.log_path,
            returncode=outcome.returncode,
            output_tail=outcome.output_tail,
        )

    return ExecutionResult(
        label=label or Path(command[0]).stem or 'run',
        command=command,
        cwd=cwd,
        log_path=outcome.log_path,
        returncode=outcome.returncode,
    )
