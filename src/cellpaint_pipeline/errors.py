"""Shared error types for the pipeline layers.

WHY THIS MODULE EXISTS
-----------------------
Three failure modes used to be reported by unrelated exception classes:
:class:`cellpaint_pipeline.runner.CommandExecutionError`,
:class:`cellpaint_pipeline.workflows.orchestration.WorkflowExecutionError`
and :class:`cellpaint_pipeline.delivery.SuiteExecutionError`.  A caller that
wanted "any pipeline failure" had to list all three, and each class repeated the
same message-assembly logic.

:class:`PipelineError` provides the common base plus the shared fields
(``label``, ``reason``, ``details``).  The existing classes now inherit from it
**without changing their names, constructor signatures or message wording**, so
``except CommandExecutionError`` and friends keep working exactly as before and
``except PipelineError`` becomes possible for new code.
"""
from __future__ import annotations

__all__ = [
    'PipelineError',
    'PipelineStepError',
]


class PipelineError(RuntimeError):
    """Base class for every recoverable pipeline failure.

    Subclasses keep their own ``__init__`` signature; this base only provides
    the structured attributes and a uniform ``__str__``.
    """

    def __init__(self, message: str, *, reason: str | None = None,
                 details: list[str] | None = None) -> None:
        self.reason = reason
        self.details = list(details or [])
        super().__init__(message)

    def __str__(self) -> str:
        return super().__str__()


class PipelineStepError(PipelineError):
    """A failure that can be attributed to one named step of a pipeline.

    This is the shape both the runner and the workflow layer need: a step
    label, a human readable reason and optional detail lines.
    """

    default_message = 'Step failed.'

    def __init__(self, message: str, *, step_label: str | None = None,
                 reason: str | None = None, details: list[str] | None = None) -> None:
        self.step_label = step_label
        super().__init__(message, reason=reason, details=details)

    def detail_lines(self) -> list[str]:
        """Return the detail lines, ready to be appended to a message."""
        return [f'- {detail}' for detail in self.details]
