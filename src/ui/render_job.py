"""Running an offline render on a thread, and telling the window about it.

The worker never touches a widget. It holds an immutable
:class:`~src.audio.sam_workbench.render_job.RenderSnapshot`, calls the Qt-free
render, and reports through signals; every widget change happens in a slot on
the GUI thread. That separation is the point of the exercise: the previous
export ran the render *on* the GUI thread and kept the window responsive by
calling ``QApplication.processEvents()`` from inside it, which meant the user
could edit the project - or start a second export - part-way through a render
that was reading the project as it went.
"""

from __future__ import annotations

from typing import Any

from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from src.audio.render_job import (
    RenderOutcome,
    RenderSnapshot,
    StepPreviewSnapshot,
    estimate_peak_bytes,
    estimate_render_seconds,
    run_render,
    run_step_preview,
)

__all__ = ["RenderJobManager", "RenderWorker", "StepPreviewJob", "StepPreviewWorker"]


class RenderWorker(QObject):
    """Renders one snapshot off the GUI thread."""

    progressed = pyqtSignal(float)
    finished = pyqtSignal(object)

    def __init__(self, snapshot: RenderSnapshot) -> None:
        super().__init__()
        self._snapshot = snapshot
        self._cancelled = False

    @property
    def snapshot(self) -> RenderSnapshot:
        return self._snapshot

    def cancel(self) -> None:
        """Ask the render to stop.

        Safe to call from the GUI thread while the worker runs: it sets a flag
        the render consults where it reports progress. Nothing is interrupted
        mid-buffer, so the stop happens at the next chunk boundary.
        """

        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    @pyqtSlot()
    def run(self) -> None:
        outcome = run_render(
            self._snapshot,
            progress=self.progressed.emit,
            should_cancel=lambda: self._cancelled,
        )
        self.finished.emit(outcome)


class RenderJobManager(QObject):
    """Owns the running render jobs and the threads they live on.

    One manager per window. It exists so the window does not have to keep
    thread lifetimes straight, and so closing the window has a single place to
    ask "is anything still running?".
    """

    started = pyqtSignal(object)
    progressed = pyqtSignal(object, float)
    finished = pyqtSignal(object, object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._jobs: list[tuple[QThread, RenderWorker]] = []
        #: Audio-seconds per wall-second from the last completed render, used
        #: to estimate the next one. A measurement beats a guess.
        self._throughput: float | None = None

    # --- introspection ------------------------------------------------------

    @property
    def active_count(self) -> int:
        return len(self._jobs)

    @property
    def busy(self) -> bool:
        return bool(self._jobs)

    def snapshots(self) -> tuple[RenderSnapshot, ...]:
        return tuple(worker.snapshot for _thread, worker in self._jobs)

    def estimate(self, snapshot: RenderSnapshot) -> dict[str, Any]:
        """What this render is expected to cost, before it starts."""

        return {
            "seconds": estimate_render_seconds(snapshot, self._throughput),
            "peakBytes": estimate_peak_bytes(snapshot),
            "measured": self._throughput is not None,
        }

    # --- lifecycle ----------------------------------------------------------

    def start(self, snapshot: RenderSnapshot) -> RenderWorker:
        """Begin rendering ``snapshot`` on its own thread."""

        thread = QThread()
        worker = RenderWorker(snapshot)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progressed.connect(
            lambda fraction, w=worker: self.progressed.emit(w, fraction)
        )
        worker.finished.connect(lambda outcome, w=worker: self._on_finished(w, outcome))

        self._jobs.append((thread, worker))
        thread.start()
        self.started.emit(worker)
        return worker

    def cancel_all(self) -> None:
        for _thread, worker in list(self._jobs):
            worker.cancel()

    def wait_for_idle(self, timeout_ms: int = 30_000) -> bool:
        """Block until every job has finished. Returns False on timeout.

        Used on close, after cancelling: a thread outliving the window it
        reports to is how a shutdown turns into a crash.
        """

        deadline = timeout_ms
        for thread, _worker in list(self._jobs):
            if not thread.wait(max(0, deadline)):
                return False
        return not self._jobs

    def _on_finished(self, worker: RenderWorker, outcome: RenderOutcome) -> None:
        metrics = getattr(outcome, "metrics", None)
        if outcome.succeeded and metrics and metrics.audio_seconds_per_second > 0:
            self._throughput = metrics.audio_seconds_per_second

        for index, (thread, candidate) in enumerate(list(self._jobs)):
            if candidate is worker:
                thread.quit()
                thread.wait(5_000)
                self._jobs.pop(index)
                thread.deleteLater()
                break
        worker.deleteLater()
        self.finished.emit(worker, outcome)


class StepPreviewWorker(QObject):
    """Generates one step's audition audio off the GUI thread.

    Separate from :class:`RenderWorker` because the two produce different
    things: a render writes a file and reports progress, an audition returns
    audio in memory and has no progress to report - the step generator takes no
    callback to hang one on. Sharing a worker would mean one of the two
    pretending to do something it cannot.
    """

    finished = pyqtSignal(object)

    def __init__(self, snapshot: StepPreviewSnapshot) -> None:
        super().__init__()
        self._snapshot = snapshot
        self._cancelled = False

    @property
    def snapshot(self) -> StepPreviewSnapshot:
        return self._snapshot

    def cancel(self) -> None:
        """Discard this preview when it finishes.

        Not an interruption: there is nowhere inside the step generator to stop
        it. What this prevents is a preview the user has already moved away
        from arriving and starting to play.
        """

        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    @pyqtSlot()
    def run(self) -> None:
        outcome = run_step_preview(
            self._snapshot, should_cancel=lambda: self._cancelled
        )
        self.finished.emit(outcome)


class StepPreviewJob(QObject):
    """Owns the thread one step preview runs on, and only the newest one.

    A user clicking through steps starts a preview per step. Only the last one
    is wanted, so an earlier one still running is cancelled - it finishes, and
    its result is dropped rather than played over the top of the one that was
    actually asked for.
    """

    finished = pyqtSignal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._current: StepPreviewWorker | None = None
        # Every thread still running, superseded or not. A superseded thread
        # must stay referenced until it finishes: dropping the last reference
        # to a running QThread destroys it mid-run, and Qt aborts the process.
        # That is what replacing the pair outright used to do.
        self._running: list[tuple[QThread, StepPreviewWorker]] = []

    @property
    def busy(self) -> bool:
        return bool(self._running)

    @property
    def active_count(self) -> int:
        return len(self._running)

    def start(self, snapshot: StepPreviewSnapshot) -> StepPreviewWorker:
        """Begin a preview, superseding any already running."""

        self.cancel()

        thread = QThread()
        worker = StepPreviewWorker(snapshot)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(lambda outcome, w=worker: self._on_finished(w, outcome))

        self._running.append((thread, worker))
        self._current = worker
        thread.start()
        return worker

    def cancel(self) -> None:
        """Ask everything in flight to discard its result."""

        for _thread, worker in list(self._running):
            worker.cancel()
        self._current = None

    def wait_for_idle(self, timeout_ms: int = 30_000) -> bool:
        for thread, _worker in list(self._running):
            if not thread.wait(max(0, timeout_ms)):
                return False
        return not self._running

    def _on_finished(self, worker: StepPreviewWorker, outcome) -> None:
        for index, (thread, candidate) in enumerate(list(self._running)):
            if candidate is worker:
                thread.quit()
                thread.wait(5_000)
                self._running.pop(index)
                thread.deleteLater()
                break
        if worker is self._current:
            self._current = None
        worker.deleteLater()
        # A cancelled preview is still reported, so a caller can re-enable its
        # controls; it simply carries no audio to play.
        self.finished.emit(outcome)
