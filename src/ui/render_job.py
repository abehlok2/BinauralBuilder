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
    estimate_peak_bytes,
    estimate_render_seconds,
    run_render,
)

__all__ = ["RenderJobManager", "RenderWorker"]


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
