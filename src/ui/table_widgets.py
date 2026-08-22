"""Rebuilding a table whose cells hold widgets, without leaving ghosts behind.

``QTableWidget.setCellWidget`` on a cell that already holds one destroys the old
widget with ``deleteLater``. Until the event loop collects it, the orphan keeps
its place in the viewport - and having lost its index, it lays out at the
origin, drawing itself over the first cell of the first row. It disappears once
the loop turns, which makes the artifact intermittent rather than absent: it
shows up in a screenshot, in a repaint that happens before the collection, and
in a test that never runs an event loop at all.

Removing each widget explicitly before repopulating settles it, so a rebuilt
table shows exactly the widgets it was given.
"""

from __future__ import annotations

from PyQt5.QtWidgets import QTableWidget, QWidget

__all__ = ["clear_cell_widgets", "reset_rows"]


def clear_cell_widgets(table: QTableWidget) -> None:
    """Detach every cell widget the table currently holds."""

    for row in range(table.rowCount()):
        for column in range(table.columnCount()):
            existing = table.cellWidget(row, column)
            if existing is not None:
                table.removeCellWidget(row, column)
                existing.setParent(None)


def reset_rows(table: QTableWidget, rows: int) -> None:
    """Clear the table's cell widgets, then size it to ``rows``."""

    clear_cell_widgets(table)
    table.setRowCount(int(rows))


def place(table: QTableWidget, row: int, column: int, widget: QWidget) -> None:
    """Put a widget in a cell, detaching whatever was there first."""

    existing = table.cellWidget(row, column)
    if existing is not None:
        table.removeCellWidget(row, column)
        existing.setParent(None)
    table.setCellWidget(row, column, widget)


__all__.append("place")
