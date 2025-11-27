from PyQt5.QtWidgets import QStyledItemDelegate, QStyle
from PyQt5.QtGui import QTextDocument, QAbstractTextDocumentLayout, QPalette
from PyQt5.QtCore import QSize, Qt, QRectF
import math

class HTMLDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        options = option
        self.initStyleOption(options, index)

        style = options.widget.style() if options.widget else QApplication.style()
        doc = QTextDocument()
        doc.setDocumentMargin(4) # Add padding
        doc.setHtml(options.text)
        
        # Set text color based on selection state
        if options.state & QStyle.State_Selected:
            ctx = QAbstractTextDocumentLayout.PaintContext()
            ctx.palette.setColor(QPalette.Text, options.palette.color(QPalette.HighlightedText))
        else:
            ctx = QAbstractTextDocumentLayout.PaintContext()
            ctx.palette.setColor(QPalette.Text, options.palette.color(QPalette.Text))

        options.text = "" # Clear text so default delegate doesn't paint it
        style.drawControl(QStyle.CE_ItemViewItem, options, painter)

        # Use the palette's text color for the default style
        text_color = options.palette.color(QPalette.Text).name()
        doc.setDefaultStyleSheet(f"body {{ color: {text_color}; }}")

        painter.save()
        painter.translate(options.rect.left(), options.rect.top())
        
        # Clip to the item's local rect (0, 0, width, height)
        # The painter is already translated, so we clip to the local bounds.
        ctx.clip = QRectF(0, 0, options.rect.width(), options.rect.height())
        
        # Ensure width is set for wrapping if needed
        if options.rect.width() > 0:
            doc.setTextWidth(options.rect.width())

        doc.documentLayout().draw(painter, ctx)
        painter.restore()

    def sizeHint(self, option, index):
        options = option
        self.initStyleOption(options, index)
        doc = QTextDocument()
        doc.setDocumentMargin(4) # Add padding
        
        # Use the same default stylesheet as paint to ensure metrics match
        text_color = options.palette.color(QPalette.Text).name()
        doc.setDefaultStyleSheet(f"body {{ color: {text_color}; }}")
        
        doc.setHtml(options.text)
        if options.rect.width() > 0:
            doc.setTextWidth(options.rect.width())
            
        # Document margin handles padding now
        height = math.ceil(doc.size().height())
        return QSize(int(doc.idealWidth()), height)
