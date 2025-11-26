from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPalette, QColor
from dataclasses import dataclass

@dataclass
class Theme:
    palette_func: callable
    stylesheet: str = ""

def dark_palette() -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, QColor(0, 0, 0))
    palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    return palette

# Style sheet ensuring editable widgets use white text in the dark theme
GLOBAL_STYLE_SHEET_DARK = """

QTreeWidget {
    color: #ffffff;
}

"""
    
# Green cymatic theme derived from the example in README
GLOBAL_STYLE_SHEET_GREEN = """
/* Base Widget Styling */
QWidget {
    font-size: 10pt;
    background-color: #0a0a0a;
    color: #00ffaa;
    font-family: 'Consolas', 'Courier New', monospace;
}

/* Group Boxes */
QGroupBox {
    background-color: #1a1a1a;
    border: 1px solid rgba(0, 255, 136, 0.4);
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 8px;
}

QGroupBox::title {
    color: #00ffaa;
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 3px 0 3px;
    background-color: #1a1a1a;
}

/* Push Buttons */
QPushButton {
    background-color: rgba(0, 255, 136, 0.25);
    border: 1px solid #00ff88;
    color: #00ffaa;
    padding: 4px 12px;
    border-radius: 4px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: rgba(0, 255, 136, 0.4);
    border: 1px solid #00ffcc;
    box-shadow: 0 0 10px rgba(0, 255, 136, 0.6);
}

QPushButton:pressed {
    background-color: rgba(0, 255, 136, 0.6);
}

QPushButton:disabled {
    background-color: rgba(0, 136, 68, 0.2);
    border: 1px solid rgba(0, 255, 136, 0.2);
    color: rgba(0, 255, 136, 0.5);
}

/* Column Headers */
QHeaderView::section {
    background-color: #000000;
    color: #00ffaa;
}

QLineEdit, QComboBox, QSlider {
    background-color: #202020;
    border: 1px solid #555555;
    color: #ffffff;     /* use white text */
}
"""

# Light blue theme with a neutral light palette and blue highlights
GLOBAL_STYLE_SHEET_LIGHT_BLUE = """
QTreeWidget {
    color: #000000;
}
QLineEdit, QComboBox, QSlider {
    background-color: #ffffff;
    border: 1px solid #a0a0a0;
    color: #000000;
}
"""

# Material theme with teal and orange accents
GLOBAL_STYLE_SHEET_MATERIAL = """
QTreeWidget {
    color: #212121;
}
QGroupBox {
    background-color: #ffffff;
    border: 1px solid #d0d0d0;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 8px;
    padding-left: 8px;
    padding-right: 8px;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 4px 0 4px;
}
QPushButton {
    background-color: #009688;
    border: none;
    color: white;
    padding: 6px 16px;
    border-radius: 4px;
}
QPushButton:hover {
    background-color: #26a69a;
}
QPushButton:pressed {
    background-color: #00796b;
}
QLineEdit, QComboBox, QSlider {
    background-color: #ffffff;
    border: 1px solid #bdbdbd;
    color: #212121;
    border-radius: 4px;
}
"""

# --- Modern Dark Theme ---
GLOBAL_STYLE_SHEET_MODERN_DARK = """
/* Global Reset & Base */
QWidget {
    background-color: #1e1e1e;
    color: #e0e0e0;
    font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif;
    font-size: 10pt;
}

/* Group Box */
QGroupBox {
    background-color: #252526;
    border: 1px solid #3e3e42;
    border-radius: 6px;
    margin-top: 1.2em; /* Leave space for title */
    padding: 12px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    color: #007acc; /* Accent color */
    font-weight: bold;
    background-color: #1e1e1e; /* Match parent background to mask border */
}

/* Buttons */
QPushButton {
    background-color: #333337;
    border: 1px solid #3e3e42;
    color: #f0f0f0;
    padding: 6px 16px;
    border-radius: 4px;
    min-width: 60px;
}

QPushButton:hover {
    background-color: #3e3e42;
    border-color: #007acc;
}

QPushButton:pressed {
    background-color: #007acc;
    color: #ffffff;
    border-color: #007acc;
}

QPushButton:disabled {
    background-color: #252526;
    color: #6d6d6d;
    border-color: #333333;
}

/* Input Fields */
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {
    background-color: #333337;
    border: 1px solid #3e3e42;
    color: #f0f0f0;
    border-radius: 4px;
    padding: 4px;
    selection-background-color: #264f78;
}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #007acc;
}

/* Combo Box */
QComboBox {
    background-color: #333337;
    border: 1px solid #3e3e42;
    color: #f0f0f0;
    border-radius: 4px;
    padding: 4px;
    min-width: 6em;
}

QComboBox:hover {
    border-color: #007acc;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left-width: 0px;
    border-top-right-radius: 3px;
    border-bottom-right-radius: 3px;
}

/* Trees and Lists */
QTreeView, QListView, QTableWidget, QTableView {
    background-color: #252526;
    border: 1px solid #3e3e42;
    color: #e0e0e0;
    gridline-color: #3e3e42;
    selection-background-color: #37373d;
    selection-color: #ffffff;
    outline: 0;
}

QHeaderView::section {
    background-color: #333337;
    color: #e0e0e0;
    padding: 4px;
    border: 1px solid #3e3e42;
}

/* Scrollbars */
QScrollBar:vertical {
    border: none;
    background: #1e1e1e;
    width: 12px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #424242;
    min-height: 20px;
    border-radius: 6px;
    margin: 2px;
}

QScrollBar::handle:vertical:hover {
    background: #686868;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    border: none;
    background: #1e1e1e;
    height: 12px;
    margin: 0px;
}

QScrollBar::handle:horizontal {
    background: #424242;
    min-width: 20px;
    border-radius: 6px;
    margin: 2px;
}

QScrollBar::handle:horizontal:hover {
    background: #686868;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* Sliders */
QSlider::groove:horizontal {
    border: 1px solid #3e3e42;
    height: 6px;
    background: #252526;
    margin: 2px 0;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #007acc;
    border: 1px solid #007acc;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}

QSlider::handle:horizontal:hover {
    background: #1f8ad2;
}

/* Splitter */
QSplitter::handle {
    background-color: #3e3e42;
}

QSplitter::handle:hover {
    background-color: #007acc;
}

/* Progress Bar */
QProgressBar {
    border: 1px solid #3e3e42;
    border-radius: 4px;
    text-align: center;
    background-color: #252526;
    color: #e0e0e0;
}

QProgressBar::chunk {
    background-color: #007acc;
    width: 10px;
}

/* Tab Widget */
QTabWidget::pane {
    border: 1px solid #3e3e42;
    background-color: #252526;
}

QTabBar::tab {
    background: #2d2d30;
    border: 1px solid #3e3e42;
    padding: 6px 12px;
    margin-right: 2px;
    color: #cccccc;
}

QTabBar::tab:selected {
    background: #1e1e1e;
    border-bottom-color: #1e1e1e; /* Blend with pane */
    color: #007acc;
    font-weight: bold;
}

QTabBar::tab:hover {
    background: #3e3e42;
}
"""

def green_palette() -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(0x0a, 0x0a, 0x0a))
    palette.setColor(QPalette.WindowText, QColor(0x00, 0xff, 0xaa))
    palette.setColor(QPalette.Base, QColor(0x1a, 0x1a, 0x1a))
    palette.setColor(QPalette.AlternateBase, QColor(0x15, 0x20, 0x15))
    palette.setColor(QPalette.Text, QColor(0x00, 0xff, 0xaa))
    palette.setColor(QPalette.Button, QColor(0x00, 0x88, 0x44, 0x60))
    palette.setColor(QPalette.ButtonText, QColor(0x00, 0xff, 0xaa))
    palette.setColor(QPalette.Highlight, QColor(0x00, 0xff, 0x88, 0xaa))
    palette.setColor(QPalette.HighlightedText, QColor(0xff, 0xff, 0xff))
    palette.setColor(QPalette.Link, QColor(0x00, 0xff, 0xcc))
    return palette

def light_blue_palette() -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(240, 248, 255))
    palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(230, 240, 250))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
    palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
    palette.setColor(QPalette.Text, QColor(0, 0, 0))
    palette.setColor(QPalette.Button, QColor(225, 238, 255))
    palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Link, QColor(0, 122, 204))
    palette.setColor(QPalette.Highlight, QColor(0, 120, 215))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    return palette

def material_palette() -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(250, 250, 250))
    palette.setColor(QPalette.WindowText, QColor(33, 33, 33))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
    palette.setColor(QPalette.ToolTipText, QColor(33, 33, 33))
    palette.setColor(QPalette.Text, QColor(33, 33, 33))
    palette.setColor(QPalette.Button, QColor(238, 238, 238))
    palette.setColor(QPalette.ButtonText, QColor(33, 33, 33))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Link, QColor(0, 150, 136))
    palette.setColor(QPalette.Highlight, QColor(255, 87, 34))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    return palette

def modern_dark_palette() -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.WindowText, QColor(224, 224, 224))
    palette.setColor(QPalette.Base, QColor(37, 37, 38))
    palette.setColor(QPalette.AlternateBase, QColor(45, 45, 48))
    palette.setColor(QPalette.ToolTipBase, QColor(30, 30, 30))
    palette.setColor(QPalette.ToolTipText, QColor(224, 224, 224))
    palette.setColor(QPalette.Text, QColor(224, 224, 224))
    palette.setColor(QPalette.Button, QColor(51, 51, 55))
    palette.setColor(QPalette.ButtonText, QColor(240, 240, 240))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Link, QColor(0, 122, 204))
    palette.setColor(QPalette.Highlight, QColor(0, 122, 204))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    return palette

THEMES = {
    "Modern Dark": Theme(modern_dark_palette, GLOBAL_STYLE_SHEET_MODERN_DARK),
    "Dark": Theme(dark_palette, GLOBAL_STYLE_SHEET_DARK),
    "Green": Theme(green_palette, GLOBAL_STYLE_SHEET_GREEN),
    "light-blue": Theme(light_blue_palette, GLOBAL_STYLE_SHEET_LIGHT_BLUE),
    "Material": Theme(material_palette, GLOBAL_STYLE_SHEET_MATERIAL),
}

def apply_theme(app: QApplication, name: str):
    theme = THEMES.get(name)
    if not theme:
        # Fallback to Modern Dark if theme not found
        theme = THEMES["Modern Dark"]
    
    palette = theme.palette_func()
    app.setPalette(palette)
    app.setStyleSheet(theme.stylesheet)
