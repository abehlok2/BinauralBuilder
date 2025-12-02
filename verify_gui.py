import sys
from PyQt5.QtWidgets import QApplication
from src.ui.session_builder_window import SessionBuilderWindow
from src.ui import themes

def main():
    app = QApplication(sys.argv)
    
    # Apply the Modern Dark theme
    themes.apply_theme(app, "Modern Dark")
    
    # Create dummy catalogs for initialization
    binaural_catalog = {
        "preset1": type('obj', (object,), {'label': 'Alpha Waves', 'id': 'preset1'}),
        "preset2": type('obj', (object,), {'label': 'Theta Meditation', 'id': 'preset2'})
    }
    noise_catalog = {
        "noise1": type('obj', (object,), {'label': 'Pink Noise', 'id': 'noise1'})
    }
    
    window = SessionBuilderWindow(
        binaural_catalog=binaural_catalog,
        noise_catalog=noise_catalog
    )
    window.show()
    
    print("Session Builder Window launched. Close the window to exit.")
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
