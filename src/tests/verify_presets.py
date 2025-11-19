
import sys
import os
from pathlib import Path

# Add project root to python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

from src.utils.voice_file import load_voice_preset
from src.utils.noise_file import load_noise_params

# Mock audio_engine to avoid numpy dependency
import sys
from unittest.mock import MagicMock
sys.modules["src.synth_functions.audio_engine"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["scipy"] = MagicMock()
sys.modules["scipy.signal"] = MagicMock()

from src.audio.session_model import build_binaural_preset_catalog

def test_preset_loading():
    print("Testing preset loading...")
    presets_dir = Path(project_root) / "src/presets/binaurals"
    if not presets_dir.exists():
        print(f"Error: Presets directory not found at {presets_dir}")
        return

    catalog = build_binaural_preset_catalog(preset_dirs=[presets_dir])
    
    print(f"Found {len(catalog)} presets.")
    
    json_presets = [p for p in catalog.values() if p.id.startswith("json:")]
    print(f"Found {len(json_presets)} JSON presets.")
    
    for p in json_presets:
        print(f"  - {p.label} ({p.id})")
        if "voices" in p.payload:
            print(f"    - Has {len(p.payload['voices'])} voices")
        else:
            print("    - ERROR: Missing 'voices' in payload")

    if len(json_presets) > 0:
        print("SUCCESS: JSON presets loaded.")
    else:
        print("FAILURE: No JSON presets loaded.")

if __name__ == "__main__":
    test_preset_loading()
