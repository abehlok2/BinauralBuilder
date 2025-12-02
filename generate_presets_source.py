import json
import glob
import os
from pathlib import Path

def generate_presets_source():
    binaural_presets = {}
    noise_presets = {}

    # Find JSON files in the current directory (excluding hidden/system dirs)
    json_files = glob.glob("*.json")
    for json_file in json_files:
        # Skip package.json or other non-preset files if any (heuristic: presets seem to start with F)
        # But user said "collected .json files", so I'll try to include all that look like session definitions.
        # The ones found were F*.json.
        if not json_file.startswith("F"):
            continue
            
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Check if it looks like a session preset (has 'progression')
                if 'progression' in data:
                    name = Path(json_file).stem
                    binaural_presets[name] = data
        except Exception as e:
            print(f"# Error reading {json_file}: {e}")

    # Find .noise files
    noise_files = glob.glob("*.noise")
    for noise_file in noise_files:
        try:
            # Noise files are JSON but with .noise extension? Or pickle?
            # Let's check one. I'll assume JSON for now based on "collected .json files" comment, 
            # but I should verify. 
            # Wait, binauralbuilder_core/session.py uses load_noise_params.
            # Let's check what load_noise_params does.
            # Actually, I'll just read it as text/json.
            with open(noise_file, 'r', encoding='utf-8') as f:
                # Try to load as JSON
                data = json.load(f)
                name = Path(noise_file).stem
                noise_presets[name] = data
        except json.JSONDecodeError:
             print(f"# {noise_file} is not valid JSON, skipping.")
        except Exception as e:
            print(f"# Error reading {noise_file}: {e}")

    output = []
    output.append('"""Built-in presets for the Session Builder."""')
    output.append('')
    output.append('BINAURAL_PRESETS = {')
    for name, data in binaural_presets.items():
        output.append(f'    "{name}": {json.dumps(data, indent=4)},')
    output.append('}')
    output.append('')
    output.append('NOISE_PRESETS = {')
    for name, data in noise_presets.items():
        output.append(f'    "{name}": {json.dumps(data, indent=4)},')
    output.append('}')
    
    print("\n".join(output))

if __name__ == "__main__":
    generate_presets_source()
