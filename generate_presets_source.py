import json
import glob
import os
import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Dict, Any

# Try to import existing presets to preserve them
# We assume the script is run from the project root
sys.path.append(str(Path.cwd() / "src"))

def load_existing_presets():
    binaural = {}
    noise = {}
    try:
        # We try to import the module. 
        # Note: This requires src/presets.py to be valid python.
        presets_path = Path.cwd() / "src" / "presets.py"
        if presets_path.exists():
            import presets
            # Reload in case it was already imported
            import importlib
            importlib.reload(presets)
            
            if hasattr(presets, "BINAURAL_PRESETS"):
                binaural = presets.BINAURAL_PRESETS.copy()
            if hasattr(presets, "NOISE_PRESETS"):
                noise = presets.NOISE_PRESETS.copy()
            print("# Loaded existing presets from src/presets.py")
    except ImportError:
        print("# src/presets.py not found or could not be imported. Starting fresh.")
    except Exception as e:
        print(f"# Warning: Could not load existing presets: {e}")

    return binaural, noise


def _resolve_color_params(noise_type: str, embedded_params: Dict[str, Any]) -> Dict[str, Any]:
    """Return colour parameters for ``noise_type``.

    Embedded parameters from the ``.noise`` file take precedence. If none
    exist, fall back to user-defined custom presets and finally the built-in
    defaults so the generated preset always carries the concrete spectral
    settings rather than just a colour name.
    """

    # Avoid circular imports at module load time
    from src.utils.colored_noise import (
        DEFAULT_COLOR_PRESETS,
        load_custom_color_presets,
        normalized_color_params,
    )

    if embedded_params:
        return normalized_color_params(noise_type, embedded_params)

    key = (noise_type or "").strip().lower()
    if not key:
        return normalized_color_params(noise_type, {})

    for name, preset in load_custom_color_presets().items():
        if name.lower() == key:
            return normalized_color_params(name, preset)

    for name, preset in DEFAULT_COLOR_PRESETS.items():
        if name.lower() == key:
            return normalized_color_params(name, preset)

    return normalized_color_params(noise_type, {})


def _normalize_existing_noise_presets(noise_presets: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    # Avoid importing noise utilities at module import time to keep the script
    # lightweight when simply inspecting its contents.
    from src.utils.colored_noise import normalized_color_params

    normalized = {}
    for name, params in noise_presets.items():
        merged = dict(params)
        color_params = merged.get("noise_parameters") or merged.get("color_params") or {}
        noise_type = merged.get("noise_type", color_params.get("name", ""))
        merged["noise_parameters"] = normalized_color_params(noise_type, color_params)
        merged.pop("color_params", None)
        normalized[name] = merged
    return normalized

def generate_presets_source(input_files: List[str], remove_list: List[str]):
    binaural_presets, noise_presets = load_existing_presets()
    noise_presets = _normalize_existing_noise_presets(noise_presets)

    files_to_process = []
    if input_files:
        for f in input_files:
            expanded = glob.glob(f)
            if not expanded:
                files_to_process.append(f)
            else:
                files_to_process.extend(expanded)

    for file_path in files_to_process:
        path_obj = Path(file_path)
        if not path_obj.exists():
            print(f"# File not found: {file_path}")
            continue
            
        if path_obj.suffix.lower() == '.json':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'progression' in data:
                        name = path_obj.stem
                        binaural_presets[name] = data
                        print(f"# Added/Updated binaural preset: {name}")
                    else:
                        print(f"# {file_path} missing 'progression', skipping.")
            except Exception as e:
                print(f"# Error reading {file_path}: {e}")
        
        elif path_obj.suffix.lower() == '.noise':
            try:
                # Load and normalize the noise params so colour settings are
                # fully expanded rather than relying only on a colour name.
                from src.utils.noise_file import load_noise_params

                params = load_noise_params(file_path)
                params.color_params = _resolve_color_params(
                    getattr(params, "noise_type", ""), getattr(params, "color_params", {})
                )
                data = asdict(params)
                name = path_obj.stem
                noise_presets[name] = data
                print(f"# Added/Updated noise preset: {name}")
            except Exception as e:
                print(f"# Error reading {file_path}: {e}")
        else:
            print(f"# Skipping unknown file type: {file_path}")

    # Handle removals
    if remove_list:
        for name in remove_list:
            if name in binaural_presets:
                del binaural_presets[name]
                print(f"# Removed binaural preset: {name}")
            elif name in noise_presets:
                del noise_presets[name]
                print(f"# Removed noise preset: {name}")
            else:
                print(f"# Preset to remove not found: {name}")

    # Generate Output
    import pprint
    output = []
    output.append('"""Built-in presets for the Session Builder."""')
    output.append('')
    output.append('BINAURAL_PRESETS = {')
    for name, data in sorted(binaural_presets.items()):
        formatted_data = pprint.pformat(data, indent=4, width=120)
        output.append(f'    "{name}": {formatted_data},')
    output.append('}')
    output.append('')
    output.append('NOISE_PRESETS = {')
    for name, data in sorted(noise_presets.items()):
        formatted_data = pprint.pformat(data, indent=4, width=120)
        output.append(f'    "{name}": {formatted_data},')
    output.append('}')
    
    # Write to src/presets.py directly
    output_path = Path("src/presets.py")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(output))
        print(f"# Successfully wrote to {output_path}")
    except Exception as e:
        print(f"# Error writing to {output_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage presets.py from JSON/Noise files.")
    parser.add_argument("files", nargs="*", help="List of files to add/update.")
    parser.add_argument("-r", "--remove", nargs="+", help="List of preset names to remove.")
    args = parser.parse_args()
    
    generate_presets_source(args.files, args.remove)
