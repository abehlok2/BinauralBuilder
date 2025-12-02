import json
import glob
import os
import argparse
from pathlib import Path
from typing import List

def generate_presets_source(input_files: List[str]):
    binaural_presets = {}
    noise_presets = {}

    # If no input files provided, maybe default to nothing or all? 
    # User said "just particular ones", so if list is empty, we do nothing or warn.
    # But for convenience let's assume if specific files are given we use them.
    # If the user provides a mix of json and noise files, we handle them.

    files_to_process = []
    if not input_files:
        # Fallback to all if none specified? Or just print help?
        # "I don't want to add every single json file" implies we should NOT default to all.
        print("# No input files provided. Usage: python generate_presets_source.py file1.json file2.noise ...")
        return
    else:
        for f in input_files:
            # Expand globs if the shell didn't do it (Windows cmd might not)
            expanded = glob.glob(f)
            if not expanded:
                # If it's not a glob or file not found, just add it to try opening (might be a specific path)
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
                    else:
                        print(f"# {file_path} does not look like a binaural session preset (missing 'progression').")
            except Exception as e:
                print(f"# Error reading {file_path}: {e}")
        
        elif path_obj.suffix.lower() == '.noise':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    name = path_obj.stem
                    noise_presets[name] = data
            except Exception as e:
                print(f"# Error reading {file_path}: {e}")
        else:
            print(f"# Skipping unknown file type: {file_path}")

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
    parser = argparse.ArgumentParser(description="Generate presets.py from JSON/Noise files.")
    parser.add_argument("files", nargs="*", help="List of files to include in the presets.")
    args = parser.parse_args()
    
    generate_presets_source(args.files)
