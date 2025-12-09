
import os
import sys
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import subprocess
import json

# Ensure we can import the built module
sys.path.append(os.path.join(os.getcwd(), "src", "realtime_backend"))

def build_extension():
    print("Building Rust extension...")
    try:
        subprocess.check_call(["maturin", "develop", "--release"], cwd=os.path.join("src", "realtime_backend"))
    except Exception as e:
        print(f"Build failed: {e}")
        sys.exit(1)

def compute_slope(audio, sample_rate):
    f, Pxx = scipy.signal.welch(audio, fs=sample_rate, nperseg=4096)
    # Fit line to log-log spectrum between 100Hz and 10kHz
    mask = (f > 100) & (f < 10000)
    log_f = np.log10(f[mask])
    log_p = 10 * np.log10(Pxx[mask])
    
    slope, intercept = np.polyfit(log_f, log_p, 1)
    return slope # dB per decade

def verify():
    # Force rebuild
    build_extension()
    
    import realtime_backend

    sr = 44100
    duration = 1.0
    
    test_cases = [
        {"name": "White (Legacy)", "params": {"noise_type": "white"}, "expected_slope": 0, "tolerance": 2},
        {"name": "Pink (Legacy)", "params": {"noise_type": "pink"}, "expected_slope": -10, "tolerance": 2},
        {"name": "Brown (Legacy)", "params": {"noise_type": "brown"}, "expected_slope": -20, "tolerance": 3},
        {"name": "Blue (Legacy)", "params": {"noise_type": "blue"}, "expected_slope": 10, "tolerance": 3},
        {"name": "Purple (Legacy)", "params": {"noise_type": "purple"}, "expected_slope": 20, "tolerance": 3},
        {"name": "Deep Brown (Legacy)", "params": {"noise_type": "deep brown"}, "expected_slope": -30, "tolerance": 4},
        
        # Power Law tests via GEQ
        {"name": "Custom Pink (Exp=1.0)", "params": {"noise_type": "custom", "exponent": 1.0}, "expected_slope": -10, "tolerance": 3},
        {"name": "Custom Brown (Exp=2.0)", "params": {"noise_type": "custom", "exponent": 2.0}, "expected_slope": -20, "tolerance": 3},
        {"name": "Custom Blue (Exp=-1.0)", "params": {"noise_type": "custom", "exponent": -1.0}, "expected_slope": 10, "tolerance": 4},
    ]

    print("\n--- Verification Results ---")
    all_passed = True
    
    for case in test_cases:
        track_json = {
            "global": {"sample_rate": sr, "output_filename": "test.wav"},
            "progression": [{
                "duration": duration,
                "voices": [{
                    "synth_function_name": "noise",
                    "params": case["params"]
                }]
            }]
        }
        
        # Render
        # We need a direct way to get audio buffer or we write to file and read back
        # The python binding `render_sample_wav` writes to file.
        
        out_file = f"test_noise_{case['name'].replace(' ', '_')}.wav"
        try:
            realtime_backend.render_sample_wav(json.dumps(track_json), out_file)
            
            # Read back
            import soundfile as sf
            audio, _ = sf.read(out_file)
            if len(audio.shape) > 1:
                audio = audio[:, 0] # Use left channel
                
            slope = compute_slope(audio, sr)
            diff = abs(slope - case["expected_slope"])
            passed = diff <= case["tolerance"]
            
            status = "PASS" if passed else "FAIL"
            print(f"{case['name']:<25} | Expected: {case['expected_slope']:5.1f} dB/dec | Actual: {slope:5.1f} dB/dec | {status}")
            
            if not passed:
                all_passed = False
                
        except Exception as e:
            print(f"{case['name']}: Error - {e}")
            all_passed = False
        finally:
             if os.path.exists(out_file):
                 os.remove(out_file)

    if all_passed:
        print("\nAll noise types verified successfully.")
    else:
        print("\nSome tests failed.")

if __name__ == "__main__":
    verify()
