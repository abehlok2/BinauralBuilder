"""Built-in binaural and noise presets.

This lightweight default set keeps the session builder and tests working even
when no generated preset file is present on disk.  The structures mirror the
output of :mod:`generate_presets_source.py` so downstream code can treat them
as if they were loaded from a generated module.
"""

BINAURAL_PRESETS = {
    "theta": {
        "progression": [
            {
                "start": 0,
                "duration": 300.0,
                "voices": [
                    {
                        "synth_function_name": "binaural_beat",
                        "params": {
                            "baseFreq": 200.0,
                            "beatFreq": 5.0,
                            "ampL": 0.5,
                            "ampR": 0.5,
                        },
                        "is_transition": False,
                        "description": "Built-in theta preset",
                    }
                ],
                "description": "Theta relaxation",
            }
        ]
    },
    "alpha": {
        "progression": [
            {
                "start": 0,
                "duration": 300.0,
                "voices": [
                    {
                        "synth_function_name": "binaural_beat",
                        "params": {
                            "baseFreq": 200.0,
                            "beatFreq": 10.0,
                            "ampL": 0.5,
                            "ampR": 0.5,
                        },
                        "is_transition": False,
                        "description": "Built-in alpha preset",
                    }
                ],
                "description": "Alpha focus",
            }
        ]
    },
}

# Noise presets are optional; keep this empty by default so tests relying on
# on-disk ``.noise`` files do not inadvertently pick up bundled content.
NOISE_PRESETS = {}

__all__ = ["BINAURAL_PRESETS", "NOISE_PRESETS"]
