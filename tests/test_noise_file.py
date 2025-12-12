from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.noise_file import NoiseParams, load_noise_params, save_noise_params


def test_noise_params_round_trip_includes_color_params(tmp_path: Path):
    color_params = {
        "exponent": 1.5,
        "high_exponent": 0.75,
        "distribution_curve": 1.2,
    }

    params = NoiseParams(noise_type="custom blue", color_params=color_params)
    target = tmp_path / "custom.noise"

    save_noise_params(params, str(target))
    loaded = load_noise_params(str(target))

    assert loaded.color_params == color_params
    assert loaded.noise_type == "custom blue"
