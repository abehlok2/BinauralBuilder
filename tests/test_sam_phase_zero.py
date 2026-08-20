import json

import numpy as np
import pytest

from src.audio.sam_workbench.cli import main
from src.audio.sam_workbench.conventions import (
    cartesian_to_spherical_deg,
    db_to_linear,
    spherical_to_cartesian_m,
)
from src.audio.sam_workbench.model import (
    Project,
    ProjectValidationError,
    Source,
    load_project,
    project_from_dict,
    save_project,
)


@pytest.mark.parametrize(
    ("azimuth_deg", "expected"),
    [(0.0, (1.0, 0.0, 0.0)), (90.0, (0.0, 1.0, 0.0)), (-90.0, (0.0, -1.0, 0.0))],
)
def test_coordinate_cardinals(azimuth_deg, expected):
    point = spherical_to_cartesian_m(azimuth_deg, 0.0, 1.0)
    np.testing.assert_allclose(point, expected, atol=1e-15)
    actual_azimuth, actual_elevation, actual_radius = cartesian_to_spherical_deg(point)
    assert actual_azimuth == pytest.approx(azimuth_deg)
    assert actual_elevation == pytest.approx(0.0)
    assert actual_radius == pytest.approx(1.0)


def test_safe_defaults_and_atomic_round_trip(tmp_path):
    path = tmp_path / "session.sam.json"
    project = Project()
    save_project(project, path)
    assert load_project(path) == project
    assert not list(tmp_path.glob("*.tmp"))
    assert project.audio.sample_rate_hz == 44_100
    assert project.output.master_gain_db == -6.0
    assert db_to_linear(project.output.master_gain_db) == pytest.approx(0.5011872336)


def test_validation_reports_all_source_errors():
    project = Project(sources=(Source(id="same", amplitude_linear=-1), Source(id="same", duration_s=0)))
    with pytest.raises(ProjectValidationError) as caught:
        project_from_dict(project.to_dict())
    paths = {issue.path for issue in caught.value.issues}
    assert "sources[0].amplitude_linear" in paths
    assert "sources[1].id" in paths
    assert "sources[1].duration_s" in paths


def test_cli_creates_then_validates_project(tmp_path, capsys):
    path = tmp_path / "demo.sam.json"
    assert main(["new", str(path), "--name", "Reference SAM"]) == 0
    assert main(["validate", str(path)]) == 0
    assert json.loads(path.read_text())["name"] == "Reference SAM"
    assert "Valid SAM project" in capsys.readouterr().out
