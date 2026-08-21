"""The HRTF Lab: controls, asset inspection, auditioning, ratings, ranking.

The rules this file protects:

* every control the specification asks for exists, and stores the value the
  renderer actually understands;
* an absent dataset degrades to "missing resources" instead of raising;
* the HD650 correction is never applied unless it was explicitly chosen;
* ratings are kept per profile, so one voice can prefer a different subject
  from another;
* the dataset stays wherever the user installed it - only its path and the
  subject identifier reach the project.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("pytestqt", reason="GUI tests need pytest-qt")

from src.audio.sam_workbench.hrtf import storage  # noqa: E402
from src.audio.sam_workbench.hrtf.subject_test import (  # noqa: E402
    AUDITION_AZIMUTHS_DEG,
    AUDITION_ELEVATIONS_DEG,
    RATING_CRITERIA,
    SIGNAL_TYPES,
)
from src.ui.hrtf_coverage_plot import CoveragePlotWidget  # noqa: E402
from src.ui.sam_hrtf_lab import AuditionWorker, SamHrtfLab  # noqa: E402

FIXTURE = str(Path(__file__).resolve().parent / "fixtures" / "synthetic_sonicom_hrir.sofa")


@pytest.fixture
def isolated_storage(tmp_path, monkeypatch):
    """Keep ratings out of the developer's real application-data folder."""

    storage.set_data_root(tmp_path)
    yield tmp_path
    storage.set_data_root(None)


@pytest.fixture
def lab(qtbot, isolated_storage):
    widget = SamHrtfLab()
    qtbot.addWidget(widget)
    return widget


def combo_values(combo):
    return [combo.itemData(index) for index in range(combo.count())]


# --- controls ---------------------------------------------------------------


def test_every_requested_control_is_present(lab):
    assert combo_values(lab.dataset_combo) == ["SONICOM"]
    assert combo_values(lab.source_combo) == ["measured", "synthetic", "kemar"]
    assert combo_values(lab.processing_combo) == [
        "Windowed",
        "FreeFieldComp",
        "FreeFieldCompMinPhase",
    ]
    assert combo_values(lab.rate_combo) == [44_100, 48_000, 96_000]
    assert combo_values(lab.delay_combo) == ["bake_delay_into_ir", "preserve_external_delay"]
    assert combo_values(lab.interpolation_combo) == [
        "nearest",
        "three_neighbor",
        "spherical_triangular",
        "delay_magnitude",
        "spherical_harmonic",
    ]
    assert combo_values(lab.headphone_combo) == [
        "off",
        "sonicom_hd650",
        "user_impulse_response",
        "user_equalization_curve",
    ]


def test_the_audition_grid_offers_the_requested_directions_and_signals(lab):
    assert combo_values(lab.azimuth_combo) == [float(x) for x in AUDITION_AZIMUTHS_DEG]
    assert combo_values(lab.elevation_combo) == [float(x) for x in AUDITION_ELEVATIONS_DEG]
    assert combo_values(lab.signal_combo) == list(SIGNAL_TYPES)
    assert set(lab._criteria_sliders) == set(RATING_CRITERIA)


def test_interpolation_modes_are_listed_in_the_order_they_should_be_adopted(lab):
    """Nearest first: it is the one mode that cannot smear a transient."""

    assert lab.interpolation_combo.itemData(0) == "nearest"
    assert "Nearest" in lab.interpolation_combo.itemText(0)


# --- parameter round trip ---------------------------------------------------


def test_params_round_trip_through_the_project_dictionary(lab):
    lab.set_params(
        {
            "hrtfAsset": FIXTURE,
            "hrtfOptions": {
                "interpolation": "spherical_triangular",
                "delayPolicy": "preserve_external_delay",
                "sampleRateHz": 96_000,
                "processing": "Windowed",
                "crossfadeMs": 8.0,
                "headphoneCorrection": "off",
            },
        }
    )
    assert lab.interpolation_combo.currentData() == "spherical_triangular"
    assert lab.delay_combo.currentData() == "preserve_external_delay"
    assert lab.rate_combo.currentData() == 96_000
    assert lab.processing_combo.currentData() == "Windowed"
    assert lab.crossfade_spin.value() == pytest.approx(8.0)

    options = lab.params()["hrtfOptions"]
    assert options["interpolation"] == "spherical_triangular"
    assert options["delayPolicy"] == "preserve_external_delay"
    assert options["sampleRateHz"] == 96_000
    assert options["schemaVersion"] == 1


def test_only_the_path_and_subject_identify_the_dataset(lab):
    """The measurements stay where the user installed them."""

    lab.set_params({"hrtfAsset": FIXTURE})
    params = lab.params()

    assert params["hrtfAsset"] == FIXTURE
    assert params["hrtfSubject"]
    assert params["hrtfAssetHash"]
    # No impulse responses, positions, or other bulk data reach the project.
    serialised = repr(params)
    assert "Data.IR" not in serialised
    assert len(serialised) < 2000


def test_unknown_option_keys_survive_a_round_trip(lab):
    """An option written by a newer build must not be dropped here."""

    lab.set_params({"hrtfAsset": FIXTURE, "hrtfOptions": {"somethingNewer": 42}})
    assert lab.params()["hrtfOptions"]["somethingNewer"] == 42


# --- asset inspection -------------------------------------------------------


def test_a_valid_asset_reports_normal_quality_and_plots_its_coverage(lab):
    lab.set_params({"hrtfAsset": FIXTURE})

    assert lab.quality_label.text() == "normal"
    assert lab.coverage_plot.measurement_count == 144
    report = lab.report_view.toPlainText()
    assert "sha256" in report
    assert "Spherical coverage:" in report


def test_a_missing_asset_degrades_instead_of_raising(lab, tmp_path):
    lab.set_params({"hrtfAsset": str(tmp_path / "not-here.sofa")})

    assert lab.quality_label.text() == "missing resources"
    assert lab.coverage_plot.measurement_count == 0
    # The other renderers are still usable, and the message must say so.
    assert "renderers remain available" in lab.report_view.toPlainText()


def test_no_library_folder_is_a_message_not_an_error(lab):
    lab.library_edit.setText("")
    lab._refresh_library()

    assert "No library folder set" in lab.status_label.text()
    assert lab.subject_combo.count() >= 1


def test_the_library_defaults_outside_the_repository(lab):
    """The dataset is licensed and large; it does not live in the project."""

    lab._use_default_library()
    root = Path(lab.library_edit.text())
    repository = Path(__file__).resolve().parents[2]

    assert repository not in root.parents and root != repository


def test_subjects_are_discovered_and_filtered_by_source(lab, tmp_path):
    for name in (
        "P0001_FreeFieldComp_48kHz.sofa",
        "P0002_FreeFieldComp_48kHz.sofa",
        "KB0065_FreeFieldComp_48kHz.sofa",
    ):
        (tmp_path / name).write_bytes(b"")
    lab.library_edit.setText(str(tmp_path))
    lab._refresh_library()
    lab.processing_combo.setCurrentIndex(lab.processing_combo.findData("FreeFieldComp"))
    lab.rate_combo.setCurrentIndex(lab.rate_combo.findData(48_000))

    lab.source_combo.setCurrentIndex(lab.source_combo.findData("measured"))
    measured = [lab.subject_combo.itemText(i).split(" ")[0] for i in range(lab.subject_combo.count())]
    assert measured == ["P0001", "P0002"]

    lab.source_combo.setCurrentIndex(lab.source_combo.findData("kemar"))
    kemar = [lab.subject_combo.itemText(i) for i in range(lab.subject_combo.count())]
    assert kemar[0].startswith("KB0065")
    assert "large ears" in kemar[0]


# --- auditioning ------------------------------------------------------------


def test_an_audition_places_the_source_on_the_chosen_side(lab):
    lab.set_params({"hrtfAsset": FIXTURE})
    options = lab.audition_options()

    left, rate = AuditionWorker(FIXTURE, (90.0, 0.0), "pink_noise_burst", options, 0.25).render()
    right, _ = AuditionWorker(FIXTURE, (-90.0, 0.0), "pink_noise_burst", options, 0.25).render()

    def ild_db(audio):
        power = np.sqrt(np.mean(np.square(audio), axis=1))
        return 20.0 * np.log10(power[0] / max(power[1], 1e-12))

    assert rate > 0
    assert ild_db(left) > 6.0
    assert ild_db(right) < -6.0
    assert ild_db(left) == pytest.approx(-ild_db(right), abs=0.5)


@pytest.mark.parametrize("signal_type", SIGNAL_TYPES)
def test_every_signal_type_renders(lab, signal_type):
    lab.set_params({"hrtfAsset": FIXTURE})
    # session_material auditions the voice's own audio, so it needs some.
    material = np.sin(np.linspace(0.0, 400.0, 8192)) if signal_type == "session_material" else None
    audio, rate = AuditionWorker(
        FIXTURE, (30.0, 0.0), signal_type, lab.audition_options(), 0.2, material=material
    ).render()

    assert audio.shape[0] == 2
    assert audio.shape[1] > 0
    assert np.all(np.isfinite(audio))


def test_session_material_comes_from_the_rendered_preview(lab):
    """A subject may suit test signals and still be wrong for real material."""

    assert not lab.has_session_material
    lab.set_params({"hrtfAsset": FIXTURE})
    lab.signal_combo.setCurrentIndex(lab.signal_combo.findData("session_material"))

    # Asking for it before there is any must explain, not raise.
    lab.start_audition()
    assert "Render a preview first" in lab.status_label.text()

    # Channel-major stereo from the preview is downmixed to mono material.
    stereo = np.stack([np.linspace(-1.0, 1.0, 4096), np.linspace(1.0, -1.0, 4096)])
    lab.set_session_material(stereo, 44_100)
    assert lab.has_session_material
    assert lab._session_material.shape == (4096,)
    assert lab._session_material == pytest.approx(np.zeros(4096), abs=1e-12)


def test_filters_that_exclude_everything_say_which_ones(lab, tmp_path):
    """"No subjects" with files present means wrong filter, not wrong folder."""

    (tmp_path / "P0001_FreeFieldComp_48kHz.sofa").write_bytes(b"")
    lab.library_edit.setText(str(tmp_path))
    lab._refresh_library()
    lab.processing_combo.setCurrentIndex(lab.processing_combo.findData("Windowed"))

    assert "none match" in lab.status_label.text()
    assert "Windowed" in lab.status_label.text()


def test_the_default_processing_variant_is_the_one_work_starts_from(lab):
    assert lab.processing_combo.currentData() == "FreeFieldComp"


def test_the_moving_signal_actually_moves(lab):
    """It is the one signal whose purpose is to exercise the crossfade."""

    lab.set_params({"hrtfAsset": FIXTURE})
    options = lab.audition_options()
    moving, _ = AuditionWorker(FIXTURE, (0.0, 0.0), "moving_broadband", options, 0.4).render()

    half = moving.shape[1] // 2
    first = np.sqrt(np.mean(np.square(moving[:, :half]), axis=1))
    second = np.sqrt(np.mean(np.square(moving[:, half:]), axis=1))
    first_ild = 20.0 * np.log10(first[0] / max(first[1], 1e-12))
    second_ild = 20.0 * np.log10(second[0] / max(second[1], 1e-12))

    # The balance shifts across the clip because the source swept past.
    assert abs(second_ild - first_ild) > 1.0
    assert np.all(np.isfinite(moving))


def test_auditioning_without_an_asset_reports_rather_than_raising(lab):
    lab.start_audition()
    assert "Select a SOFA asset" in lab.status_label.text()


def test_the_hd650_correction_is_not_applied_unless_it_is_chosen(lab):
    """It is measured on HD650s and is wrong for anything else."""

    lab.set_params({"hrtfAsset": FIXTURE})
    assert lab.headphone_combo.currentData() == "off"

    worker = AuditionWorker(FIXTURE, (30.0, 0.0), "click", lab.audition_options(), 0.2)
    assert worker._headphone_correction(44_100) is None

    # Choosing the mode but supplying no file must also stay inert rather than
    # silently reaching for some other measurement.
    options = dict(lab.audition_options())
    options["headphoneCorrection"] = "sonicom_hd650"
    armed = AuditionWorker(FIXTURE, (30.0, 0.0), "click", options, 0.2)
    assert armed._headphone_correction(44_100) is None


# --- ratings and ranking ----------------------------------------------------


def test_ratings_are_saved_locally_and_ranked(lab, isolated_storage):
    lab.set_params({"hrtfAsset": FIXTURE})
    lab.profile_edit.setText("carrier")
    for slider in lab._criteria_sliders.values():
        slider.setValue(5)
    lab.save_rating()

    assert storage.ratings_path().exists()
    assert isolated_storage in storage.ratings_path().parents

    lab.refresh_profiles()
    lab.ranking_profile_combo.setCurrentText("carrier")
    lab.refresh_ranking()
    assert lab.ranking_table.rowCount() == 1
    assert lab.ranking_table.item(0, 1).text() == "5.00"


def test_a_voice_can_prefer_a_different_subject_from_another(lab):
    """Separate profiles, because a carrier and a noise bed are not the same job."""

    def rate(subject_path, profile, value):
        lab._params["hrtfAsset"] = subject_path
        lab.profile_edit.setText(profile)
        for slider in lab._criteria_sliders.values():
            slider.setValue(value)
        lab.save_rating()

    rate("P0001_FreeFieldComp_48kHz.sofa", "carrier", 5)
    rate("P0002_FreeFieldComp_48kHz.sofa", "carrier", 2)
    rate("P0001_FreeFieldComp_48kHz.sofa", "noise", 1)
    rate("P0002_FreeFieldComp_48kHz.sofa", "noise", 5)

    lab.refresh_profiles()

    def top_subject(profile):
        lab.ranking_profile_combo.setCurrentText(profile)
        lab.refresh_ranking()
        return lab.ranking_table.item(0, 0).text()

    assert top_subject("carrier") == "P0001"
    assert top_subject("noise") == "P0002"


def test_saving_a_rating_without_an_asset_reports_rather_than_raising(lab):
    lab.save_rating()
    assert "Select a SOFA asset" in lab.status_label.text()


# --- coverage plot ----------------------------------------------------------


def test_the_coverage_plot_reports_what_it_was_given(qtbot):
    plot = CoveragePlotWidget()
    qtbot.addWidget(plot)
    assert plot.measurement_count == 0

    plot.set_directions([0.0, 90.0, 270.0], [0.0, 30.0, -30.0])
    assert plot.measurement_count == 3

    plot.set_highlight(45.0, 15.0)
    plot.set_highlight(None)
    plot.clear()
    assert plot.measurement_count == 0


def test_the_coverage_plot_rejects_mismatched_inputs(qtbot):
    plot = CoveragePlotWidget()
    qtbot.addWidget(plot)
    with pytest.raises(ValueError):
        plot.set_directions([0.0, 90.0], [0.0])
