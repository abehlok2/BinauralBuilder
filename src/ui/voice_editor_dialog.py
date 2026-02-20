from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QLabel,
    QComboBox,
    QCheckBox,
    QSplitter,
    QGroupBox,
    QScrollArea,
    QGridLayout,
    QPushButton,
    QTextEdit,
    QMessageBox,
    QSpacerItem,
    QSizePolicy,
    QLineEdit,
    QInputDialog,
    QFileDialog,
    QSpinBox,
    QDoubleSpinBox,
)
from PyQt5.QtCore import Qt, pyqtSlot, QTimer
from PyQt5.QtGui import QIntValidator, QDoubleValidator, QFont
import ast
import copy
from collections import OrderedDict
import json
import math
import inspect
import traceback
from src.synth_functions import sound_creator  # Updated import path
from src.utils.voice_file import (
    VoicePreset,
    load_voice_preset,
    save_voice_preset,
    VOICE_FILE_EXTENSION,
)
from src.utils.amp_utils import amplitude_to_db, db_to_amplitude, is_amp_key
from .spatial_trajectory_dialog import SpatialTrajectoryDialog

# Constants from your original dialog structure for envelopes
ENVELOPE_TYPE_NONE = "None"
ENVELOPE_TYPE_LINEAR = "linear_fade" # Corresponds to create_linear_fade_envelope in sound_creator
SUPPORTED_ENVELOPE_TYPES = [ENVELOPE_TYPE_NONE, ENVELOPE_TYPE_LINEAR]


def get_default_param_display_flag(param_name: str) -> bool:
    """Return whether a parameter should be displayed by default in details views."""

    name_lower = param_name.lower()
    return not (
        name_lower.startswith("flange")
        or name_lower.startswith("spatial")
        or param_name in FLANGE_TOOLTIPS
        or param_name in SPATIAL_TOOLTIPS
    )

# Synth functions that should be available for parameter lookup but hidden
# from the UI drop-down list. These are typically transition variants that
# are selected automatically when the "Is Transition?" box is checked.
UI_EXCLUDED_FUNCTION_NAMES = [
    'rhythmic_waveshaping_transition',
    'stereo_am_independent_transition',
    'wave_shape_stereo_am_transition',
    'binaural_beat_transition',
    'isochronic_tone_transition',
    'monaural_beat_stereo_amps_transition',
    'qam_beat_transition',
    'hybrid_qam_monaural_beat_transition',
    'spatial_angle_modulation_transition',
    'spatial_angle_modulation_sam2_transition',
    'spatial_angle_modulation_monaural_beat_transition',
    'dual_pulse_binaural_transition',
]

# Human readable descriptions for synth parameters. These are used to provide
# hover tooltips in the editor so users can quickly understand what each field
# controls.  Only the `binaural_beat` voice is documented exhaustively for now;
# other voices will fall back to generic hints.
PARAM_TOOLTIPS = {
    'binaural_beat': {
        'ampL': 'Amplitude of left channel (0–1).',
        'ampR': 'Amplitude of right channel (0–1).',
        'baseFreq': 'Central frequency in Hz around which the binaural beat is generated.',
        'beatFreq': 'Frequency difference between left and right channels in Hz.',
        'leftHigh': 'If checked, left channel is higher in frequency than right.',
        'startPhaseL': 'Initial phase offset of left channel in radians.',
        'startPhaseR': 'Initial phase offset of right channel in radians.',
        'ampOscDepthL': 'Depth of amplitude modulation applied to the left channel.',
        'ampOscFreqL': 'Frequency of amplitude modulation for the left channel in Hz.',
        'ampOscPhaseOffsetL': 'Phase offset for left channel amplitude modulation.',
        'ampOscDepthR': 'Depth of amplitude modulation applied to the right channel.',
        'ampOscFreqR': 'Frequency of amplitude modulation for the right channel in Hz.',
        'ampOscPhaseOffsetR': 'Phase offset for right channel amplitude modulation.',
        'freqOscRangeL': 'Range of frequency modulation for the left channel in Hz.',
        'freqOscFreqL': 'Frequency of frequency modulation for the left channel in Hz.',
        'freqOscSkewL': 'Skew factor for left channel frequency modulation waveform.',
        'freqOscPhaseOffsetL': 'Phase offset for left channel frequency modulation.',
        'freqOscRangeR': 'Range of frequency modulation for the right channel in Hz.',
        'freqOscFreqR': 'Frequency of frequency modulation for the right channel in Hz.',
        'freqOscSkewR': 'Skew factor for right channel frequency modulation waveform.',
        'freqOscPhaseOffsetR': 'Phase offset for right channel frequency modulation.',
        'freqOscShape': 'Waveform shape for frequency modulation ("sine" or "triangle").',
        'phaseOscFreq': 'Frequency of phase modulation applied to both channels.',
        'phaseOscRange': 'Range of phase modulation in radians.',
    },
    'spatial_angle_modulation_sam2': {
        'amp': 'Overall signal amplitude.',
        'carrierFreq': 'Carrier tone frequency (f_s) in Hz.',
        'modFreq': 'Spatial modulation frequency (f_m) in Hz.',
        'peakPhaseDev': 'Peak phase deviation (φ_p) in radians; controls spatial width.',
        'phaseOffsetL': 'Absolute left-channel phase offset (φ_L) in radians.',
        'phaseOffsetR': 'Absolute right-channel phase offset (φ_R) in radians.',
        'pathType': 'Spatial path mode: open (sine), closed (ramp), discontinuous (step).',
        'discontinuousSteps': 'Number of discrete positions used by discontinuous path type.',
    }
}


# Tooltips for flanger parameters
FLANGE_TOOLTIPS = {
    'flangeEnable': 'Enable or disable the flanger effect.',
    'flangeDelayMs': 'Base delay time in milliseconds.',
    'flangeDepthMs': 'Depth of delay modulation in milliseconds.',
    'flangeRateHz': 'LFO rate controlling the delay modulation (Hz).',
    'flangeShape': 'LFO waveform shape (sine or triangle).',
    'flangeFeedback': 'Amount of output fed back into the input (-1 to +1).',
    'flangeMix': 'Wet/dry mix of the flanged signal (0-1).',
    'flangeLoopLpfHz': 'Low-pass filter cutoff in the feedback loop (Hz).',
    'flangeLoopHpfHz': 'High-pass filter cutoff in the feedback loop (Hz).',
    'flangeStereoMode': 'Stereo mode: 0 linked, 1 spread, 2 mid-only, 3 side-only.',
    'flangeSpreadDeg': 'LFO phase offset between channels in spread mode (degrees).',
    'flangeDelayLaw': 'Sweep law for delay modulation: 0 \u03c4-linear, 1 1/\u03c4-linear, 2 exp-\u03c4.',
    'flangeInterp': 'Delay-line interpolation: 0 linear, 1 Lagrange3.',
    'flangeMinDelayMs': 'Minimum allowed delay time (ms).',
    'flangeMaxDelayMs': 'Maximum allowed delay time (ms).',
    'flangeDezipperDelayMs': 'Dezipper time constant for delay parameter (ms).',
    'flangeDezipperDepthMs': 'Dezipper time constant for depth parameter (ms).',
    'flangeDezipperRateMs': 'Dezipper time constant for LFO rate (ms).',
    'flangeDezipperFeedbackMs': 'Dezipper time constant for feedback (ms).',
    'flangeDezipperWetMs': 'Dezipper time constant for wet mix (ms).',
    'flangeDezipperFilterMs': 'Dezipper time constant for loop filters (ms).',
    'flangeLoudnessMode': 'Loudness compensation: 0 off, 1 match input RMS.',
    'flangeLoudnessTcMs': 'RMS detector time constant (ms).',
    'flangeLoudnessMinGain': 'Minimum makeup gain when loudness mode is on.',
    'flangeLoudnessMaxGain': 'Maximum makeup gain when loudness mode is on.',
}

# Tooltips for 2D ambisonic spatialization parameters
SPATIAL_TOOLTIPS = {
    'spatialEnable': 'Enable or disable 2D ambisonic spatialization.',
    'spatialUseItdIld': 'Apply interaural time and level differences (0 off, 1 on).',
    'spatialEarAngleDeg': 'Virtual ear angle for stereo decode (degrees).',
    'spatialHeadRadiusM': 'Assumed listener head radius in meters.',
    'spatialItdScale': 'Scale factor for interaural time differences.',
    'spatialIldMaxDb': 'Maximum interaural level difference in dB.',
    'spatialIldXoverHz': 'Frequency where ILD shelf begins (Hz).',
    'spatialDecoder': 'Spatial decoder: "itd_head" (time-delay) or "foa_cardioid" (legacy).',
    'spatialRefDistanceM': 'Reference distance for distance attenuation (meters).',
    'spatialRolloff': 'Distance rolloff exponent.',
    'spatialHfRollDbPerM': 'High-frequency rolloff per meter (dB).',
    'spatialMinDistanceM': 'Minimum distance clamp to avoid singularities (meters).',
    'spatialMaxDegPerS': 'Maximum allowed azimuth change per second (deg/s).',
    'spatialMaxDelayStepSamples': 'Maximum ITD delay change per sample to maintain stability (samples).',
    'spatialDezipperThetaMs': 'Dezipper time constant for azimuth changes (ms).',
    'spatialDezipperDistMs': 'Dezipper time constant for distance changes (ms).',
    'spatialTrajectory': 'Array of movement segments (rotate or oscillate) defining azimuth and distance over time.',
}


# Helper utilities specific to sweep / static notch editing
def _coerce_to_sequence(value):
    """Convert stored parameter ``value`` into a list for editing widgets."""
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return []
        else:
            value = parsed
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return value
    return []


class NoiseSweepRow(QWidget):
    """Row widget for configuring a single swept notch specification."""

    def __init__(self, is_transition=False, parent=None):
        super().__init__(parent)
        self.is_transition = is_transition
        self._build_ui()

    def _build_ui(self):
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(4)

        self.start_min_label = QLabel(self)
        self.start_min_spin = QSpinBox(self)
        self.start_min_spin.setRange(20, 20000)
        self.start_min_spin.setSingleStep(10)

        self.start_max_label = QLabel(self)
        self.start_max_spin = QSpinBox(self)
        self.start_max_spin.setRange(20, 22050)
        self.start_max_spin.setSingleStep(10)

        self.start_q_label = QLabel(self)
        self.start_q_spin = QDoubleSpinBox(self)
        self.start_q_spin.setDecimals(2)
        self.start_q_spin.setRange(0.1, 1000.0)
        self.start_q_spin.setSingleStep(0.5)

        self.start_casc_label = QLabel(self)
        self.start_casc_spin = QSpinBox(self)
        self.start_casc_spin.setRange(1, 20)

        self.end_min_label = QLabel(self)
        self.end_min_spin = QSpinBox(self)
        self.end_min_spin.setRange(20, 20000)
        self.end_min_spin.setSingleStep(10)

        self.end_max_label = QLabel(self)
        self.end_max_spin = QSpinBox(self)
        self.end_max_spin.setRange(20, 22050)
        self.end_max_spin.setSingleStep(10)

        self.end_q_label = QLabel(self)
        self.end_q_spin = QDoubleSpinBox(self)
        self.end_q_spin.setDecimals(2)
        self.end_q_spin.setRange(0.1, 1000.0)
        self.end_q_spin.setSingleStep(0.5)

        self.end_casc_label = QLabel(self)
        self.end_casc_spin = QSpinBox(self)
        self.end_casc_spin.setRange(1, 20)

        self.remove_button = QPushButton("Remove", self)

        # Row 0 -> start values + remove
        layout.addWidget(self.start_min_label, 0, 0)
        layout.addWidget(self.start_min_spin, 0, 1)
        layout.addWidget(self.start_max_label, 0, 2)
        layout.addWidget(self.start_max_spin, 0, 3)
        layout.addWidget(self.start_q_label, 0, 4)
        layout.addWidget(self.start_q_spin, 0, 5)
        layout.addWidget(self.start_casc_label, 0, 6)
        layout.addWidget(self.start_casc_spin, 0, 7)
        layout.addWidget(self.remove_button, 0, 8)

        # Row 1 -> end values
        layout.addWidget(self.end_min_label, 1, 0)
        layout.addWidget(self.end_min_spin, 1, 1)
        layout.addWidget(self.end_max_label, 1, 2)
        layout.addWidget(self.end_max_spin, 1, 3)
        layout.addWidget(self.end_q_label, 1, 4)
        layout.addWidget(self.end_q_spin, 1, 5)
        layout.addWidget(self.end_casc_label, 1, 6)
        layout.addWidget(self.end_casc_spin, 1, 7)

        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(3, 1)
        layout.setColumnStretch(5, 1)
        layout.setColumnStretch(7, 1)
        layout.setColumnStretch(8, 0)

        self._end_widgets = [
            self.end_min_label,
            self.end_min_spin,
            self.end_max_label,
            self.end_max_spin,
            self.end_q_label,
            self.end_q_spin,
            self.end_casc_label,
            self.end_casc_spin,
        ]

        self.set_transition_mode(self.is_transition)
        self.set_values({})

    def set_transition_mode(self, is_transition: bool):
        self.is_transition = bool(is_transition)
        if self.is_transition:
            self.start_min_label.setText("Start Min Hz")
            self.start_max_label.setText("Start Max Hz")
            self.start_q_label.setText("Start Q")
            self.start_casc_label.setText("Start Casc")
            self.end_min_label.setText("End Min Hz")
            self.end_max_label.setText("End Max Hz")
            self.end_q_label.setText("End Q")
            self.end_casc_label.setText("End Casc")
            for widget in self._end_widgets:
                widget.show()
        else:
            self.start_min_label.setText("Min Hz")
            self.start_max_label.setText("Max Hz")
            self.start_q_label.setText("Q")
            self.start_casc_label.setText("Casc")
            for widget in self._end_widgets:
                widget.hide()

    def set_values(self, data):
        if isinstance(data, dict):
            start_min = data.get("start_min", data.get("min", 1000))
            start_max = data.get("start_max", data.get("max", 10000))
            end_min = data.get("end_min", data.get("max", start_min))
            end_max = data.get("end_max", data.get("max", start_max))
            start_q = data.get("start_q", data.get("q", 25.0))
            end_q = data.get("end_q", data.get("q", start_q))
            start_casc = data.get("start_casc", data.get("cascade", 10))
            end_casc = data.get("end_casc", data.get("cascade", start_casc))
        elif isinstance(data, (list, tuple)):
            items = list(data)
            start_min = items[0] if len(items) > 0 else 1000
            start_max = items[1] if len(items) > 1 else 10000
            end_min = items[2] if len(items) > 2 else start_min
            end_max = items[3] if len(items) > 3 else start_max
            start_q = items[4] if len(items) > 4 else 25.0
            end_q = items[5] if len(items) > 5 else start_q
            start_casc = items[6] if len(items) > 6 else 10
            end_casc = items[7] if len(items) > 7 else start_casc
        else:
            start_min = 1000
            start_max = 10000
            end_min = start_min
            end_max = start_max
            start_q = 25.0
            end_q = start_q
            start_casc = 10
            end_casc = start_casc

        self.start_min_spin.setValue(int(round(start_min)))
        self.start_max_spin.setValue(int(round(start_max)))
        self.end_min_spin.setValue(int(round(end_min)))
        self.end_max_spin.setValue(int(round(end_max)))
        self.start_q_spin.setValue(float(start_q))
        self.end_q_spin.setValue(float(end_q))
        self.start_casc_spin.setValue(int(round(start_casc)))
        self.end_casc_spin.setValue(int(round(end_casc)))

    def get_value(self):
        start_data = {
            "start_min": int(self.start_min_spin.value()),
            "start_max": int(self.start_max_spin.value()),
            "start_q": float(self.start_q_spin.value()),
            "start_casc": int(self.start_casc_spin.value()),
        }
        end_data = {
            "end_min": int(self.end_min_spin.value()),
            "end_max": int(self.end_max_spin.value()),
            "end_q": float(self.end_q_spin.value()),
            "end_casc": int(self.end_casc_spin.value()),
        }
        if self.is_transition:
            merged = {**start_data, **end_data}
            return merged
        return {
            "min": start_data["start_min"],
            "max": start_data["start_max"],
            "q": start_data["start_q"],
            "cascade": start_data["start_casc"],
        }


class NoiseSweepEditor(QWidget):
    """Widget managing a collection of swept notch specifications."""

    def __init__(self, parent=None, is_transition=False, max_rows=4):
        super().__init__(parent)
        self.is_transition = bool(is_transition)
        self.max_rows = max_rows
        self.rows = []
        self._row_wrappers = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.rows_container = QVBoxLayout()
        self.rows_container.setContentsMargins(0, 0, 0, 0)
        self.rows_container.setSpacing(4)
        layout.addLayout(self.rows_container)

        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        self.add_button = QPushButton("Add Sweep", self)
        self.add_button.clicked.connect(self.add_row)
        button_layout.addWidget(self.add_button)
        layout.addLayout(button_layout)

    def _default_entry(self):
        if self.is_transition:
            return {
                "start_min": 1000,
                "start_max": 10000,
                "end_min": 1000,
                "end_max": 10000,
                "start_q": 25.0,
                "end_q": 25.0,
                "start_casc": 10,
                "end_casc": 10,
            }
        return {"min": 1000, "max": 10000, "q": 25.0, "cascade": 10}

    def set_values(self, values):
        entries = _coerce_to_sequence(values)
        if not entries:
            entries = [self._default_entry()]
        self.clear_rows()
        for entry in entries[: self.max_rows]:
            self.add_row(entry)
        self._update_button_states()

    def add_row(self, entry=None):
        if len(self.rows) >= self.max_rows:
            return
        row = NoiseSweepRow(self.is_transition, self)
        row.set_transition_mode(self.is_transition)
        if entry is None:
            row.set_values(self._default_entry())
        else:
            row.set_values(entry)
        row.remove_button.clicked.connect(lambda _, r=row: self.remove_row(r))

        wrapper = QGroupBox(self)
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(6, 6, 6, 6)
        wrapper_layout.addWidget(row)

        self.rows.append(row)
        self._row_wrappers.append(wrapper)
        self.rows_container.addWidget(wrapper)
        self._renumber_rows()
        self._update_button_states()

    def remove_row(self, row_widget):
        if row_widget not in self.rows:
            return
        if len(self.rows) <= 1:
            return  # maintain at least one sweep
        idx = self.rows.index(row_widget)
        wrapper = self._row_wrappers.pop(idx)
        self.rows.pop(idx)
        row_widget.setParent(None)
        row_widget.deleteLater()
        wrapper.setParent(None)
        wrapper.deleteLater()
        self._renumber_rows()
        self._update_button_states()

    def clear_rows(self):
        for row in self.rows:
            row.setParent(None)
            row.deleteLater()
        for wrapper in self._row_wrappers:
            wrapper.setParent(None)
            wrapper.deleteLater()
        self.rows = []
        self._row_wrappers = []

    def set_transition_mode(self, is_transition: bool):
        self.is_transition = bool(is_transition)
        for row in self.rows:
            row.set_transition_mode(self.is_transition)
        self._update_button_states()

    def get_values(self):
        return [row.get_value() for row in self.rows]

    def _update_button_states(self):
        self.add_button.setEnabled(len(self.rows) < self.max_rows)
        self._renumber_rows()

    def _renumber_rows(self):
        for idx, wrapper in enumerate(self._row_wrappers):
            wrapper.setTitle(f"Sweep {idx + 1}")


class StaticNotchRow(QWidget):
    """Row widget describing a static notch filter."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(4)

        self.freq_label = QLabel("Frequency (Hz)", self)
        self.freq_spin = QDoubleSpinBox(self)
        self.freq_spin.setRange(20.0, 20000.0)
        self.freq_spin.setDecimals(2)
        self.freq_spin.setSingleStep(10.0)

        self.q_label = QLabel("Q", self)
        self.q_spin = QDoubleSpinBox(self)
        self.q_spin.setRange(0.1, 1000.0)
        self.q_spin.setDecimals(2)
        self.q_spin.setSingleStep(0.5)

        self.casc_label = QLabel("Casc", self)
        self.casc_spin = QSpinBox(self)
        self.casc_spin.setRange(1, 20)

        self.remove_button = QPushButton("Remove", self)

        layout.addWidget(self.freq_label, 0, 0)
        layout.addWidget(self.freq_spin, 0, 1)
        layout.addWidget(self.q_label, 0, 2)
        layout.addWidget(self.q_spin, 0, 3)
        layout.addWidget(self.casc_label, 0, 4)
        layout.addWidget(self.casc_spin, 0, 5)
        layout.addWidget(self.remove_button, 0, 6)

        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(3, 1)
        layout.setColumnStretch(5, 0)

        self.set_values({})

    def set_values(self, data):
        if isinstance(data, dict):
            freq = data.get("freq", data.get("frequency", data.get("hz", 1000.0)))
            q_val = data.get("q", data.get("quality", 25.0))
            cascade = data.get("cascade", data.get("count", 1))
        elif isinstance(data, (list, tuple)):
            items = list(data)
            freq = items[0] if len(items) > 0 else 1000.0
            q_val = items[1] if len(items) > 1 else 25.0
            cascade = items[2] if len(items) > 2 else 1
        else:
            freq = 1000.0
            q_val = 25.0
            cascade = 1

        self.freq_spin.setValue(float(freq))
        self.q_spin.setValue(float(q_val))
        self.casc_spin.setValue(int(round(cascade)))

    def get_value(self):
        return {
            "freq": float(self.freq_spin.value()),
            "q": float(self.q_spin.value()),
            "cascade": int(self.casc_spin.value()),
        }


class StaticNotchEditor(QWidget):
    """Widget managing a collection of static notch filters."""

    def __init__(self, parent=None, max_rows=10):
        super().__init__(parent)
        self.max_rows = max_rows
        self.rows = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.empty_label = QLabel("No static notches configured.", self)
        self.empty_label.setStyleSheet("color: gray;")
        layout.addWidget(self.empty_label)

        self.rows_container = QVBoxLayout()
        self.rows_container.setContentsMargins(0, 0, 0, 0)
        self.rows_container.setSpacing(4)
        layout.addLayout(self.rows_container)

        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        self.add_button = QPushButton("Add Static Notch", self)
        self.add_button.clicked.connect(self.add_row)
        button_layout.addWidget(self.add_button)
        layout.addLayout(button_layout)

    def _default_entry(self):
        return {"freq": 1000.0, "q": 25.0, "cascade": 1}

    def set_values(self, values):
        entries = _coerce_to_sequence(values)
        if not entries:
            entries = []
        self.clear_rows()
        for entry in entries[: self.max_rows]:
            self.add_row(entry)
        self._update_state()

    def add_row(self, entry=None):
        if len(self.rows) >= self.max_rows:
            return
        row = StaticNotchRow(self)
        if entry is None:
            row.set_values(self._default_entry())
        else:
            row.set_values(entry)
        row.remove_button.clicked.connect(lambda: self.remove_row(row))
        self.rows.append(row)
        self.rows_container.addWidget(row)
        self._update_state()

    def remove_row(self, row_widget):
        if row_widget not in self.rows:
            return
        self.rows.remove(row_widget)
        row_widget.setParent(None)
        row_widget.deleteLater()
        self._update_state()

    def clear_rows(self):
        for row in self.rows:
            row.setParent(None)
            row.deleteLater()
        self.rows = []

    def get_values(self):
        return [row.get_value() for row in self.rows]

    def _update_state(self):
        has_rows = bool(self.rows)
        self.empty_label.setVisible(not has_rows)
        self.add_button.setEnabled(len(self.rows) < self.max_rows)

class VoiceEditorDialog(QDialog): # Standard class name

    DEFAULT_WIDTH = 900
    DEFAULT_HEIGHT = 700
    ENTRY_WIDTH = 120

    # Default column widths used previously to keep parameter entry fields
    # aligned. These are retained for backward compatibility but no longer
    # applied so that labels take only the space they need.
    MAIN_LABEL_WIDTH = 150
    SUB_LABEL_WIDTH = 40
    

    def __init__(self, parent, app_ref, step_index, voice_index=None):
        super().__init__(parent)
        self.app = app_ref # Main application reference
        self.step_index = step_index
        self.voice_index = voice_index
        self.is_new_voice = (voice_index is None)
        if parent:
            self.setPalette(parent.palette())
            self.setStyleSheet(parent.styleSheet())

        # Validators
        self.double_validator_non_negative = QDoubleValidator(0.0, 999999.0, 6, self)
        self.double_validator_zero_to_one = QDoubleValidator(0.0, 1.0, 6, self)
        self.double_validator = QDoubleValidator(-999999.0, 999999.0, 6, self)
        self.int_validator = QIntValidator(-999999, 999999, self)

        self.setWindowTitle(f"{'Add' if self.is_new_voice else 'Edit'} Voice for Step {step_index + 1}")
        self.resize(self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT)
        self.setMinimumSize(700, 600)
        self.setModal(True)

        self.param_widgets = {}  # To store {'param_name': {'widget': widget, 'type': type_str}}
        self.envelope_param_widgets = {} # Similar for envelope params

        self._load_initial_data() # Loads or creates self.current_voice_data

        # Maintain separate parameter sets for transition and non-transition modes
        self._standard_params = OrderedDict()
        self._transition_params = OrderedDict()
        initial_params = OrderedDict(self.current_voice_data.get("params", {}))
        func_name = self.current_voice_data.get("synth_function_name", "")
        if self.current_voice_data.get("is_transition", False):
            self._transition_params = copy.deepcopy(initial_params)
            # Derive non-transition params without overwriting transition state
            self._standard_params = self._convert_params_to_standard(initial_params, func_name)
        else:
            self._standard_params = copy.deepcopy(initial_params)
            # Prepare initial transition params so they can be restored if toggled
            self._transition_params = self._convert_params_to_transition(initial_params, func_name)

        self._setup_ui()          # Creates UI elements

        # Initial population after UI setup
        if self.current_voice_data.get("synth_function_name") != "Error": # Check if data load failed
            self.populate_parameters() # Populates synth parameters based on current_voice_data
            self._populate_envelope_controls() # Populates envelope controls
        self._update_swap_button_visibility()
        
        self._populate_reference_step_combo()

        # Set initial reference selection (if possible)
        initial_ref_step = self.app.get_selected_step_index() 
        initial_ref_voice = self.app.get_selected_voice_index()
        if initial_ref_step is not None:
            step_combo_index = self.reference_step_combo.findData(initial_ref_step)
            if step_combo_index != -1:
                self.reference_step_combo.setCurrentIndex(step_combo_index)
                # Defer voice selection slightly to ensure voice combo is populated
                if initial_ref_voice is not None:
                    QTimer.singleShot(50, lambda: self._select_initial_reference_voice(initial_ref_voice))
            elif self.reference_step_combo.count() > 0: # Fallback to first step if specific not found
                self.reference_step_combo.setCurrentIndex(0)
        elif self.reference_step_combo.count() > 0: # Fallback if no initial step selected in main
            self.reference_step_combo.setCurrentIndex(0)


    def _load_initial_data(self):
        if self.is_new_voice:
            prefs_voice = getattr(getattr(self.app, "prefs", None), "default_voice", None)
            if isinstance(prefs_voice, dict) and prefs_voice.get("synth_function_name"):
                try:
                    self.current_voice_data = copy.deepcopy(prefs_voice)
                    self.current_voice_data.setdefault("params", {})
                    self.current_voice_data.setdefault("volume_envelope", None)
                    self.current_voice_data.setdefault("description", "")
                    self.current_voice_data.setdefault("is_transition", False)
                    return
                except Exception as e:
                    print(f"Warning: failed to apply default voice from prefs: {e}")

            available_funcs = sorted(
                name for name in sound_creator.SYNTH_FUNCTIONS.keys()
                if name not in UI_EXCLUDED_FUNCTION_NAMES
            )
            if not available_funcs:
                available_funcs = sorted(sound_creator.SYNTH_FUNCTIONS.keys())
            first_func_name = available_funcs[0] if available_funcs else "default_sine"

            is_trans = first_func_name.endswith("_transition")
            default_params = self._get_default_params(first_func_name, is_trans)

            self.current_voice_data = {
                "synth_function_name": first_func_name,
                "is_transition": is_trans,
                "params": default_params,
                "volume_envelope": None,
                "description": "",
            }
        else:
            try:
                original_voice = self.app.track_data["steps"][self.step_index]["voices"][self.voice_index]
                self.current_voice_data = copy.deepcopy(original_voice)
                
                # Ensure essential keys exist
                if "params" not in self.current_voice_data:
                    self.current_voice_data["params"] = {}
                if "volume_envelope" not in self.current_voice_data: # Default to no envelope
                    self.current_voice_data["volume_envelope"] = None # Or {"type": ENVELOPE_TYPE_NONE, "params": {}}
                if "is_transition" not in self.current_voice_data: # Infer if missing
                    self.current_voice_data["is_transition"] = self.current_voice_data.get("synth_function_name","").endswith("_transition")
                if "description" not in self.current_voice_data:
                    self.current_voice_data["description"] = ""

            except (IndexError, KeyError, AttributeError) as e: # Added AttributeError for self.app.track_data access
                QMessageBox.critical(self.parent(), "Error", f"Could not load voice data for editing:\n{e}")
                # Fallback to a clearly erroneous state or a very basic default
                self.current_voice_data = {
                    "synth_function_name": "Error", 
                    "is_transition": False, 
                    "params": {}, 
                    "volume_envelope": None
                }
                QTimer.singleShot(0, self.reject) # Close dialog if data load fails critically


    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        # Top: Synth Function and Transition Check
        top_frame = QWidget()
        top_layout = QHBoxLayout(top_frame)
        top_layout.setContentsMargins(0,0,0,0)
        top_layout.addWidget(QLabel("Synth Function:"))
        self.synth_func_combo = QComboBox()
        try:
            # Populate from sound_creator.SYNTH_FUNCTIONS but hide functions
            # that are intended for internal use only.
            func_names = sorted(
                name for name in sound_creator.SYNTH_FUNCTIONS.keys()
                if name not in UI_EXCLUDED_FUNCTION_NAMES
            )
            if not func_names:
                raise ValueError("No synth functions found in sound_creator.SYNTH_FUNCTIONS")
            self.synth_func_combo.addItems(func_names)
        except Exception as e:
            print(f"Error populating synth_func_combo: {e}")
            self.synth_func_combo.addItem("Error: Functions unavailable")
            self.synth_func_combo.setEnabled(False)

        current_func_name = self.current_voice_data.get("synth_function_name", "")
        if current_func_name and self.synth_func_combo.findText(current_func_name) != -1:
            self.synth_func_combo.setCurrentText(current_func_name)
        elif self.synth_func_combo.count() > 0:
             self.synth_func_combo.setCurrentIndex(0) # Select first if current not found or invalid

        self.synth_func_combo.currentIndexChanged.connect(self.on_synth_function_change)
        top_layout.addWidget(self.synth_func_combo, 1) # Give combo box more stretch

        self.transition_check = QCheckBox("Is Transition?")
        self.transition_check.setChecked(self.current_voice_data.get("is_transition", False))
        self.transition_check.stateChanged.connect(self.on_transition_toggle)
        top_layout.addWidget(self.transition_check)
        main_layout.addWidget(top_frame)

        # Main Horizontal Splitter
        h_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(h_splitter, 1) # Allow splitter to take up most space

        # Left side: Synth Parameters
        self.params_groupbox = QGroupBox("Synth Parameters") # Removed "(Editing)" for cleaner UI
        params_groupbox_layout = QVBoxLayout(self.params_groupbox)
        self.params_scroll_area = QScrollArea()
        self.params_scroll_area.setWidgetResizable(True)
        self.params_scroll_content = QWidget()
        self.params_scroll_layout = QGridLayout(self.params_scroll_content)
        self.params_scroll_layout.setAlignment(Qt.AlignTop)
        # Columns: 0=Label, 1=StartVal, 2=Unit/Swap, 3=EndLabel, 4=EndVal, 5=EndUnit
        self.params_scroll_layout.setColumnStretch(1, 2) 
        self.params_scroll_layout.setColumnStretch(4, 2)
        self.params_scroll_area.setWidget(self.params_scroll_content)
        params_groupbox_layout.addWidget(self.params_scroll_area)

        self.swap_params_button = QPushButton("Swap Transition Parameters")
        self.swap_params_button.clicked.connect(self.swap_transition_parameters)
        swap_btn_layout = QHBoxLayout()
        swap_btn_layout.addStretch(1)
        swap_btn_layout.addWidget(self.swap_params_button)
        params_groupbox_layout.addLayout(swap_btn_layout)
        h_splitter.addWidget(self.params_groupbox)

        # Right side: Reference Voice Viewer
        reference_groupbox = QGroupBox("Reference Voice Details")
        reference_layout = QVBoxLayout(reference_groupbox)
        
        # Use a grid for the reference selection for better alignment
        ref_select_grid = QGridLayout()
        ref_select_grid.addWidget(QLabel("Step:"), 0, 0)
        self.reference_step_combo = QComboBox()
        self.reference_step_combo.setMinimumWidth(80)
        self.reference_step_combo.currentIndexChanged.connect(self._update_reference_voice_combo)
        ref_select_grid.addWidget(self.reference_step_combo, 0, 1)
        
        ref_select_grid.addWidget(QLabel("Voice:"), 0, 2)
        self.reference_voice_combo = QComboBox()
        self.reference_voice_combo.setMinimumWidth(120)
        self.reference_voice_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.reference_voice_combo.currentIndexChanged.connect(self._update_reference_display)
        ref_select_grid.addWidget(self.reference_voice_combo, 0, 3)
        
        ref_select_grid.setColumnStretch(3, 1) # Voice combo stretches
        reference_layout.addLayout(ref_select_grid)

        self.reference_details_text = QTextEdit()
        self.reference_details_text.setReadOnly(True)
        self.reference_details_text.setFont(QFont("Consolas", 9))
        reference_layout.addWidget(self.reference_details_text, 1)

        self.set_ref_end_button = QPushButton("Set Reference as End State")
        self.set_ref_end_button.setEnabled(self.transition_check.isChecked())
        self.set_ref_end_button.clicked.connect(self._apply_reference_as_end_state)
        reference_layout.addWidget(self.set_ref_end_button)

        h_splitter.addWidget(reference_groupbox)
        h_splitter.setSizes([int(self.DEFAULT_WIDTH * 0.6), int(self.DEFAULT_WIDTH * 0.4)]) # Adjust initial split

        # Bottom: Envelope Settings
        self.env_groupbox = QGroupBox("Volume Envelope")
        env_layout = QVBoxLayout(self.env_groupbox)
        env_type_layout = QHBoxLayout()
        env_type_layout.addWidget(QLabel("Type:"))
        self.env_type_combo = QComboBox()
        self.env_type_combo.addItems(SUPPORTED_ENVELOPE_TYPES) # Uses dialog's constants
        self.env_type_combo.currentIndexChanged.connect(self._on_envelope_type_change)
        env_type_layout.addWidget(self.env_type_combo)
        env_type_layout.addStretch(1)
        env_layout.addLayout(env_type_layout)
        self.env_params_widget = QWidget() # This widget will hold the dynamic envelope parameters
        self.env_params_layout = QGridLayout(self.env_params_widget)
        self.env_params_layout.setContentsMargins(10, 5, 5, 5)
        self.env_params_layout.setAlignment(Qt.AlignTop)
        env_layout.addWidget(self.env_params_widget)
        env_layout.addStretch(1) # Push envelope params to the top
        main_layout.addWidget(self.env_groupbox)

        # Dialog Buttons (Save, Cancel, Presets)
        button_frame = QWidget()
        button_layout = QHBoxLayout(button_frame)
        button_layout.addStretch(1) # Push buttons to the right
        self.cancel_button = QPushButton("Cancel")
        self.save_button = QPushButton("Save Voice")
        self.save_button.setStyleSheet("QPushButton { background-color: #0078D7; color: white; padding: 6px; font-weight: bold; border-radius: 3px; } QPushButton:hover { background-color: #005A9E; } QPushButton:pressed { background-color: #003C6A; }")
        self.load_preset_button = QPushButton("Load Preset")
        self.save_preset_button = QPushButton("Save Preset")
        self.cancel_button.clicked.connect(self.reject) # QDialog's built-in reject
        self.save_button.clicked.connect(self.save_voice)
        self.load_preset_button.clicked.connect(self.load_preset)
        self.save_preset_button.clicked.connect(self.save_preset)
        self.save_button.setDefault(True)
        button_layout.addWidget(self.load_preset_button)
        button_layout.addWidget(self.save_preset_button)
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.save_button)
        main_layout.addWidget(button_frame)

    def _clear_layout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                if item is None: continue
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater() # Recommended way
                else: # If item is a layout
                    sub_layout = item.layout()
                    if sub_layout is not None:
                        self._clear_layout(sub_layout) # Recurse

    def populate_parameters(self):
        self._clear_layout(self.params_scroll_layout)
        self.param_widgets = {}
        self._hidden_params = {}

        func_name = self.synth_func_combo.currentText()
        is_transition = self.transition_check.isChecked()

        default_params_ordered = self._get_default_params(func_name, is_transition)

        # Respect user preferences that hide specific parameters from display
        prefs_obj = getattr(self.app, "prefs", None)
        func_display_prefs = {}
        if isinstance(getattr(prefs_obj, "voice_detail_display", None), dict):
            func_display_prefs = prefs_obj.voice_detail_display.get(func_name, {}) or {}

        def _is_hidden_param(param_name: str) -> bool:
            cfg = func_display_prefs.get(param_name)
            return isinstance(cfg, dict) and cfg.get("display") is False

        current_saved_params = self.current_voice_data.get("params", {})
        if func_display_prefs:
            # Preserve hidden parameters so they aren't lost on save
            self._hidden_params = {
                name: current_saved_params.get(name, default_val)
                for name, default_val in default_params_ordered.items()
                if _is_hidden_param(name)
            }
            default_params_ordered = OrderedDict(
                (name, val)
                for name, val in default_params_ordered.items()
                if not _is_hidden_param(name)
            )

        if not default_params_ordered and func_name != "Error":
            self.params_scroll_layout.addWidget(QLabel(f"Warning: Could not determine parameters for '{func_name}'.\nEnsure it's defined in _get_default_params.", self), 0, 0, 1, 6)
            return

        params_to_display = OrderedDict()
        for name, default_value in default_params_ordered.items():
            params_to_display[name] = current_saved_params.get(name, default_value)

        processed_end_params = set()
        transition_pairs = {}
        processed_names = set()

        if is_transition:
            for name in default_params_ordered.keys():
                if name.startswith('start'):
                    base_name = name[len('start'):]
                    end_name = 'end' + base_name
                    if end_name in default_params_ordered:
                        transition_pairs[base_name] = {'start': name, 'end': end_name}
                        processed_end_params.add(end_name)
                elif name.startswith('end') and name not in processed_end_params:
                    base_name = name[len('end'):]
                    if ('start' + base_name) in default_params_ordered and base_name not in transition_pairs:
                        pass

        current_row = 0

        for name in default_params_ordered.keys():
            if name in processed_names or name in processed_end_params:
                continue

            default_value = default_params_ordered[name]
            current_value = params_to_display.get(name, default_value)

            # --- Special Editors (Full Width) ---

            if name == 'spatialTrajectory':
                param_label = QLabel('spatialTrajectory:')
                self.params_scroll_layout.addWidget(param_label, current_row, 0)
                
                summary_label = QLabel(
                    f"{len(current_value) if isinstance(current_value, list) else 0} segments"
                )
                edit_btn = QPushButton('Edit...')
                
                container = QWidget()
                cont_layout = QHBoxLayout(container)
                cont_layout.setContentsMargins(0,0,0,0)
                cont_layout.addWidget(summary_label)
                cont_layout.addWidget(edit_btn)
                cont_layout.addStretch(1)
                
                self.params_scroll_layout.addWidget(container, current_row, 1, 1, 5) # Span rest of columns

                value_list = current_value if isinstance(current_value, list) else []
                self.param_widgets[name] = {'widget': summary_label, 'type': 'json', 'value': value_list}
                
                def _edit_traj():
                    dlg = SpatialTrajectoryDialog(self, self.param_widgets[name]['value'])
                    if dlg.exec_() == QDialog.Accepted:
                        self.param_widgets[name]['value'] = dlg.get_segments()
                        summary_label.setText(
                            f"{len(self.param_widgets[name]['value'])} segments"
                        )
                edit_btn.clicked.connect(_edit_traj)
                
                processed_names.add(name)
                current_row += 1
                continue

            if name == 'sweeps':
                label = QLabel('sweeps:')
                label.setStyleSheet('font-weight: bold;')
                self.params_scroll_layout.addWidget(label, current_row, 0)
                
                editor = NoiseSweepEditor(self, is_transition=is_transition, max_rows=4)
                editor.set_transition_mode(is_transition)
                editor.set_values(current_value)
                
                self.params_scroll_layout.addWidget(editor, current_row, 1, 1, 5) # Span rest
                
                self.param_widgets[name] = {'widget': editor, 'type': 'sweep_list'}
                tooltip = self._get_param_tooltip(func_name, name)
                if tooltip:
                    label.setToolTip(tooltip)
                    editor.setToolTip(tooltip)
                
                processed_names.add(name)
                current_row += 1
                continue

            if name == 'static_notches':
                label = QLabel('static_notches:')
                label.setStyleSheet('font-weight: bold;')
                self.params_scroll_layout.addWidget(label, current_row, 0)
                
                editor = StaticNotchEditor(self)
                editor.set_values(current_value)
                
                self.params_scroll_layout.addWidget(editor, current_row, 1, 1, 5) # Span rest
                
                self.param_widgets[name] = {'widget': editor, 'type': 'static_notch_list'}
                tooltip = self._get_param_tooltip(func_name, name)
                if tooltip:
                    label.setToolTip(tooltip)
                    editor.setToolTip(tooltip)
                
                processed_names.add(name)
                current_row += 1
                continue

            # --- Standard Parameters ---

            display_current = current_value
            if (
                isinstance(current_value, (int, float))
                and getattr(self.app, "prefs", None)
                and getattr(self.app.prefs, "amplitude_display_mode", "absolute") == "dB"
                and is_amp_key(name)
            ):
                display_current = amplitude_to_db(float(current_value))

            prefix, base_after = self._split_name_prefix(name)

            # 1. Transition Pair (Start/End)
            if is_transition and prefix == 'start' and base_after in transition_pairs:
                base_name_for_pair = base_after
                end_name = transition_pairs[base_name_for_pair]['end']
                if end_name in default_params_ordered:
                    end_default_value = default_params_ordered.get(end_name, default_value)
                    end_current_value = params_to_display.get(end_name, end_default_value)
                    display_start = display_current
                    display_end = end_current_value
                    if (
                        isinstance(end_current_value, (int, float))
                        and getattr(self.app, "prefs", None)
                        and getattr(self.app.prefs, "amplitude_display_mode", "absolute") == "dB"
                        and is_amp_key(end_name)
                    ):
                        display_end = amplitude_to_db(float(end_current_value))

                    param_storage_type = 'str'
                    param_type_hint = 'any'
                    range_hint = self._get_param_range_hint(base_name_for_pair)
                    value_for_hint = default_value if default_value is not None else current_value
                    
                    if isinstance(value_for_hint, bool): param_type_hint = 'bool'
                    elif isinstance(value_for_hint, int): param_type_hint = 'bool' if value_for_hint in (0, 1) else 'int'
                    elif isinstance(value_for_hint, float): param_type_hint = 'float'
                    elif isinstance(value_for_hint, str): param_type_hint = 'bool' if value_for_hint.lower() in ('true', 'false') else 'str'

                    current_validator = None
                    if param_type_hint == 'int':
                        current_validator = QIntValidator(-999999, 999999, self)
                        param_storage_type = 'int'
                    elif param_type_hint == 'float':
                        current_validator = QDoubleValidator(-999999.0, 999999.0, 6, self)
                        current_validator.setNotation(QDoubleValidator.StandardNotation)
                        param_storage_type = 'float'
                    else:
                        param_storage_type = 'str'

                    hint_text = f"({param_storage_type}{', ' + range_hint if range_hint else ''})"

                    # Row Layout:
                    # Col 0: Label
                    base_label = QLabel(f"{base_name_for_pair}:")
                    self.params_scroll_layout.addWidget(base_label, current_row, 0, Qt.AlignLeft)

                    # Col 1: Start Widget
                    start_widget = self._create_param_widget(
                        base_name_for_pair, display_start, param_type_hint, current_validator, self.ENTRY_WIDTH
                    )
                    self.params_scroll_layout.addWidget(start_widget, current_row, 1)
                    self.param_widgets[name] = {'widget': start_widget, 'type': param_storage_type}

                    # Col 2: Start Label ("Start") or Hint
                    # Actually, for transition pairs, we might want "Start" label? 
                    # The original had "Start:" label. But in a grid, maybe we can omit it if column headers existed?
                    # But we don't have headers. Let's put "Start" in tooltip or just rely on position.
                    # Or we can put "Start" in Col 1 placeholder? No.
                    # Let's put a small label in Col 1 if we want, but we are using grid.
                    # Let's use the hint text in Col 2.
                    # Wait, the original had "Start:" label.
                    # Let's add "Start" label to the widget or layout?
                    # Maybe just put the widget. The user knows left is start, right is end.
                    
                    # Col 2: Hint
                    self.params_scroll_layout.addWidget(QLabel(hint_text), current_row, 2)

                    # Col 3: End Label (arrow or "End")
                    self.params_scroll_layout.addWidget(QLabel("-> End:"), current_row, 3, Qt.AlignRight)

                    # Col 4: End Widget
                    end_widget = self._create_param_widget(
                        base_name_for_pair, display_end, param_type_hint, current_validator, self.ENTRY_WIDTH
                    )
                    self.params_scroll_layout.addWidget(end_widget, current_row, 4)
                    self.param_widgets[end_name] = {'widget': end_widget, 'type': param_storage_type}

                    # Col 5: Hint (End) - usually same as start, maybe empty
                    # self.params_scroll_layout.addWidget(QLabel(hint_text), current_row, 5)

                    # Tooltips
                    tooltip_base = self._get_param_tooltip(func_name, base_name_for_pair)
                    if tooltip_base:
                        base_label.setToolTip(tooltip_base)
                        start_widget.setToolTip(tooltip_base)
                        end_widget.setToolTip(tooltip_base)

                    processed_names.add(name)
                    processed_names.add(end_name)
                    current_row += 1
                    continue

            # 2. Left/Right Pair
            lr_info = self._parse_lr_suffix(base_after)
            if lr_info:
                base_lr, left_suffix, right_suffix = lr_info
                left_name = prefix + base_lr + left_suffix
                right_name = prefix + base_lr + right_suffix
                if left_name in default_params_ordered and right_name in default_params_ordered and left_name not in processed_names and right_name not in processed_names:
                    left_def = default_params_ordered[left_name]
                    right_def = default_params_ordered[right_name]
                    left_cur = params_to_display.get(left_name, left_def)
                    right_cur = params_to_display.get(right_name, right_def)
                    disp_left = left_cur
                    disp_right = right_cur
                    
                    if (isinstance(left_cur, (int, float)) and getattr(self.app, "prefs", None) and getattr(self.app.prefs, "amplitude_display_mode", "absolute") == "dB" and is_amp_key(left_name)):
                        disp_left = amplitude_to_db(float(left_cur))
                    if (isinstance(right_cur, (int, float)) and getattr(self.app, "prefs", None) and getattr(self.app.prefs, "amplitude_display_mode", "absolute") == "dB" and is_amp_key(right_name)):
                        disp_right = amplitude_to_db(float(right_cur))

                    param_type_hint = 'any'
                    value_for_hint = left_def if left_def is not None else left_cur
                    if isinstance(value_for_hint, bool): param_type_hint = 'bool'
                    elif isinstance(value_for_hint, int): param_type_hint = 'bool' if value_for_hint in (0, 1) else 'int'
                    elif isinstance(value_for_hint, float): param_type_hint = 'float'
                    elif isinstance(value_for_hint, str): param_type_hint = 'bool' if value_for_hint.lower() in ('true', 'false') else 'str'

                    param_storage_type = param_type_hint if param_type_hint in ['int','float','bool','str'] else 'str'
                    range_hint = self._get_param_range_hint(base_lr)
                    hint_text = f"({param_storage_type}{', ' + range_hint if range_hint else ''})"

                    # Row 1: Left Parameter
                    # Col 0: Label
                    base_label = QLabel(f"{left_name}:")
                    self.params_scroll_layout.addWidget(base_label, current_row, 0, Qt.AlignLeft)

                    # Col 1: Widget
                    current_validator = None
                    if param_type_hint == 'int': current_validator = QIntValidator(-999999, 999999, self)
                    elif param_type_hint == 'float': 
                        current_validator = QDoubleValidator(-999999.0, 999999.0, 6, self)
                        current_validator.setNotation(QDoubleValidator.StandardNotation)

                    left_widget = self._create_param_widget(
                        left_name, disp_left, param_type_hint, current_validator, self.ENTRY_WIDTH
                    )
                    self.params_scroll_layout.addWidget(left_widget, current_row, 1)
                    self.param_widgets[left_name] = {'widget': left_widget, 'type': param_storage_type}

                    # Col 2: Swap Button (only on first row of pair)
                    swap_btn = QPushButton("Swap R/L")
                    swap_btn.clicked.connect(lambda _, ln=left_name, rn=right_name: self._swap_lr([ln, rn]))
                    self.params_scroll_layout.addWidget(swap_btn, current_row, 2)
                    
                    tooltip_left = self._get_param_tooltip(func_name, left_name)
                    if tooltip_left:
                        base_label.setToolTip(tooltip_left)
                        left_widget.setToolTip(tooltip_left)

                    processed_names.add(left_name)
                    current_row += 1

                    # Row 2: Right Parameter
                    # Col 0: Label
                    r_label = QLabel(f"{right_name}:")
                    self.params_scroll_layout.addWidget(r_label, current_row, 0, Qt.AlignLeft)

                    # Col 1: Widget
                    right_widget = self._create_param_widget(
                        right_name, disp_right, param_type_hint, current_validator, self.ENTRY_WIDTH
                    )
                    self.params_scroll_layout.addWidget(right_widget, current_row, 1)
                    self.param_widgets[right_name] = {'widget': right_widget, 'type': param_storage_type}

                    # Col 2: Hint
                    self.params_scroll_layout.addWidget(QLabel(hint_text), current_row, 2)

                    tooltip_right = self._get_param_tooltip(func_name, right_name)
                    if tooltip_right:
                        r_label.setToolTip(tooltip_right)
                        right_widget.setToolTip(tooltip_right)

                    processed_names.add(right_name)
                    current_row += 1
                    continue

            # 3. Single Parameter
            base_name_for_pair = None
            is_pair_start = False
            if is_transition and name.startswith('start'):
                base_name_for_pair_candidate = name[len('start'):]
                if base_name_for_pair_candidate in transition_pairs and transition_pairs[base_name_for_pair_candidate]['start'] == name:
                    base_name_for_pair = base_name_for_pair_candidate
                    is_pair_start = True
            
            param_type_hint = "any"
            range_hint = self._get_param_range_hint(name if not is_pair_start else base_name_for_pair)

            value_for_hint = default_value if default_value is not None else current_value
            if value_for_hint is not None:
                if isinstance(value_for_hint, bool): param_type_hint = 'bool'
                elif isinstance(value_for_hint, int): param_type_hint = 'bool' if value_for_hint in (0, 1) else 'int'
                elif isinstance(value_for_hint, float): param_type_hint = 'float'
                elif isinstance(value_for_hint, str): param_type_hint = 'bool' if value_for_hint.lower() in ('true', 'false') else 'str'

            param_storage_type = param_type_hint if param_type_hint in ['int','float','bool','str'] else 'str'
            hint_text = f"({param_storage_type}{', ' + range_hint if range_hint else ''})"

            # Col 0: Label
            label_text = f"{name}:"
            if name == 'leftHigh':
                label_text = "leftHigh:"
            
            base_label = QLabel(label_text)
            self.params_scroll_layout.addWidget(base_label, current_row, 0, Qt.AlignLeft)

            # Col 1: Widget
            current_validator = None
            if param_type_hint == 'int': current_validator = QIntValidator(-999999, 999999, self)
            elif param_type_hint == 'float': 
                current_validator = QDoubleValidator(-999999.0, 999999.0, 6, self)
                current_validator.setNotation(QDoubleValidator.StandardNotation)

            widget = self._create_param_widget(
                name, display_current, param_type_hint, current_validator, self.ENTRY_WIDTH
            )
            self.params_scroll_layout.addWidget(widget, current_row, 1)
            self.param_widgets[name] = {'widget': widget, 'type': param_storage_type}

            # Col 2: Hint
            self.params_scroll_layout.addWidget(QLabel(hint_text), current_row, 2)

            tooltip = self._get_param_tooltip(func_name, name)
            if tooltip:
                base_label.setToolTip(tooltip)
                widget.setToolTip(tooltip)

            processed_names.add(name)
            current_row += 1
        self.params_scroll_layout.setRowStretch(current_row, 1)

        # Apply tooltips to flanger parameters
        for fname, tip in FLANGE_TOOLTIPS.items():
            variants = [fname,
                        'start' + fname[0].upper() + fname[1:],
                        'end' + fname[0].upper() + fname[1:]]
            for var in variants:
                data = self.param_widgets.get(var)
                if data:
                    data['widget'].setToolTip(tip)

        flange_enable_data = self.param_widgets.get('flangeEnable')
        if flange_enable_data:
            enable_widget = flange_enable_data['widget']

            flange_widgets = []
            for n, data in self.param_widgets.items():
                if 'flange' in n.lower() and n != 'flangeEnable':
                    row_widget = data['widget'].parentWidget()
                    if row_widget and row_widget not in flange_widgets:
                        flange_widgets.append(row_widget)

            def _set_flange_enabled(state):
                visible = bool(state)
                for w in flange_widgets:
                    w.setVisible(visible)
                    w.setEnabled(visible)

            enable_widget.stateChanged.connect(_set_flange_enabled)
            _set_flange_enabled(enable_widget.isChecked())

        # Apply tooltips to 2D ambisonic spatialization parameters
        for fname, tip in SPATIAL_TOOLTIPS.items():
            data = self.param_widgets.get(fname)
            if data:
                data['widget'].setToolTip(tip)

        spatial_enable_data = self.param_widgets.get('spatialEnable')
        if spatial_enable_data:
            enable_widget = spatial_enable_data['widget']

            spatial_widgets = []
            for n, data in self.param_widgets.items():
                if n.lower().startswith('spatial') and n != 'spatialEnable':
                    row_widget = data['widget'].parentWidget()
                    if row_widget and row_widget not in spatial_widgets:
                        spatial_widgets.append(row_widget)

            def _set_spatial_enabled(state):
                visible = bool(state)
                for w in spatial_widgets:
                    w.setVisible(visible)
                    w.setEnabled(visible)

            enable_widget.stateChanged.connect(_set_spatial_enabled)
            _set_spatial_enabled(enable_widget.isChecked())

    def _create_param_widget(self, base_name, value, type_hint, validator, width):
        """Helper to create the appropriate widget for a parameter."""
        if base_name == 'noiseType' and type_hint in ('int', 'bool'):
            widget = QComboBox()
            widget.addItems(['1', '2', '3'])
            val_to_set = str(int(value)) if isinstance(value, (int, float)) and int(value) in [1, 2, 3] else '1'
            widget.setCurrentText(val_to_set)
            widget.setMaximumWidth(100)
        elif base_name.endswith('FlangeShape') and type_hint == 'str':
            widget = QComboBox()
            widget.addItems(['sine', 'triangle'])
            val_to_set = value if value in ['sine', 'triangle'] else 'sine'
            widget.setCurrentText(str(val_to_set))
            widget.setMinimumWidth(width)
        elif base_name.endswith('FlangeStereoMode') and type_hint in ('int', 'bool'):
            widget = QComboBox()
            widget.addItem('Linked', 0); widget.addItem('Spread', 1); widget.addItem('Mid-only', 2); widget.addItem('Side-only', 3)
            idx = widget.findData(int(value)) if isinstance(value, (int, float)) else 0
            widget.setCurrentIndex(idx if idx >= 0 else 0)
            widget.setMinimumWidth(width)
        elif base_name.endswith('FlangeDelayLaw') and type_hint in ('int', 'bool'):
            widget = QComboBox()
            widget.addItem('τ-linear', 0); widget.addItem('1/τ-linear', 1); widget.addItem('exp-τ', 2)
            idx = widget.findData(int(value)) if isinstance(value, (int, float)) else 0
            widget.setCurrentIndex(idx if idx >= 0 else 0)
            widget.setMinimumWidth(width)
        elif base_name.endswith('FlangeInterp') and type_hint in ('int', 'bool'):
            widget = QComboBox()
            widget.addItem('Linear', 0); widget.addItem('Lagrange3', 1)
            idx = widget.findData(int(value)) if isinstance(value, (int, float)) else 0
            widget.setCurrentIndex(idx if idx >= 0 else 0)
            widget.setMinimumWidth(width)
        elif base_name.endswith('FlangeLoudnessMode') and type_hint in ('int', 'bool'):
            widget = QComboBox()
            widget.addItem('Off', 0); widget.addItem('Match Input RMS', 1)
            idx = widget.findData(int(value)) if isinstance(value, (int, float)) else 0
            widget.setCurrentIndex(idx if idx >= 0 else 0)
            widget.setMinimumWidth(width)
        elif base_name.endswith('freqOscShape') and type_hint == 'str':
            widget = QComboBox()
            widget.addItems(['sine', 'triangle'])
            val_to_set = value if value in ['sine', 'triangle'] else 'sine'
            widget.setCurrentText(str(val_to_set))
            widget.setMinimumWidth(width)
        elif base_name.endswith('panType') and type_hint == 'str':
            widget = QComboBox()
            widget.addItems(['linear', 'inward', 'outward'])
            val_to_set = value if value in ['linear', 'inward', 'outward'] else 'linear'
            widget.setCurrentText(str(val_to_set))
            widget.setMinimumWidth(width)
        elif base_name == 'transition_curve' and type_hint == 'str':
            widget = QComboBox()
            widget.addItems(['linear', 'logarithmic', 'exponential'])
            val_to_set = value if value in ['linear', 'logarithmic', 'exponential'] else 'linear'
            widget.setCurrentText(str(val_to_set))
            widget.setMinimumWidth(width)
        elif base_name.endswith('spatialDecoder') and type_hint == 'str':
            widget = QComboBox()
            widget.addItems(['itd_head', 'foa_cardioid'])
            val_to_set = value if value in ['itd_head', 'foa_cardioid'] else 'itd_head'
            widget.setCurrentText(str(val_to_set))
            widget.setMinimumWidth(width)
        elif base_name.endswith('spatialInterp') and type_hint in ('int', 'bool'):
            widget = QComboBox()
            widget.addItem('Linear', 0); widget.addItem('Lagrange3', 1)
            idx = widget.findData(int(value)) if isinstance(value, (int, float)) else 0
            widget.setCurrentIndex(idx if idx >= 0 else 0)
            widget.setMinimumWidth(width)
        elif base_name == 'pathShape' and type_hint == 'str' and hasattr(sound_creator, 'VALID_SAM_PATHS'):
            widget = QComboBox()
            widget.addItems(sound_creator.VALID_SAM_PATHS)
            val_to_set = value if value in sound_creator.VALID_SAM_PATHS else sound_creator.VALID_SAM_PATHS[0]
            widget.setCurrentText(str(val_to_set))
            widget.setMinimumWidth(width)
        elif base_name == 'pathType' and type_hint == 'str':
            widget = QComboBox()
            path_types = ['open', 'closed', 'discontinuous']
            widget.addItems(path_types)
            val_to_set = value if value in path_types else 'open'
            widget.setCurrentText(str(val_to_set))
            widget.setMinimumWidth(width)
        elif base_name == 'spatialDecoder' and type_hint == 'str':
            widget = QComboBox()
            widget.addItem('ITD Head (time-delay)', 'itd_head')
            widget.addItem('FOA Cardioid (legacy)', 'foa_cardioid')
            val_to_set = value if value in ['itd_head', 'foa_cardioid'] else 'itd_head'
            idx = widget.findData(val_to_set)
            widget.setCurrentIndex(idx if idx >= 0 else 0)
            widget.setMinimumWidth(width)
        elif type_hint == 'bool':
            widget = QCheckBox()
            if isinstance(value, str):
                checked = value.strip().lower() in ('1', 'true', 'yes', 'on')
            else:
                checked = bool(value)
            widget.setChecked(checked)
        else:
            widget = QLineEdit(str(value) if value is not None else "")
            if validator:
                widget.setValidator(validator)
            widget.setMinimumWidth(width)
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        return widget

    def _populate_envelope_controls(self):
        env_data = self.current_voice_data.get("volume_envelope") # main.py uses "volume_envelope"
        env_type = ENVELOPE_TYPE_NONE
        
        if isinstance(env_data, dict) and "type" in env_data:
            env_type_from_data = env_data["type"]
            if env_type_from_data in SUPPORTED_ENVELOPE_TYPES: # Ensure it's a supported type
                env_type = env_type_from_data
        
        self.env_type_combo.blockSignals(True)
        self.env_type_combo.setCurrentText(env_type)
        self.env_type_combo.blockSignals(False)
        
        self._on_envelope_type_change() # This will build and populate params for the selected type

    @pyqtSlot()
    def _on_envelope_type_change(self):
        self._clear_layout(self.env_params_layout)
        self.envelope_param_widgets = {} # Reset
        selected_type = self.env_type_combo.currentText()
        
        env_data = self.current_voice_data.get("volume_envelope") # From main.py format
        current_env_params = {}
        if isinstance(env_data, dict) and env_data.get("type") == selected_type:
            current_env_params = env_data.get("params", {})

        row = 0
        if selected_type == ENVELOPE_TYPE_LINEAR:
            # Definition from your original VoiceEditorDialogue for linear envelope
            params_def = [
                ("Fade Duration (s):", "fade_duration", 0.1, self.double_validator_non_negative, "float"), # Corresponds to 'create_linear_fade_envelope'
                ("Start Amplitude:", "start_amp", 0.0, self.double_validator_zero_to_one, "float"),
                ("End Amplitude:", "end_amp", 1.0, self.double_validator_zero_to_one, "float")
            ]
            # If your main.py "linear_fade" uses "fade_in_duration" and "fade_out_duration" like my previous dialog suggestion, adjust here.
            # Assuming the above params_def is what sound_creator.create_linear_fade_envelope expects.
            
            for label_text, param_name, default_val, validator_type, val_type in params_def:
                label = QLabel(label_text)
                entry = QLineEdit()
                entry.setValidator(copy.deepcopy(validator_type))
                val = current_env_params.get(param_name, default_val)
                if (
                    isinstance(val, (int, float))
                    and getattr(self.app, "prefs", None)
                    and getattr(self.app.prefs, "amplitude_display_mode", "absolute") == "dB"
                    and is_amp_key(param_name)
                ):
                    val = amplitude_to_db(float(val))
                entry.setText(str(val))

                entry.setMinimumWidth(self.ENTRY_WIDTH)
                entry.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                self.env_params_layout.addWidget(label, row, 0)
                self.env_params_layout.addWidget(entry, row, 1)
                self.envelope_param_widgets[param_name] = {'widget': entry, 'type': val_type}
                row += 1
            self.env_params_layout.setColumnStretch(1, 1) # Allow entry fields to expand
        
        # Add other envelope types (ADSR, Linen) here if you support them
        # elif selected_type == "adsr": ...
            
        if row == 0 and selected_type != ENVELOPE_TYPE_NONE : # If params were expected but none defined
             self.env_params_layout.addWidget(QLabel(f"No parameters defined for '{selected_type}' envelope."), 0,0)

        self.env_params_layout.addItem(QSpacerItem(20,10, QSizePolicy.Minimum, QSizePolicy.Expanding), row +1, 0)
        self.env_params_widget.setVisible(selected_type != ENVELOPE_TYPE_NONE and row > 0)


    def _populate_reference_step_combo(self):
        self.reference_step_combo.blockSignals(True)
        self.reference_step_combo.clear()
        steps = self.app.track_data.get("steps", [])
        if not steps:
            self.reference_step_combo.addItem("No Steps Available", -1)
            self.reference_step_combo.setEnabled(False)
        else:
            self.reference_step_combo.setEnabled(True)
            for i, step_data in enumerate(steps):
                desc = step_data.get("description","")
                item_text = f"Step {i+1}"
                if desc: item_text += f": {desc[:20]}{'...' if len(desc)>20 else ''}"
                self.reference_step_combo.addItem(item_text, i) # Store original index as data
        self.reference_step_combo.blockSignals(False)
        # Trigger update for voice combo if items were added
        if self.reference_step_combo.count() > 0:
            self._update_reference_voice_combo(self.reference_step_combo.currentIndex())


    @pyqtSlot(int)
    def _update_reference_voice_combo(self, combo_idx=-1): # combo_idx is from signal, not used directly if data is used
        self.reference_voice_combo.blockSignals(True)
        self.reference_voice_combo.clear()
        selected_step_index = self.reference_step_combo.currentData() # Get original step index

        if selected_step_index is None or selected_step_index < 0:
            self.reference_voice_combo.addItem("No Voices Available", -1)
            self.reference_voice_combo.setEnabled(False)
        else:
            try:
                voices = self.app.track_data["steps"][selected_step_index].get("voices", [])
                if not voices:
                    self.reference_voice_combo.addItem("No Voices in Step", -1)
                    self.reference_voice_combo.setEnabled(False)
                else:
                    self.reference_voice_combo.setEnabled(True)
                    for i, voice in enumerate(voices):
                        self.reference_voice_combo.addItem(f"Voice {i+1} ({voice.get('synth_function_name', 'N/A')[:25]})", i) # Store original voice index
            except IndexError:
                self.reference_voice_combo.addItem("Error loading voices", -1)
                self.reference_voice_combo.setEnabled(False)
        self.reference_voice_combo.blockSignals(False)
        # Trigger display update if items were added
        if self.reference_voice_combo.count() > 0 :
             self._update_reference_display(self.reference_voice_combo.currentIndex())


    @pyqtSlot(int)
    def _update_reference_display(self, combo_idx=-1): # combo_idx is from signal
        self.reference_details_text.clear()
        ref_step_idx = self.reference_step_combo.currentData()
        ref_voice_idx = self.reference_voice_combo.currentData()
        
        details = "Select a Step and Voice to see its details for reference."
        if ref_step_idx is not None and ref_step_idx >= 0 and \
           ref_voice_idx is not None and ref_voice_idx >= 0: # Valid selection
            
            is_editing_same_voice = (not self.is_new_voice and 
                                     self.step_index == ref_step_idx and 
                                     self.voice_index == ref_voice_idx)
            
            if is_editing_same_voice:
                # Show current UI state of the voice being edited for "reference"
                details = "Reference is the voice currently being edited.\nDetails reflect current UI settings (unsaved):\n------------------------------------\n"
                current_ui_data = self._collect_data_for_main_app() # Get data from current dialog UI
                details += f"Function: {current_ui_data.get('synth_function_name', 'N/A')}\n"
                details += f"Transition: {'Yes' if current_ui_data.get('is_transition', False) else 'No'}\n"
                details += "Parameters:\n"
                params = current_ui_data.get("params", {})
                func_name = current_ui_data.get('synth_function_name', '')
                is_trans = current_ui_data.get('is_transition', False)
            else:
                # Show saved data of the selected reference voice
                try:
                    voice_data = self.app.track_data["steps"][ref_step_idx]["voices"][ref_voice_idx]
                    details = f"Reference: Step {ref_step_idx+1}, Voice {ref_voice_idx+1} (Saved State)\n------------------------------------\n"
                    details += f"Function: {voice_data.get('synth_function_name', 'N/A')}\n"
                    details += f"Transition: {'Yes' if voice_data.get('is_transition', False) else 'No'}\n"
                    details += "Parameters:\n"
                    params = voice_data.get("params", {})
                    func_name = voice_data.get('synth_function_name', '')
                    is_trans = voice_data.get('is_transition', False)
                except IndexError:
                    details = "Error: Invalid Step or Voice index for reference."
                    params = {}
                    func_name = ''
                    is_trans = False
            
            prefs_obj = getattr(self.app, "prefs", None)
            func_display_prefs = {}
            if isinstance(getattr(prefs_obj, "voice_detail_display", None), dict):
                func_display_prefs = prefs_obj.voice_detail_display.get(func_name, {}) or {}

            default_params = OrderedDict()
            try:
                default_params = self._get_default_params(func_name, is_trans)
            except Exception:
                default_params = OrderedDict()

            def _should_display_param(param_name: str) -> bool:
                cfg = func_display_prefs.get(param_name, {}) if isinstance(func_display_prefs, dict) else {}
                if isinstance(cfg, dict) and "display" in cfg:
                    return cfg.get("display", True)
                return get_default_param_display_flag(param_name)

            # Build the ordered list of params based on defaults first, then extras
            ordered_keys = list(default_params.keys())
            extra_keys = [k for k in params.keys() if k not in default_params]
            ordered_keys.extend(sorted(extra_keys))

            shown_any = False
            for param_name in ordered_keys:
                if not _should_display_param(param_name):
                    continue

                cfg = func_display_prefs.get(param_name, {}) if isinstance(func_display_prefs, dict) else {}
                value = params.get(
                    param_name,
                    cfg.get("default", default_params.get(param_name)),
                )

                if value is None:
                    continue

                shown_any = True
                display_val = f"{value:.4g}" if isinstance(value, float) else value
                details += f"  {param_name}: {display_val}\n"

            if not shown_any and "Function:" in details:
                details += "  (No parameters selected for display)\n"
            elif not params and "Function:" in details:
                details += "  (No parameters defined)\n"

            # Envelope details (either current UI if editing same, or saved for other ref)
            if is_editing_same_voice:
                env_data_collected = self._collect_data_from_ui().get("volume_envelope") # Dialog's internal structure uses "envelope"
                env_data_to_display = env_data_collected if env_data_collected else self.current_voice_data.get("volume_envelope")
            else:
                try:
                     env_data_to_display = self.app.track_data["steps"][ref_step_idx]["voices"][ref_voice_idx].get("volume_envelope")
                except: env_data_to_display = None

            if env_data_to_display and isinstance(env_data_to_display, dict):
                details += f"\nEnvelope Type: {env_data_to_display.get('type', 'N/A')}\n  Envelope Params:\n"
                env_params = env_data_to_display.get('params', {})
                if env_params:
                    for k,v in sorted(env_params.items()): details += f"    {k}: {v:.4g if isinstance(v,float) else v}\n"
                else: details += "  (No envelope parameters defined)\n"
            elif "Function:" in details: # Only if not an error message
                 details += "\nEnvelope Type: None\n"

        elif ref_step_idx is not None and ref_step_idx >= 0:
            details = "Select a Voice from the chosen Step."
        elif self.reference_step_combo.count() > 0 and self.reference_step_combo.itemData(0) == -1:
            details = "No steps available in the track to reference."
            
        self.reference_details_text.setPlainText(details)


    def _select_initial_reference_voice(self, voice_index_to_select):
        """Tries to select a specific voice index in the reference voice combo."""
        if self.reference_voice_combo.isEnabled(): # Only if combo is enabled (has voices)
            voice_combo_index = self.reference_voice_combo.findData(voice_index_to_select)
            if voice_combo_index != -1:
                self.reference_voice_combo.setCurrentIndex(voice_combo_index)
            elif self.reference_voice_combo.count() > 0 and self.reference_voice_combo.itemData(0) != -1: # Not "No voices"
                self.reference_voice_combo.setCurrentIndex(0) # Fallback to first actual voice

    @pyqtSlot()
    def _apply_reference_as_end_state(self):
        if not self.transition_check.isChecked():
            return

        ref_step_idx = self.reference_step_combo.currentData()
        ref_voice_idx = self.reference_voice_combo.currentData()
        if ref_step_idx is None or ref_step_idx < 0 or ref_voice_idx is None or ref_voice_idx < 0:
            QMessageBox.warning(self, "Reference Voice", "Please select a valid reference voice.")
            return

        try:
            ref_voice = self.app.track_data["steps"][ref_step_idx]["voices"][ref_voice_idx]
        except Exception as e:
            QMessageBox.warning(self, "Reference Voice", f"Could not load reference voice: {e}")
            return

        ref_params = ref_voice.get("params", {})
        amplitude_in_db = getattr(self.app, "prefs", None) and getattr(self.app.prefs, "amplitude_display_mode", "absolute") == "dB"

        for name, data in self.param_widgets.items():
            if not name.startswith("end"):
                continue

            widget = data["widget"]
            _prefix, base = self._split_name_prefix(name)
            candidates = [base, base[0].lower() + base[1:] if base else base]
            value_found = False
            for cand in candidates:
                if cand in ref_params:
                    raw_val = ref_params[cand]
                    value_found = True
                    break
                elif f"end{cand}" in ref_params:
                    raw_val = ref_params[f"end{cand}"]
                    value_found = True
                    break
                elif f"end_{cand}" in ref_params:
                    raw_val = ref_params[f"end_{cand}"]
                    value_found = True
                    break
            if not value_found:
                continue

            display_val = raw_val
            if (
                amplitude_in_db
                and isinstance(raw_val, (int, float))
                and is_amp_key(name)
            ):
                display_val = amplitude_to_db(float(raw_val))

            if isinstance(widget, QLineEdit):
                widget.setText(str(display_val))
            elif isinstance(widget, QComboBox):
                idx = widget.findData(display_val)
                if idx >= 0:
                    widget.setCurrentIndex(idx)
                else:
                    widget.setCurrentText(str(display_val))
            elif isinstance(widget, QCheckBox):
                if isinstance(display_val, str):
                    checked = display_val.strip().lower() in ('1', 'true', 'yes', 'on')
                else:
                    checked = bool(display_val)
                widget.setChecked(checked)

            self.current_voice_data.setdefault("params", {})[name] = raw_val


    @pyqtSlot()
    def on_synth_function_change(self):
        selected_func = self.synth_func_combo.currentText()
        if not selected_func or selected_func.startswith("Error:"):
            return

        # Preserve current parameter values before switching function
        current_params = self._collect_data_from_ui().get("params", {})
        if self.current_voice_data.get("is_transition", False):
            self._transition_params = OrderedDict(current_params)
        else:
            self._standard_params = OrderedDict(current_params)

        # Auto-update transition checkbox based on function name convention
        is_transition_by_name = selected_func.endswith("_transition")
        self.transition_check.blockSignals(True)
        self.transition_check.setChecked(is_transition_by_name)
        self.transition_check.blockSignals(False)

        # Update current_voice_data to reflect selection
        self.current_voice_data["synth_function_name"] = selected_func
        self.current_voice_data["is_transition"] = is_transition_by_name

        try:
            # Merge existing stored params with defaults for the new function
            new_std_defaults = self._get_default_params(selected_func, False)
            new_trans_defaults = self._get_default_params(selected_func, True)

            if not new_std_defaults and self._standard_params:
                new_std_defaults = OrderedDict(self._standard_params)
            if not new_trans_defaults and self._transition_params:
                new_trans_defaults = OrderedDict(self._transition_params)

            if new_std_defaults:
                self._standard_params = self._merge_params(self._standard_params, new_std_defaults)
            if new_trans_defaults:
                self._transition_params = self._merge_params(self._transition_params, new_trans_defaults)

            target_params = (
                self._transition_params if is_transition_by_name else self._standard_params
            )
            self.current_voice_data["params"] = OrderedDict(target_params or current_params)
        except Exception as exc:  # noqa: BLE001 - surface UI-safe error to user
            traceback.print_exc()
            QMessageBox.warning(
                self,
                "Parameter Load Error",
                f"Failed to load parameters for '{selected_func}':\n{exc}",
            )
            fallback_params = (
                self._transition_params if is_transition_by_name else self._standard_params
            ) or current_params
            self.current_voice_data["params"] = OrderedDict(fallback_params)
        finally:
            try:
                self.populate_parameters()
            finally:
                self._update_swap_button_visibility()

    @pyqtSlot(int)
    def on_transition_toggle(self, state):
        # Preserve parameters from current mode
        current_params = self._collect_data_from_ui().get("params", {})
        if self.current_voice_data.get("is_transition", False):
            self._transition_params = OrderedDict(current_params)
        else:
            self._standard_params = OrderedDict(current_params)

        is_transition = bool(state == Qt.Checked)
        self.current_voice_data["is_transition"] = is_transition

        func_name = self.synth_func_combo.currentText()
        if is_transition:
            if not self._transition_params:
                source = self._standard_params if self._standard_params else current_params
                self._transition_params = self._convert_params_to_transition(source, func_name)
            self.current_voice_data["params"] = OrderedDict(self._transition_params)
        else:
            if not self._standard_params:
                source = self._transition_params if self._transition_params else current_params
                self._standard_params = self._convert_params_to_standard(source, func_name)
            self.current_voice_data["params"] = OrderedDict(self._standard_params)

        self.populate_parameters()
        self._update_swap_button_visibility()

    def _update_swap_button_visibility(self):
        if hasattr(self, "swap_params_button"):
            self.swap_params_button.setVisible(self.transition_check.isChecked())
        if hasattr(self, "set_ref_end_button"):
            self.set_ref_end_button.setEnabled(self.transition_check.isChecked())

    # ---- Internal helpers for parameter state management ----

    def _merge_params(self, existing: OrderedDict, defaults: OrderedDict) -> OrderedDict:
        """Merge ``existing`` with ``defaults`` keeping only keys in defaults."""
        merged = OrderedDict()
        for name, default_val in defaults.items():
            merged[name] = existing.get(name, default_val)
        return merged

    def _convert_params_to_transition(self, params: OrderedDict, func_name: str) -> OrderedDict:
        """Create a transition-param dict from non-transition ``params``."""
        new_default_params = self._get_default_params(func_name, True)

        def _norm(key: str) -> str:
            return key.replace("_", "").lower()

        base_map = {_norm(k): v for k, v in params.items()}
        updated = OrderedDict()
        for name, default_val in new_default_params.items():
            if name in params:
                updated[name] = params[name]
            elif name.startswith("start"):
                base_key = _norm(name[len("start"):])
                updated[name] = base_map.get(base_key, default_val)
            elif name.startswith("end"):
                base_key = _norm(name[len("end"):])
                if base_key in base_map:
                    updated[name] = base_map[base_key]
                else:
                    start_name = "start" + name[len("end"):]
                    updated[name] = params.get(start_name, default_val)
            else:
                updated[name] = params.get(name, default_val)
        return updated

    def _convert_params_to_standard(self, params: OrderedDict, func_name: str) -> OrderedDict:
        """Create a non-transition param dict from transition ``params``."""
        new_default_params = self._get_default_params(func_name, False)

        def _norm(key: str) -> str:
            return key.replace("_", "").lower()

        start_map = {
            _norm(k[len("start"):]): v
            for k, v in params.items()
            if k.startswith("start")
        }
        updated = OrderedDict()
        for name, default_val in new_default_params.items():
            base_key = _norm(name)
            if base_key in start_map:
                updated[name] = start_map[base_key]
            else:
                updated[name] = params.get(name, default_val)
        return updated

    @pyqtSlot()
    def swap_transition_parameters(self):
        if not self.transition_check.isChecked():
            return

        for name, data in list(self.param_widgets.items()):
            if name.startswith("start"):
                base = name[len("start"):]
                end_name = "end" + base
                if end_name in self.param_widgets:
                    start_w = self.param_widgets[name]["widget"]
                    end_w = self.param_widgets[end_name]["widget"]
                    if isinstance(start_w, QLineEdit) and isinstance(end_w, QLineEdit):
                        tmp = start_w.text()
                        start_w.setText(end_w.text())
                        end_w.setText(tmp)
                    elif isinstance(start_w, QComboBox) and isinstance(end_w, QComboBox):
                        tmp = start_w.currentText()
                        start_w.setCurrentText(end_w.currentText())
                        end_w.setCurrentText(tmp)
                    elif isinstance(start_w, QCheckBox) and isinstance(end_w, QCheckBox):
                        tmp = start_w.isChecked()
                        start_w.setChecked(end_w.isChecked())
                        end_w.setChecked(tmp)

    def _handle_custom_curve(self, combo_box):
        if combo_box.currentText() == 'Custom...':
            text, ok = QInputDialog.getText(self, 'Custom Transition Formula',
                                           'Enter custom transition formula:')
            if ok and text.strip():
                custom_text = text.strip()
                if combo_box.findText(custom_text) == -1:
                    combo_box.insertItem(combo_box.count() - 1, custom_text)
                combo_box.setCurrentText(custom_text)
            else:
                combo_box.setCurrentIndex(0)

    def _split_name_prefix(self, name: str):
        for p in ('start_', 'start', 'end_', 'end'):
            if name.startswith(p):
                return p, name[len(p):]
        return '', name

    def _parse_lr_suffix(self, base: str):
        patterns = [('_L', '_R'), ('_R', '_L'), ('L', 'R'), ('R', 'L')]
        for ls, rs in patterns:
            if base.endswith(ls):
                return base[:-len(ls)], ls, rs
        return None

    def _swap_lr(self, names):
        for i in range(0, len(names), 2):
            n1, n2 = names[i], names[i+1]
            if n1 in self.param_widgets and n2 in self.param_widgets:
                w1 = self.param_widgets[n1]['widget']
                w2 = self.param_widgets[n2]['widget']
                if isinstance(w1, QLineEdit) and isinstance(w2, QLineEdit):
                    tmp = w1.text(); w1.setText(w2.text()); w2.setText(tmp)
                elif isinstance(w1, QComboBox) and isinstance(w2, QComboBox):
                    tmp = w1.currentText(); w1.setCurrentText(w2.currentText()); w2.setCurrentText(tmp)
                elif isinstance(w1, QCheckBox) and isinstance(w2, QCheckBox):
                    tmp = w1.isChecked(); w1.setChecked(w2.isChecked()); w2.setChecked(tmp)


    # --- Helper methods for collecting UI data ---
    def _collect_data_for_main_app(self):
        """
        Collects current UI data and returns it in the format expected by main app.
        This is similar to save_voice() but doesn't validate or save, just collects.
        """
        # Collect synth parameters
        synth_params = {}
        for name, data in self.param_widgets.items():
            widget, param_type = data['widget'], data['type']
            value = None

            if param_type == 'sweep_list':
                value = widget.get_values()
                if value:
                    synth_params[name] = value
                continue
            if param_type == 'static_notch_list':
                value = widget.get_values()
                if value:
                    synth_params[name] = value
                continue

            if isinstance(widget, QCheckBox):
                value = widget.isChecked()
            elif isinstance(widget, QComboBox):
                value_str = widget.currentText()
                data_val = widget.currentData()
                try:
                    if param_type == 'int':
                        value = int(data_val if data_val is not None else value_str)
                    elif param_type == 'float':
                        value = float(data_val if data_val is not None else value_str)
                    else:
                        value = data_val if data_val is not None else value_str
                except (ValueError, TypeError):
                    value = None
            elif isinstance(widget, QLineEdit):
                value_str = widget.text().strip()
                if value_str:
                    try:
                        if param_type == 'int':
                            value = int(value_str)
                        elif param_type == 'float': 
                            value = float(value_str.replace(',', '.'))
                        else: 
                            value = value_str
                    except ValueError:
                        value = None  # Will use default
                else:
                    value = None
            
            if value is not None:

                if (
                    param_type == 'float'
                    and getattr(self.app, 'prefs', None)
                    and getattr(self.app.prefs, 'amplitude_display_mode', 'absolute') == 'dB'
                    and is_amp_key(name)
                ):
                    value = db_to_amplitude(float(value))

                synth_params[name] = value
        
        # Collect envelope data
        envelope_data = None
        selected_env_type = self.env_type_combo.currentText()
        if selected_env_type != ENVELOPE_TYPE_NONE:
            env_params = {}
            for name, data in self.envelope_param_widgets.items():
                widget, param_type = data['widget'], data['type']
                if isinstance(widget, QLineEdit):
                    value_str = widget.text().strip()
                    if value_str:
                        try:
                            if param_type == 'float':
                                val = float(value_str.replace(',', '.'))

                                if (
                                    getattr(self.app, 'prefs', None)
                                    and getattr(self.app.prefs, 'amplitude_display_mode', 'absolute') == 'dB'
                                    and is_amp_key(name)
                                ):
                                    val = db_to_amplitude(val)

                                env_params[name] = val
                            elif param_type == 'int': 
                                env_params[name] = int(value_str)
                            else: 
                                env_params[name] = value_str
                        except ValueError:
                            pass  # Skip invalid values
            
            if env_params:
                envelope_data = {"type": selected_env_type, "params": env_params}
        
        return {
            "synth_function_name": self.synth_func_combo.currentText(),
            "is_transition": self.transition_check.isChecked(),
            "params": synth_params,
            "volume_envelope": envelope_data
        }
    
    def _collect_data_from_ui(self):
        """
        Alias for _collect_data_for_main_app() for consistency with existing code.
        """
        return self._collect_data_for_main_app()

    def _get_param_range_hint(self, param_name_base): # param_name_base is without start/end
        name_lower = param_name_base.lower()
        # More specific first
        if 'pan' in name_lower: return '-1 L to 1 R'
        if 'noiseType' in name_lower: return '1:W,2:P,3:B'
        if 'shape' in name_lower and 'path' not in name_lower : return 'e.g. 0-10'
        if 'pathtype' in name_lower: return 'open, closed, discontinuous'
        if 'pathShape' in name_lower: return 'e.g. circle, line'


        if any(s in name_lower for s in ['amp', 'gain', 'level', 'depth']) and 'mod' in name_lower : return 'e.g. 0.0-1.0'
        if any(s in name_lower for s in ['amp', 'gain', 'level']) : return 'e.g. 0.0-1.0+' # Amplitudes can exceed 1 prior to mixing
        
        if any(s in name_lower for s in ['freq', 'frequency', 'rate']): return 'Hz, >0'
        if 'q' == name_lower or 'rq' == name_lower : return '>0, e.g. 0.1-20' # Quality factor
        if any(s in name_lower for s in ['dur', 'attack', 'decay', 'release', 'delay', 'interval']): return 'secs, >=0'
        if 'phase' in name_lower and 'offset' not in name_lower: return 'radians, e.g. 0-2pi'
        if 'radius' in name_lower : return '>=0'
        if 'width' in name_lower : return 'Hz or ratio'
        if 'ratio' in name_lower : return '>0'
        if 'amount' in name_lower : return 'varies'
        if 'factor' in name_lower : return 'varies'


        return '' # No specific hint

    def _get_param_tooltip(self, func_name, param_name):
        """Return a tooltip description for a given parameter name."""
        descriptions = PARAM_TOOLTIPS.get(func_name, {})
        if param_name in descriptions:
            return descriptions[param_name]

        # Handle parameters with start/end prefixes used for transitions
        if param_name.startswith('start') or param_name.startswith('end'):
            prefix = 'Start' if param_name.startswith('start') else 'End'
            base = param_name[5:] if param_name.startswith('start') else param_name[3:]
            if base:
                base = base[0].lower() + base[1:]
            base_desc = descriptions.get(base)
            if base_desc:
                return f"{prefix} value for {base_desc.lower()}"

        return None

    def _get_default_params(self, func_name_from_combo: str, is_transition_mode: bool) -> OrderedDict:
        """
        Retrieves an OrderedDict of default parameters for a given synth function.
        Uses the shared get_default_params_for_function helper.
        """
        return get_default_params_for_function(func_name_from_combo, is_transition_mode)


    def save_voice(self):
        new_synth_params = {}
        error_occurred = False
        validation_errors = []

        # Collect Synth Parameters
        for name, data in self.param_widgets.items():
            widget, param_type = data['widget'], data['type']
            value = None
            widget.setStyleSheet("") # Clear previous error styles

            if param_type == 'sweep_list':
                sweeps_value = widget.get_values()
                if sweeps_value:
                    new_synth_params[name] = sweeps_value
                continue

            if param_type == 'static_notch_list':
                static_value = widget.get_values()
                if static_value:
                    new_synth_params[name] = static_value
                continue

            if param_type == 'json':
                new_synth_params[name] = data.get('value', [])
                continue

            if isinstance(widget, QCheckBox):
                value = widget.isChecked()
            elif isinstance(widget, QComboBox):
                value_str = widget.currentText()
                data_val = widget.currentData()
                if param_type == 'int':
                    try:
                        value = int(data_val if data_val is not None else value_str)
                    except (ValueError, TypeError):
                        error_occurred = True
                        validation_errors.append(f"Invalid int for '{name}': {value_str}")
                        widget.setStyleSheet("border: 1px solid red;")
                elif param_type == 'float':
                    try:
                        value = float(data_val if data_val is not None else value_str)
                    except (ValueError, TypeError):
                        error_occurred = True
                        validation_errors.append(f"Invalid float for '{name}': {value_str}")
                        widget.setStyleSheet("border: 1px solid red;")
                else:
                    value = data_val if data_val is not None else value_str
            elif isinstance(widget, QLineEdit):
                value_str = widget.text().strip()
                if not value_str and param_type != 'str': # Allow empty strings if type is str, otherwise might be error or None
                     # Check if param can be None or needs a value
                     # For now, let's assume None is okay if empty and not string.
                     # Or, find default for this param:
                     # default_val_for_name = self._get_default_params(self.synth_func_combo.currentText(), self.transition_check.isChecked()).get(name)
                     # if default_val_for_name is not None: value = default_val_for_name
                     # else: # param cannot be None and is empty
                     #    error_occurred=True; validation_errors.append(f"Parameter '{name}' cannot be empty."); widget.setStyleSheet("border: 1px solid red;")
                     value = None # Allow None if field is empty (synth function should handle None with defaults)
                elif value_str: # Only parse if not empty
                    try:
                        if param_type == 'int': value = int(value_str)
                        elif param_type == 'float': value = float(value_str.replace(',', '.')) # Allow comma for float
                        else: value = value_str # String type
                    except ValueError:
                        error_occurred=True; validation_errors.append(f"Invalid {param_type} for '{name}': {value_str}"); widget.setStyleSheet("border: 1px solid red;")
            
            if value is not None: # Only add if value is set (allows params to be omitted if they are None)
                new_synth_params[name] = value
            elif (
                param_type == 'str'
                and isinstance(widget, QLineEdit)
                and widget.text().strip() == ""
            ):
                new_synth_params[name] = ""

        # Preserve parameters hidden by user preferences
        if getattr(self, "_hidden_params", None):
            for name, value in self._hidden_params.items():
                new_synth_params.setdefault(name, value)

        # Collect Envelope Parameters
        new_envelope_data = None
        selected_env_type = self.env_type_combo.currentText()
        if selected_env_type != ENVELOPE_TYPE_NONE:
            new_env_params = {}
            for name, data in self.envelope_param_widgets.items():
                widget, param_type = data['widget'], data['type']
                value = None
                widget.setStyleSheet("") # Clear error styles
                if isinstance(widget, QLineEdit):
                    value_str = widget.text().strip()
                    if not value_str:
                        error_occurred=True; validation_errors.append(f"Envelope parameter '{name}' cannot be empty."); widget.setStyleSheet("border: 1px solid red;")
                    else:
                        try:
                            if param_type == 'float': value = float(value_str.replace(',', '.'))
                            elif param_type == 'int': value = int(value_str) # If any int env params
                            else: value = value_str
                            # Example validation for envelope params
                            if 'amp' in name.lower() and not (0.0 <= value <= 1.0):
                                validation_errors.append(f"Envelope amp '{name}' ({value}) out of 0-1 range (warning).") # Non-blocking
                            if 'dur' in name.lower() and value < 0:
                                error_occurred=True; validation_errors.append(f"Envelope duration '{name}' ({value}) cannot be negative."); widget.setStyleSheet("border: 1px solid red;")

                        except ValueError:
                            error_occurred=True; validation_errors.append(f"Invalid envelope {param_type} for '{name}': {value_str}"); widget.setStyleSheet("border: 1px solid red;")
                if value is not None: new_env_params[name] = value
            
            if not any(f"Envelope parameter '{n}'" in e for n in self.envelope_param_widgets for e in validation_errors if "cannot be empty" in e): # check if fatal error occurred
                new_envelope_data = {"type": selected_env_type, "params": new_env_params}

        if error_occurred:
            QMessageBox.warning(self, "Input Error", "Please correct highlighted fields:\n\n" + "\n".join(validation_errors))
            return

        # Final voice data structure for main.py
        final_voice_data = {
            "synth_function_name": self.synth_func_combo.currentText(),
            "is_transition": self.transition_check.isChecked(),
            "params": new_synth_params,
            "volume_envelope": new_envelope_data,  # This is the correct key for main.py
            "description": self.current_voice_data.get("description", ""),
        }
        
        # Update the main application's track_data
        try:
            target_step = self.app.track_data["steps"][self.step_index]
            if "voices" not in target_step: # Should exist if step exists
                target_step["voices"] = []
            
            if self.is_new_voice:
                target_step["voices"].append(final_voice_data)
            elif 0 <= self.voice_index < len(target_step["voices"]):
                target_step["voices"][self.voice_index] = final_voice_data
            else: # Should not happen if initialized correctly
                QMessageBox.critical(self.app, "Save Error", "Voice index out of bounds during save.")
                self.reject() # Close without saving
                return
            
            self.accept() # Signal to main.py that dialog was accepted (and data is ready)
        except IndexError:
            QMessageBox.critical(self.app, "Save Error", "Failed to save voice (step index issue). Check track data integrity.")
            self.reject()
        except Exception as e:
            QMessageBox.critical(self.app, "Save Error", f"An unexpected error occurred while saving voice data:\n{e}")
            traceback.print_exc()
            self.reject()

    #get_voice_data is not strictly needed if main.py updates on accept() by re-reading from track_data
    #but if main.py explicitly calls dialog.get_voice_data(), it should return the final structure.
    def get_voice_data(self):
        # This would be called by main.py if it needs the data explicitly after accept()
        # For this version, main.py modifies its own track_data directly in save_voice() before accept()
        # So, this method can just return the self.current_voice_data which was updated.
        # To be more robust, it should return what was actually decided to be saved.
        # However, the original save_voice directly modifies self.app.track_data.
        # Let's assume main.py will refresh from its track_data.
        # If main.py *does* call this, current_voice_data is what was loaded/being edited.
        # A better pattern for dialogs is to return the *new* data, not modify external data directly.
        # But sticking to "original version" structure for save_voice.
        return self.current_voice_data # This reflects data at dialog open or after UI changes if not saved yet.
                                      # For the "saved" data, main.py should re-read its own track_data.

    def save_preset(self):
        """Save the current voice parameters to a preset file."""
        preset_data = self._collect_data_from_ui()
        preset = VoicePreset(
            synth_function_name=preset_data.get("synth_function_name", ""),
            is_transition=preset_data.get("is_transition", False),
            params=preset_data.get("params", {}),
            volume_envelope=preset_data.get("volume_envelope"),
            description=self.current_voice_data.get("description", ""),
        )
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Voice Preset",
            "",
            f"Voice Presets (*{VOICE_FILE_EXTENSION})",
        )
        if not path:
            return
        try:
            save_voice_preset(preset, path)
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))

    def load_preset(self):
        """Load a preset from file and apply it to the editor."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Voice Preset",
            "",
            f"Voice Presets (*{VOICE_FILE_EXTENSION})",
        )
        if not path:
            return
        try:
            preset = load_voice_preset(path)
            self.apply_voice_preset(preset)
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))

    def apply_voice_preset(self, preset: VoicePreset) -> None:
        """Update the dialog UI from ``preset``."""
        self.current_voice_data = {
            "synth_function_name": preset.synth_function_name,
            "is_transition": preset.is_transition,
            "params": preset.params,
            "volume_envelope": preset.volume_envelope,
            "description": preset.description,
        }

        if self.synth_func_combo.findText(preset.synth_function_name) != -1:
            self.synth_func_combo.setCurrentText(preset.synth_function_name)
        elif self.synth_func_combo.count() > 0:
            self.synth_func_combo.setCurrentIndex(0)
        self.transition_check.setChecked(preset.is_transition)
        self.populate_parameters()
        self._populate_envelope_controls()


def get_default_params_for_function(func_name_from_combo: str, is_transition_mode: bool) -> OrderedDict:
    """
    Shared helper that returns the default parameter OrderedDict for a synth function.
    Extracted from :meth:`VoiceEditorDialog._get_default_params` for reuse in other dialogs.
    """
    def _introspect_params(func_name: str) -> OrderedDict:
        """Return defaults inferred from the synth function signature.

        We intentionally include parameters even when they have no Python default by
        assigning ``None``. This ensures the GUI still renders editable fields for
        required arguments instead of appearing empty when a function adds a new
        parameter without updating the static definitions.
        """

        ordered = OrderedDict()
        func = sound_creator.SYNTH_FUNCTIONS.get(func_name)
        if not func:
            return ordered

        try:
            sig = inspect.signature(func)
        except (TypeError, ValueError):
            return ordered

        for name, param in sig.parameters.items():
            if name in {"duration", "sample_rate"}:
                continue
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                # **params buckets (e.g., isochronic_tone) do not expose defaults,
                # so we rely on manual definitions instead.
                return OrderedDict()

            default_val = None if param.default is inspect._empty else param.default
            ordered[name] = default_val

        return ordered

    # This map should align QComboBox text with keys in param_definitions
    internal_func_key_map = {
        "Rhythmic Waveshaping": "rhythmic_waveshaping",
        "Rhythmic Waveshaping Transition": "rhythmic_waveshaping_transition",
        "Stereo AM Independent": "stereo_am_independent",
        "Stereo AM Independent Transition": "stereo_am_independent_transition",
        "Wave Shape Stereo AM": "wave_shape_stereo_am",
        "Wave Shape Stereo AM Transition": "wave_shape_stereo_am_transition",
        "Spatial Angle Modulation (SAM Engine)": "spatial_angle_modulation", # Uses Node/SAMVoice directly
        "Spatial Angle Modulation (SAM Engine Transition)": "spatial_angle_modulation_transition",
        "Spatial Angle Modulation (SAM2)": "spatial_angle_modulation_sam2",
        "Spatial Angle Modulation (SAM2 Transition)": "spatial_angle_modulation_sam2_transition",
        "Binaural Beat": "binaural_beat",
        "Binaural Beat Transition": "binaural_beat_transition",
        "Monaural Beat Stereo Amps": "monaural_beat_stereo_amps",
        "Monaural Beat Stereo Amps Transition": "monaural_beat_stereo_amps_transition",
        "Spatial Angle Modulation (Monaural Core)": "spatial_angle_modulation_monaural_beat", # Uses monaural_beat as core
        "Spatial Angle Modulation (Monaural Core Transition)": "spatial_angle_modulation_monaural_beat_transition",
        "Isochronic Tone": "isochronic_tone",
        "Isochronic Tone Transition": "isochronic_tone_transition",
        "QAM Beat": "qam_beat", # Ensure this mapping is correct for your UIAdd commentMore actions
        "QAM Beat Transition": "qam_beat_transition",
        "Hybrid QAM Monaural Beat": "hybrid_qam_monaural_beat",
        "Hybrid QAM Monaural Beat Transition": "hybrid_qam_monaural_beat_transition",
        "Dual Pulse Binaural": "dual_pulse_binaural",
        "Dual Pulse Binaural Transition": "dual_pulse_binaural_transition",
        # Explicit snake_case mappings to ensure robustness
        "rhythmic_waveshaping": "rhythmic_waveshaping",
        "stereo_am_independent": "stereo_am_independent",
        "wave_shape_stereo_am": "wave_shape_stereo_am",
        "spatial_angle_modulation": "spatial_angle_modulation",
        "spatial_angle_modulation_sam2": "spatial_angle_modulation_sam2",
        "binaural_beat": "binaural_beat",
        "monaural_beat_stereo_amps": "monaural_beat_stereo_amps",
        "spatial_angle_modulation_monaural_beat": "spatial_angle_modulation_monaural_beat",
        "isochronic_tone": "isochronic_tone",
        "qam_beat": "qam_beat",
        "hybrid_qam_monaural_beat": "hybrid_qam_monaural_beat",
        "dual_pulse_binaural": "dual_pulse_binaural",
        "subliminal_encode": "subliminal_encode",
    }

    # Automatically map any ``*_transition`` synth function name to the base
    # entry in ``param_definitions`` so that transition variants coming
    # directly from ``sound_creator.SYNTH_FUNCTIONS`` still resolve to a known
    # parameter list.
    if func_name_from_combo not in internal_func_key_map and func_name_from_combo.endswith("_transition"):
        base_name = func_name_from_combo.removesuffix("_transition")
        internal_func_key_map[func_name_from_combo] = base_name

    param_definitions = {
        "rhythmic_waveshaping": { # This is an example, ensure it's correctAdd commentMore actions
            "standard": [
                ('amp', 0.25), ('carrierFreq', 200), ('modFreq', 4),
                ('modDepth', 1.0), ('shapeAmount', 5.0), ('pan', 0)
            ],
            "transition": [
                ('amp', 0.25), ('startCarrierFreq', 200), ('endCarrierFreq', 80),
                ('startModFreq', 12), ('endModFreq', 7.83),
                ('startModDepth', 1.0), ('endModDepth', 1.0),
                ('startShapeAmount', 5.0), ('endShapeAmount', 5.0), ('pan', 0),
                ('initial_offset', 0.0), ('duration', 0.0), ('transition_curve', 'linear')
            ]
        },
        "stereo_am_independent": { # This is an example, ensure it's correct
            "standard": [
                ('amp', 0.25), ('carrierFreq', 200.0), ('modFreqL', 4.0),
                ('modDepthL', 0.8), ('modPhaseL', 0), ('modFreqR', 4.0),
                ('modDepthR', 0.8), ('modPhaseR', 0), ('stereo_width_hz', 0.2)
            ],
            "transition": [
                ('amp', 0.25), ('startCarrierFreq', 200), ('endCarrierFreq', 250),
                ('startModFreqL', 4), ('endModFreqL', 6),
                ('startModDepthL', 0.8), ('endModDepthL', 0.8),
                ('startModPhaseL', 0),
                ('startModFreqR', 4.1), ('endModFreqR', 5.9),
                ('startModDepthR', 0.8), ('endModDepthR', 0.8),
                ('startModPhaseR', 0),
                ('startStereoWidthHz', 0.2), ('endStereoWidthHz', 0.2),
                ('initial_offset', 0.0), ('duration', 0.0), ('transition_curve', 'linear')
            ]
        },
        "wave_shape_stereo_am": { # This is an example, ensure it's correct
            "standard": [
                ('amp', 0.15), ('carrierFreq', 200), ('shapeModFreq', 4),
                ('shapeModDepth', 0.8), ('shapeAmount', 0.5),
                ('stereoModFreqL', 4.1), ('stereoModDepthL', 0.8),
                ('stereoModPhaseL', 0), ('stereoModFreqR', 4.0),
                ('stereoModDepthR', 0.8), ('stereoModPhaseR', math.pi / 2)
            ],
            "transition": [
                ('amp', 0.15), ('startCarrierFreq', 200), ('endCarrierFreq', 100),
                ('startShapeModFreq', 4), ('endShapeModFreq', 8),
                ('startShapeModDepth', 0.8), ('endShapeModDepth', 0.8),
                ('startShapeAmount', 0.5), ('endShapeAmount', 0.5),
                ('startStereoModFreqL', 4.1), ('endStereoModFreqL', 6.0),
                ('startStereoModDepthL', 0.8), ('endStereoModDepthL', 0.8),
                ('startStereoModPhaseL', 0),
                ('startStereoModFreqR', 4.0), ('endStereoModFreqR', 6.1),
                ('startStereoModDepthR', 0.9), ('endStereoModDepthR', 0.9),
                ('startStereoModPhaseR', math.pi / 2),
                ('initial_offset', 0.0), ('duration', 0.0), ('transition_curve', 'linear')
            ]
        },
        "spatial_angle_modulation": {
            "standard": [
                ('amp', 0.7), ('carrierFreq', 440.0), ('beatFreq', 4.0),
                ('pathShape', 'circle'), ('pathRadius', 1.0),
                ('arcStartDeg', 0.0), ('arcEndDeg', 360.0),
                ('frame_dur_ms', 46.4), ('overlap_factor', 8)
            ],
            "transition": [
                ('amp', 0.7),
                ('startCarrierFreq', 440.0), ('endCarrierFreq', 440.0),
                ('startBeatFreq', 4.0), ('endBeatFreq', 4.0),
                ('startPathShape', 'circle'), ('endPathShape', 'circle'),
                ('startPathRadius', 1.0), ('endPathRadius', 1.0),
                ('startArcStartDeg', 0.0), ('endArcStartDeg', 0.0),
                ('startArcEndDeg', 360.0), ('endArcEndDeg', 360.0),
                ('frame_dur_ms', 46.4), ('overlap_factor', 8),
                ('initial_offset', 0.0), ('duration', 0.0), ('transition_curve', 'linear')
            ]
        },
        "spatial_angle_modulation_sam2": {
            "standard": [
                ('amp', 0.7), ('carrierFreq', 440.0), ('modFreq', 4.0),
                ('peakPhaseDev', 0.55),
                ('phaseOffsetL', 0.0), ('phaseOffsetR', math.pi / 2),
                ('pathType', 'open'), ('discontinuousSteps', 8)
            ],
            "transition": [
                ('amp', 0.7),
                ('startCarrierFreq', 440.0), ('endCarrierFreq', 440.0),
                ('startModFreq', 4.0), ('endModFreq', 4.0),
                ('startPeakPhaseDev', 0.55), ('endPeakPhaseDev', 0.55),
                ('startPhaseOffsetL', 0.0), ('endPhaseOffsetL', 0.0),
                ('startPhaseOffsetR', math.pi / 2), ('endPhaseOffsetR', math.pi / 2),
                ('pathType', 'open'), ('discontinuousSteps', 8),
                ('initial_offset', 0.0), ('duration', 0.0), ('transition_curve', 'linear')
            ]
        },
        "binaural_beat": {
            "standard": [
                ('ampL', 0.5), ('ampR', 0.5),
                ('baseFreq', 200.0), ('beatFreq', 4.0), ('leftHigh', False),
                ('startPhaseL', 0.0), ('startPhaseR', 0.0),
                ('ampOscDepthL', 0.0), ('ampOscFreqL', 0.0), ('ampOscPhaseOffsetL', 0.0), ('ampOscDepthR', 0.0), ('ampOscFreqR', 0.0), ('ampOscPhaseOffsetR', 0.0), ('freqOscRangeL', 0.0), ('freqOscFreqL', 0.0), ('freqOscSkewL', 0.0), ('freqOscPhaseOffsetL', 0.0), ('freqOscRangeR', 0.0), ('freqOscFreqR', 0.0), ('freqOscSkewR', 0.0), ('freqOscPhaseOffsetR', 0.0), ('freqOscShape', 'sine'), ('phaseOscFreq', 0.0), ('phaseOscRange', 0.0),
                ('pan', 0.0), ('panRangeMin', 0.0), ('panRangeMax', 0.0), ('panType', 'linear'), ('panFreq', 0.0)
            ],
            "transition": [
                ('startAmpL', 0.5), ('endAmpL', 0.5), ('startAmpR', 0.5), ('endAmpR', 0.5),
                ('startBaseFreq', 200.0), ('endBaseFreq', 200.0), ('startBeatFreq', 4.0), ('endBeatFreq', 4.0), ('leftHigh', False),
                ('startStartPhaseL', 0.0), ('endStartPhaseL', 0.0), ('startStartPhaseR', 0.0), ('endStartPhaseR', 0.0),
                ('startAmpOscDepthL', 0.0), ('endAmpOscDepthL', 0.0), ('startAmpOscFreqL', 0.0), ('endAmpOscFreqL', 0.0),
                ('startAmpOscPhaseOffsetL', 0.0), ('endAmpOscPhaseOffsetL', 0.0), ('startAmpOscDepthR', 0.0), ('endAmpOscDepthR', 0.0),
                ('startAmpOscFreqR', 0.0), ('endAmpOscFreqR', 0.0), ('startAmpOscPhaseOffsetR', 0.0), ('endAmpOscPhaseOffsetR', 0.0),
                ('startFreqOscRangeL', 0.0), ('endFreqOscRangeL', 0.0), ('startFreqOscFreqL', 0.0), ('endFreqOscFreqL', 0.0),
                ('startFreqOscSkewL', 0.0), ('endFreqOscSkewL', 0.0), ('startFreqOscPhaseOffsetL', 0.0), ('endFreqOscPhaseOffsetL', 0.0),
                ('startFreqOscRangeR', 0.0), ('endFreqOscRangeR', 0.0), ('startFreqOscFreqR', 0.0), ('endFreqOscFreqR', 0.0),
                ('startFreqOscSkewR', 0.0), ('endFreqOscSkewR', 0.0), ('startFreqOscPhaseOffsetR', 0.0), ('endFreqOscPhaseOffsetR', 0.0),
                ('freqOscShape', 'sine'),
                ('startPhaseOscFreq', 0.0), ('endPhaseOscFreq', 0.0), ('startPhaseOscRange', 0.0), ('endPhaseOscRange', 0.0),
                ('startPan', 0.0), ('endPan', 0.0),
                ('startPanRangeMin', 0.0), ('endPanRangeMin', 0.0),
                ('startPanRangeMax', 0.0), ('endPanRangeMax', 0.0),
                ('startPanType', 'linear'), ('endPanType', 'linear'),
                ('startPanFreq', 0.0), ('endPanFreq', 0.0),
                ('initial_offset', 0.0), ('duration', 0.0), ('transition_curve', 'linear')
            ]
        },
        "monaural_beat_stereo_amps": { # This is an example, ensure it's correct
            "standard": [
                ('amp_lower_L', 0.5), ('amp_upper_L', 0.5),
                ('amp_lower_R', 0.5), ('amp_upper_R', 0.5),
                ('baseFreq', 200.0), ('beatFreq', 4.0),
                ('startPhaseL', 0.0), ('startPhaseR', 0.0),
                ('phaseOscFreq', 0.0), ('phaseOscRange', 0.0),
                ('ampOscDepth', 0.0), ('ampOscFreq', 0.0), ('ampOscPhaseOffset', 0.0)
            ],
            "transition": [
                ('start_amp_lower_L', 0.5), ('end_amp_lower_L', 0.5),
                ('start_amp_upper_L', 0.5), ('end_amp_upper_L', 0.5),
                ('start_amp_lower_R', 0.5), ('end_amp_lower_R', 0.5),
                ('start_amp_upper_R', 0.5), ('end_amp_upper_R', 0.5),
                ('start_baseFreq', 200.0), ('end_baseFreq', 200.0),
                ('start_beatFreq', 4.0), ('end_beatFreq', 4.0),
                ('start_startPhaseL', 0.0), ('end_startPhaseL', 0.0),
                ('start_startPhaseR', 0.0), ('end_startPhaseR', 0.0),
                ('start_phaseOscFreq', 0.0), ('end_phaseOscFreq', 0.0),
                ('start_phaseOscRange', 0.0), ('end_phaseOscRange', 0.0),
                ('start_ampOscDepth', 0.0), ('end_ampOscDepth', 0.0),
                ('start_ampOscFreq', 0.0), ('end_ampOscFreq', 0.0),
                ('start_ampOscPhaseOffset', 0.0), ('end_ampOscPhaseOffset', 0.0),
                ('initial_offset', 0.0), ('duration', 0.0), ('transition_curve', 'linear')
            ]
        },
        "isochronic_tone": {
            "standard": [
                ('ampL', 0.5), ('ampR', 0.5),
                ('baseFreq', 200.0), ('beatFreq', 4.0),
                ('forceMono', False),
                ('startPhaseL', 0.0), ('startPhaseR', 0.0),
                ('ampOscDepthL', 0.0), ('ampOscFreqL', 0.0),
                ('ampOscDepthR', 0.0), ('ampOscFreqR', 0.0),
                ('freqOscRangeL', 0.0), ('freqOscFreqL', 0.0),
                ('freqOscRangeR', 0.0), ('freqOscFreqR', 0.0),
                ('freqOscSkewL', 0.0), ('freqOscSkewR', 0.0),
                ('freqOscPhaseOffsetL', 0.0), ('freqOscPhaseOffsetR', 0.0),
                ('ampOscPhaseOffsetL', 0.0), ('ampOscPhaseOffsetR', 0.0),
                ('ampOscSkewL', 0.0), ('ampOscSkewR', 0.0),
                ('phaseOscFreq', 0.0), ('phaseOscRange', 0.0),
                ('rampPercent', 0.2), ('gapPercent', 0.15),
                ('harmonicSuppression', False),
                ('pan', 0.0), ('panRangeMin', 0.0), ('panRangeMax', 0.0),
                ('panType', 'linear'), ('panFreq', 0.0), ('panPhase', 0.0)
            ],
            "transition": [
                ('startAmpL', 0.5), ('endAmpL', 0.5),
                ('startAmpR', 0.5), ('endAmpR', 0.5),
                ('startBaseFreq', 200.0), ('endBaseFreq', 200.0),
                ('startBeatFreq', 4.0), ('endBeatFreq', 4.0),
                ('startForceMono', False), ('endForceMono', False),
                ('startStartPhaseL', 0.0), ('endStartPhaseL', 0.0),
                ('startStartPhaseR', 0.0), ('endStartPhaseR', 0.0),
                ('startAmpOscDepthL', 0.0), ('endAmpOscDepthL', 0.0),
                ('startAmpOscFreqL', 0.0), ('endAmpOscFreqL', 0.0),
                ('startAmpOscDepthR', 0.0), ('endAmpOscDepthR', 0.0),
                ('startAmpOscFreqR', 0.0), ('endAmpOscFreqR', 0.0),
                ('startFreqOscRangeL', 0.0), ('endFreqOscRangeL', 0.0),
                ('startFreqOscFreqL', 0.0), ('endFreqOscFreqL', 0.0),
                ('startFreqOscRangeR', 0.0), ('endFreqOscRangeR', 0.0),
                ('startFreqOscFreqR', 0.0), ('endFreqOscFreqR', 0.0),
                ('startFreqOscSkewL', 0.0), ('endFreqOscSkewL', 0.0),
                ('startFreqOscSkewR', 0.0), ('endFreqOscSkewR', 0.0),
                ('startFreqOscPhaseOffsetL', 0.0), ('endFreqOscPhaseOffsetL', 0.0),
                ('startFreqOscPhaseOffsetR', 0.0), ('endFreqOscPhaseOffsetR', 0.0),
                ('startAmpOscPhaseOffsetL', 0.0), ('endAmpOscPhaseOffsetL', 0.0),
                ('startAmpOscPhaseOffsetR', 0.0), ('endAmpOscPhaseOffsetR', 0.0),
                ('startAmpOscSkewL', 0.0), ('endAmpOscSkewL', 0.0),
                ('startAmpOscSkewR', 0.0), ('endAmpOscSkewR', 0.0),
                ('startPhaseOscFreq', 0.0), ('endPhaseOscFreq', 0.0),
                ('startPhaseOscRange', 0.0), ('endPhaseOscRange', 0.0),
                ('startRampPercent', 0.2), ('endRampPercent', 0.2),
                ('startGapPercent', 0.15), ('endGapPercent', 0.15),
                ('startHarmonicSuppression', False), ('endHarmonicSuppression', False),
                ('startPan', 0.0), ('endPan', 0.0),
                ('startPanRangeMin', 0.0), ('endPanRangeMin', 0.0),
                ('startPanRangeMax', 0.0), ('endPanRangeMax', 0.0),
                ('startPanType', 'linear'), ('endPanType', 'linear'),
                ('startPanFreq', 0.0), ('endPanFreq', 0.0),
                ('startPanPhase', 0.0), ('endPanPhase', 0.0),
                ('initial_offset', 0.0), ('duration', 0.0), ('transition_curve', 'linear')
            ]
        },
        "isochronic_tone": {
            "standard": [
                ('ampL', 0.5), ('ampR', 0.5),
                ('baseFreq', 200.0), ('beatFreq', 4.0),
                ('forceMono', False),
                ('startPhaseL', 0.0), ('startPhaseR', 0.0),
                ('ampOscDepthL', 0.0), ('ampOscFreqL', 0.0),
                ('ampOscDepthR', 0.0), ('ampOscFreqR', 0.0),
                ('freqOscRangeL', 0.0), ('freqOscFreqL', 0.0),
                ('freqOscRangeR', 0.0), ('freqOscFreqR', 0.0),
                ('freqOscSkewL', 0.0), ('freqOscSkewR', 0.0),
                ('freqOscPhaseOffsetL', 0.0), ('freqOscPhaseOffsetR', 0.0),
                ('ampOscPhaseOffsetL', 0.0), ('ampOscPhaseOffsetR', 0.0),
                ('ampOscSkewL', 0.0), ('ampOscSkewR', 0.0),
                ('phaseOscFreq', 0.0), ('phaseOscRange', 0.0),
                ('rampPercent', 0.2), ('gapPercent', 0.15),
                ('harmonicSuppression', False),
                ('pan', 0.0), ('panRangeMin', 0.0), ('panRangeMax', 0.0),
                ('panType', 'linear'), ('panFreq', 0.0), ('panPhase', 0.0)
            ],
            "transition": [
                ('startAmpL', 0.5), ('endAmpL', 0.5),
                ('startAmpR', 0.5), ('endAmpR', 0.5),
                ('startBaseFreq', 200.0), ('endBaseFreq', 200.0),
                ('startBeatFreq', 4.0), ('endBeatFreq', 4.0),
                ('startForceMono', False), ('endForceMono', False),
                ('startStartPhaseL', 0.0), ('endStartPhaseL', 0.0),
                ('startStartPhaseR', 0.0), ('endStartPhaseR', 0.0),
                ('startAmpOscDepthL', 0.0), ('endAmpOscDepthL', 0.0),
                ('startAmpOscFreqL', 0.0), ('endAmpOscFreqL', 0.0),
                ('startAmpOscDepthR', 0.0), ('endAmpOscDepthR', 0.0),
                ('startAmpOscFreqR', 0.0), ('endAmpOscFreqR', 0.0),
                ('startFreqOscRangeL', 0.0), ('endFreqOscRangeL', 0.0),
                ('startFreqOscFreqL', 0.0), ('endFreqOscFreqL', 0.0),
                ('startFreqOscRangeR', 0.0), ('endFreqOscRangeR', 0.0),
                ('startFreqOscFreqR', 0.0), ('endFreqOscFreqR', 0.0),
                ('startFreqOscSkewL', 0.0), ('endFreqOscSkewL', 0.0),
                ('startFreqOscSkewR', 0.0), ('endFreqOscSkewR', 0.0),
                ('startFreqOscPhaseOffsetL', 0.0), ('endFreqOscPhaseOffsetL', 0.0),
                ('startFreqOscPhaseOffsetR', 0.0), ('endFreqOscPhaseOffsetR', 0.0),
                ('startAmpOscPhaseOffsetL', 0.0), ('endAmpOscPhaseOffsetL', 0.0),
                ('startAmpOscPhaseOffsetR', 0.0), ('endAmpOscPhaseOffsetR', 0.0),
                ('startAmpOscSkewL', 0.0), ('endAmpOscSkewL', 0.0),
                ('startAmpOscSkewR', 0.0), ('endAmpOscSkewR', 0.0),
                ('startPhaseOscFreq', 0.0), ('endPhaseOscFreq', 0.0),
                ('startPhaseOscRange', 0.0), ('endPhaseOscRange', 0.0),
                ('startRampPercent', 0.2), ('endRampPercent', 0.2),
                ('startGapPercent', 0.15), ('endGapPercent', 0.15),
                ('startHarmonicSuppression', False), ('endHarmonicSuppression', False),
                ('startPan', 0.0), ('endPan', 0.0),
                ('startPanRangeMin', 0.0), ('endPanRangeMin', 0.0),
                ('startPanRangeMax', 0.0), ('endPanRangeMax', 0.0),
                ('startPanType', 'linear'), ('endPanType', 'linear'),
                ('startPanFreq', 0.0), ('endPanFreq', 0.0),
                ('startPanPhase', 0.0), ('endPanPhase', 0.0),
                ('initial_offset', 0.0), ('duration', 0.0), ('transition_curve', 'linear')
            ]
        },
        "dual_pulse_binaural": {
            "standard": [
                ('ampL', 0.5), ('ampR', 0.5),
                ('baseFreq', 200.0), ('beatFreq', 4.0), ('leftHigh', False),
                ('startPhaseL', 0.0), ('startPhaseR', 0.0),
                ('pulseRate', 0.0), ('pulseRateL', 0.0), ('pulseRateR', 0.0),
                ('rampPercent', 0.2), ('gapPercent', 0.15),
                ('rampPercentL', 0.2), ('rampPercentR', 0.2),
                ('gapPercentL', 0.15), ('gapPercentR', 0.15),
                ('pulsePhaseOffset', 0.0), ('pulsePhaseOffsetL', 0.0), ('pulsePhaseOffsetR', 0.0),
                ('pulseShape', 'trapezoid'), ('pulseShapeL', 'trapezoid'), ('pulseShapeR', 'trapezoid'),
                ('pulseDepth', 1.0), ('pulseDepthL', 1.0), ('pulseDepthR', 1.0),
                ('harmonicSuppression', False),
                ('ampOscDepthL', 0.0), ('ampOscFreqL', 0.0), ('ampOscPhaseOffsetL', 0.0),
                ('ampOscDepthR', 0.0), ('ampOscFreqR', 0.0), ('ampOscPhaseOffsetR', 0.0),
                ('freqOscRangeL', 0.0), ('freqOscFreqL', 0.0), ('freqOscSkewL', 0.0), ('freqOscPhaseOffsetL', 0.0),
                ('freqOscRangeR', 0.0), ('freqOscFreqR', 0.0), ('freqOscSkewR', 0.0), ('freqOscPhaseOffsetR', 0.0),
                ('freqOscShape', 'sine'),
                ('phaseOscFreq', 0.0), ('phaseOscRange', 0.0),
                ('pan', 0.0), ('panRangeMin', 0.0), ('panRangeMax', 0.0), ('panType', 'linear'), ('panFreq', 0.0)
            ],
            "transition": [
                ('startAmpL', 0.5), ('endAmpL', 0.5), ('startAmpR', 0.5), ('endAmpR', 0.5),
                ('startBaseFreq', 200.0), ('endBaseFreq', 200.0), ('startBeatFreq', 4.0), ('endBeatFreq', 4.0), ('leftHigh', False),
                ('startStartPhaseL', 0.0), ('endStartPhaseL', 0.0), ('startStartPhaseR', 0.0), ('endStartPhaseR', 0.0),
                ('startPulseRate', 0.0), ('endPulseRate', 0.0),
                ('startPulseRateL', 0.0), ('endPulseRateL', 0.0),
                ('startPulseRateR', 0.0), ('endPulseRateR', 0.0),
                ('startRampPercent', 0.2), ('endRampPercent', 0.2),
                ('startGapPercent', 0.15), ('endGapPercent', 0.15),
                ('startRampPercentL', 0.2), ('endRampPercentL', 0.2),
                ('startRampPercentR', 0.2), ('endRampPercentR', 0.2),
                ('startGapPercentL', 0.15), ('endGapPercentL', 0.15),
                ('startGapPercentR', 0.15), ('endGapPercentR', 0.15),
                ('startPulsePhaseOffset', 0.0), ('endPulsePhaseOffset', 0.0),
                ('startPulsePhaseOffsetL', 0.0), ('endPulsePhaseOffsetL', 0.0),
                ('startPulsePhaseOffsetR', 0.0), ('endPulsePhaseOffsetR', 0.0),
                ('startPulseShape', 'trapezoid'), ('endPulseShape', 'trapezoid'),
                ('startPulseShapeL', 'trapezoid'), ('endPulseShapeL', 'trapezoid'),
                ('startPulseShapeR', 'trapezoid'), ('endPulseShapeR', 'trapezoid'),
                ('startPulseDepth', 1.0), ('endPulseDepth', 1.0),
                ('startPulseDepthL', 1.0), ('endPulseDepthL', 1.0),
                ('startPulseDepthR', 1.0), ('endPulseDepthR', 1.0),
                ('startHarmonicSuppression', False), ('endHarmonicSuppression', False),
                ('startAmpOscDepthL', 0.0), ('endAmpOscDepthL', 0.0), ('startAmpOscFreqL', 0.0), ('endAmpOscFreqL', 0.0),
                ('startAmpOscPhaseOffsetL', 0.0), ('endAmpOscPhaseOffsetL', 0.0),
                ('startAmpOscDepthR', 0.0), ('endAmpOscDepthR', 0.0), ('startAmpOscFreqR', 0.0), ('endAmpOscFreqR', 0.0),
                ('startAmpOscPhaseOffsetR', 0.0), ('endAmpOscPhaseOffsetR', 0.0),
                ('startFreqOscRangeL', 0.0), ('endFreqOscRangeL', 0.0), ('startFreqOscFreqL', 0.0), ('endFreqOscFreqL', 0.0),
                ('startFreqOscSkewL', 0.0), ('endFreqOscSkewL', 0.0), ('startFreqOscPhaseOffsetL', 0.0), ('endFreqOscPhaseOffsetL', 0.0),
                ('startFreqOscRangeR', 0.0), ('endFreqOscRangeR', 0.0), ('startFreqOscFreqR', 0.0), ('endFreqOscFreqR', 0.0),
                ('startFreqOscSkewR', 0.0), ('endFreqOscSkewR', 0.0), ('startFreqOscPhaseOffsetR', 0.0), ('endFreqOscPhaseOffsetR', 0.0),
                ('freqOscShape', 'sine'),
                ('startPhaseOscFreq', 0.0), ('endPhaseOscFreq', 0.0), ('startPhaseOscRange', 0.0), ('endPhaseOscRange', 0.0),
                ('startPan', 0.0), ('endPan', 0.0), ('startPanRangeMin', 0.0), ('endPanRangeMin', 0.0),
                ('startPanRangeMax', 0.0), ('endPanRangeMax', 0.0), ('startPanType', 'linear'), ('endPanType', 'linear'),
                ('startPanFreq', 0.0), ('endPanFreq', 0.0),
                ('initial_offset', 0.0), ('duration', 0.0), ('transition_curve', 'linear')
            ]
        },
        "stereo_am_independent": {
            "standard": [
                ('amp', 0.25),
                ('carrierFreq', 200.0),
                ('modFreqL', 4.0), ('modDepthL', 0.8), ('modPhaseL', 0.0),
                ('modFreqR', 4.0), ('modDepthR', 0.8), ('modPhaseR', 0.0),
                ('stereo_width_hz', 0.2)
            ],
            "transition": [
                ('amp', 0.25),
                ('startCarrierFreq', 200.0), ('endCarrierFreq', 250.0),
                ('startModFreqL', 4.0), ('endModFreqL', 6.0),
                ('startModDepthL', 0.8), ('endModDepthL', 0.8),
                ('startModPhaseL', 0.0),
                ('startModFreqR', 4.1), ('endModFreqR', 5.9),
                ('startModDepthR', 0.8), ('endModDepthR', 0.8),
                ('startModPhaseR', 0.0),
                ('startStereoWidthHz', 0.2), ('endStereoWidthHz', 0.2),
                ('initial_offset', 0.0), ('duration', 0.0), ('transition_curve', 'linear')
            ]
        },
        "subliminal_encode": {
            "standard": [
                ('carrierFreq', 17500.0), ('amp', 0.5), ('mode', 'sequence'), ('audio_path', '')
            ],
            "transition": []
        },
        "spatial_angle_modulation": {
            "standard": [
                ('amp', 0.7), ('carrierFreq', 440.0), ('beatFreq', 4.0),
                ('pathShape', 'circle'), ('pathRadius', 1.0),
                ('arcStartDeg', 0.0), ('arcEndDeg', 360.0),
                ('frame_dur_ms', 46.4), ('overlap_factor', 8)
            ],
            "transition": [
                ('amp', 0.7),
                ('startCarrierFreq', 440.0), ('endCarrierFreq', 440.0),
                ('startBeatFreq', 4.0), ('endBeatFreq', 4.0),
                ('startPathShape', 'circle'), ('endPathShape', 'circle'),
                ('startPathRadius', 1.0), ('endPathRadius', 1.0),
                ('startArcStartDeg', 0.0), ('endArcStartDeg', 0.0),
                ('startArcEndDeg', 360.0), ('endArcEndDeg', 360.0),
                ('frame_dur_ms', 46.4), ('overlap_factor', 8),
                ('initial_offset', 0.0), ('duration', 0.0), ('transition_curve', 'linear')
            ]
        },
        "spatial_angle_modulation_sam2": {
            "standard": [
                ('amp', 0.7), ('carrierFreq', 440.0), ('modFreq', 4.0),
                ('peakPhaseDev', 0.55),
                ('phaseOffsetL', 0.0), ('phaseOffsetR', math.pi / 2),
                ('pathType', 'open'), ('discontinuousSteps', 8)
            ],
            "transition": [
                ('amp', 0.7),
                ('startCarrierFreq', 440.0), ('endCarrierFreq', 440.0),
                ('startModFreq', 4.0), ('endModFreq', 4.0),
                ('startPeakPhaseDev', 0.55), ('endPeakPhaseDev', 0.55),
                ('startPhaseOffsetL', 0.0), ('endPhaseOffsetL', 0.0),
                ('startPhaseOffsetR', math.pi / 2), ('endPhaseOffsetR', math.pi / 2),
                ('pathType', 'open'), ('discontinuousSteps', 8),
                ('initial_offset', 0.0), ('duration', 0.0), ('transition_curve', 'linear')
            ]
        },
        "wave_shape_stereo_am": {
            "standard": [
                ('amp', 0.15), ('carrierFreq', 200.0),
                ('shapeModFreq', 4.0), ('shapeModDepth', 0.8), ('shapeAmount', 0.5),
                ('stereoModFreqL', 4.1), ('stereoModDepthL', 0.8), ('stereoModPhaseL', 0.0),
                ('stereoModFreqR', 4.0), ('stereoModDepthR', 0.8), ('stereoModPhaseR', 1.5708),
            ],
            "transition": [
                ('amp', 0.15),
                ('startCarrierFreq', 200.0), ('endCarrierFreq', 100.0),
                ('startShapeModFreq', 4.0), ('endShapeModFreq', 8.0),
                ('startShapeModDepth', 0.8), ('endShapeModDepth', 0.8),
                ('startShapeAmount', 0.5), ('endShapeAmount', 0.5),
                ('startStereoModFreqL', 4.1), ('endStereoModFreqL', 6.0),
                ('startStereoModDepthL', 0.8), ('endStereoModDepthL', 0.8),
                ('startStereoModPhaseL', 0.0),
                ('startStereoModFreqR', 4.0), ('endStereoModFreqR', 6.1),
                ('startStereoModDepthR', 0.9), ('endStereoModDepthR', 0.9),
                ('startStereoModPhaseR', 1.5708),
                ('initial_offset', 0.0), ('duration', 0.0), ('transition_curve', 'linear')
            ]
        },
        "spatial_angle_modulation_monaural_beat": {
            "standard": [
                ('amp', 0.7), ('pathRadius', 1.0), ('frame_dur_ms', 46.4), ('overlap_factor', 8),
                ('spatialBeatFreq', 4.0), ('spatialPhaseOffset', 0.0), ('rotationDirection', 'cw'),
                ('sam_ampOscDepth', 0.0), ('sam_ampOscFreq', 0.0), ('sam_ampOscPhaseOffset', 0.0),
                ('amp_lower_L', 0.5), ('amp_upper_L', 0.5),
                ('amp_lower_R', 0.5), ('amp_upper_R', 0.5),
                ('baseFreq', 200.0), ('beatFreq', 4.0),
                ('startPhaseL', 0.0), ('startPhaseR', 0.0),
                ('phaseOscFreq', 0.0), ('phaseOscRange', 0.0),
                ('monaural_ampOscDepth', 0.0), ('monaural_ampOscFreq', 0.0), ('monaural_ampOscPhaseOffset', 0.0)
            ],
            "transition": [
                ('startAmp', 0.7), ('endAmp', 0.7),
                ('startPathRadius', 1.0), ('endPathRadius', 1.0),
                ('frame_dur_ms', 46.4), ('overlap_factor', 8),
                ('startSpatialBeatFreq', 4.0), ('endSpatialBeatFreq', 4.0),
                ('startSpatialPhaseOffset', 0.0), ('endSpatialPhaseOffset', 0.0),
                ('rotationDirection', 'cw'),
                ('start_sam_ampOscDepth', 0.0), ('end_sam_ampOscDepth', 0.0),
                ('start_sam_ampOscFreq', 0.0), ('end_sam_ampOscFreq', 0.0),
                ('start_sam_ampOscPhaseOffset', 0.0), ('end_sam_ampOscPhaseOffset', 0.0),
                ('start_amp_lower_L', 0.5), ('end_amp_lower_L', 0.5),
                ('start_amp_upper_L', 0.5), ('end_amp_upper_L', 0.5),
                ('start_amp_lower_R', 0.5), ('end_amp_lower_R', 0.5),
                ('start_amp_upper_R', 0.5), ('end_amp_upper_R', 0.5),
                ('startBaseFreq', 200.0), ('endBaseFreq', 200.0),
                ('startBeatFreq', 4.0), ('endBeatFreq', 4.0),
                ('startStartPhaseL_monaural', 0.0), ('endStartPhaseL_monaural', 0.0),
                ('startStartPhaseU_monaural', 0.0), ('endStartPhaseU_monaural', 0.0),
                ('startPhaseOscFreq_monaural', 0.0), ('endPhaseOscFreq_monaural', 0.0),
                ('startPhaseOscRange_monaural', 0.0), ('endPhaseOscRange_monaural', 0.0),
                ('startAmpOscDepth_monaural', 0.0), ('endAmpOscDepth_monaural', 0.0),
                ('startAmpOscFreq_monaural', 0.0), ('endAmpOscFreq_monaural', 0.0),
                ('startAmpOscPhaseOffset_monaural', 0.0), ('endAmpOscPhaseOffset_monaural', 0.0),
                ('initial_offset', 0.0), ('duration', 0.0), ('transition_curve', 'linear')
            ]
        },
        "qam_beat": { # CORRECTED AND COMPLETED for qam_beat based on qam_beat.py
            "standard": [
                ('ampL', 0.5), ('ampR', 0.5),
                ('baseFreqL', 200.0), ('baseFreqR', 204.0),
                ('qamAmFreqL', 4.0), ('qamAmDepthL', 0.5), ('qamAmPhaseOffsetL', 0.0),
                ('qamAmFreqR', 4.0), ('qamAmDepthR', 0.5), ('qamAmPhaseOffsetR', 0.0),
                ('qamAm2FreqL', 0.0), ('qamAm2DepthL', 0.0), ('qamAm2PhaseOffsetL', 0.0),
                ('qamAm2FreqR', 0.0), ('qamAm2DepthR', 0.0), ('qamAm2PhaseOffsetR', 0.0),
                ('modShapeL', 1.0), ('modShapeR', 1.0),
                ('crossModDepth', 0.0), ('crossModDelay', 0.0),
                ('harmonicDepth', 0.0), ('harmonicRatio', 2.0),
                ('subHarmonicFreq', 0.0), ('subHarmonicDepth', 0.0),
                ('startPhaseL', 0.0), ('startPhaseR', 0.0),
                ('phaseOscFreq', 0.0), ('phaseOscRange', 0.0), ('phaseOscPhaseOffset', 0.0),
                ('beatingSidebands', False), ('sidebandOffset', 1.0), ('sidebandDepth', 0.1),
                ('attackTime', 0.0), ('releaseTime', 0.0)
            ],
            "transition": [
                ('startAmpL', 0.5), ('endAmpL', 0.5),
                ('startAmpR', 0.5), ('endAmpR', 0.5),
                ('startBaseFreqL', 200.0), ('endBaseFreqL', 200.0),
                ('startBaseFreqR', 204.0), ('endBaseFreqR', 204.0),
                ('startQamAmFreqL', 4.0), ('endQamAmFreqL', 4.0),
                ('startQamAmDepthL', 0.5), ('endQamAmDepthL', 0.5),
                ('startQamAmPhaseOffsetL', 0.0), ('endQamAmPhaseOffsetL', 0.0),
                ('startQamAmFreqR', 4.0), ('endQamAmFreqR', 4.0),
                ('startQamAmDepthR', 0.5), ('endQamAmDepthR', 0.5),
                ('startQamAmPhaseOffsetR', 0.0), ('endQamAmPhaseOffsetR', 0.0),
                ('startQamAm2FreqL', 0.0), ('endQamAm2FreqL', 0.0),
                ('startQamAm2DepthL', 0.0), ('endQamAm2DepthL', 0.0),
                ('startQamAm2PhaseOffsetL', 0.0), ('endQamAm2PhaseOffsetL', 0.0),
                ('startQamAm2FreqR', 0.0), ('endQamAm2FreqR', 0.0),
                ('startQamAm2DepthR', 0.0), ('endQamAm2DepthR', 0.0),
                ('startQamAm2PhaseOffsetR', 0.0), ('endQamAm2PhaseOffsetR', 0.0),
                ('startModShapeL', 1.0), ('endModShapeL', 1.0),
                ('startModShapeR', 1.0), ('endModShapeR', 1.0),
                ('startCrossModDepth', 0.0), ('endCrossModDepth', 0.0),
                ('startHarmonicDepth', 0.0), ('endHarmonicDepth', 0.0),
                ('startSubHarmonicFreq', 0.0), ('endSubHarmonicFreq', 0.0),
                ('startSubHarmonicDepth', 0.0), ('endSubHarmonicDepth', 0.0),
                ('startStartPhaseL', 0.0), ('endStartPhaseL', 0.0), # Corresponds to 'startPhaseL' in qam_beat
                ('startStartPhaseR', 0.0), ('endStartPhaseR', 0.0), # Corresponds to 'startPhaseR' in qam_beat
                ('startPhaseOscFreq', 0.0), ('endPhaseOscFreq', 0.0),
                ('startPhaseOscRange', 0.0), ('endPhaseOscRange', 0.0),
                # Static parameters for transition mode (values are fixed, not interpolated)
                ('crossModDelay', 0.0),
                ('harmonicRatio', 2.0),
                ('phaseOscPhaseOffset', 0.0),
                ('beatingSidebands', False),
                ('sidebandOffset', 1.0),
                ('sidebandDepth', 0.1),
                ('attackTime', 0.0),
                ('releaseTime', 0.0),
                ('initial_offset', 0.0), ('duration', 0.0), ('transition_curve', 'linear')
            ]
        },
        "hybrid_qam_monaural_beat": { # This is an example, ensure it's correct
            "standard": [
                ('ampL', 0.5), ('ampR', 0.5),
                ('qamCarrierFreqL', 100.0), ('qamAmFreqL', 4.0), ('qamAmDepthL', 0.5),
                ('qamAmPhaseOffsetL', 0.0), ('qamStartPhaseL', 0.0),
                ('monoCarrierFreqR', 100.0), ('monoBeatFreqInChannelR', 4.0),
                ('monoAmDepthR', 0.0), ('monoAmFreqR', 0.0), ('monoAmPhaseOffsetR', 0.0),
                ('monoFmRangeR', 0.0), ('monoFmFreqR', 0.0), ('monoFmPhaseOffsetR', 0.0),
                ('monoStartPhaseR_Tone1', 0.0), ('monoStartPhaseR_Tone2', 0.0),
                ('monoPhaseOscFreqR', 0.0), ('monoPhaseOscRangeR', 0.0), ('monoPhaseOscPhaseOffsetR', 0.0)
            ],
            "transition": [
                ('startAmpL', 0.5), ('endAmpL', 0.5),
                ('startAmpR', 0.5), ('endAmpR', 0.5),
                ('startQamCarrierFreqL', 100.0), ('endQamCarrierFreqL', 100.0),
                ('startQamAmFreqL', 4.0), ('endQamAmFreqL', 4.0),
                ('startQamAmDepthL', 0.5), ('endQamAmDepthL', 0.5),
                ('startQamAmPhaseOffsetL', 0.0), ('endQamAmPhaseOffsetL', 0.0),
                ('startQamStartPhaseL', 0.0), ('endQamStartPhaseL', 0.0),
                ('startMonoCarrierFreqR', 100.0), ('endMonoCarrierFreqR', 100.0),
                ('startMonoBeatFreqInChannelR', 4.0), ('endMonoBeatFreqInChannelR', 4.0),
                ('startMonoAmDepthR', 0.0), ('endMonoAmDepthR', 0.0),
                ('startMonoAmFreqR', 0.0), ('endMonoAmFreqR', 0.0),
                ('startMonoAmPhaseOffsetR', 0.0), ('endMonoAmPhaseOffsetR', 0.0),
                ('startMonoFmRangeR', 0.0), ('endMonoFmRangeR', 0.0),
                ('startMonoFmFreqR', 0.0), ('endMonoFmFreqR', 0.0),
                ('startMonoFmPhaseOffsetR', 0.0), ('endMonoFmPhaseOffsetR', 0.0),
                ('startMonoStartPhaseR_Tone1', 0.0), ('endMonoStartPhaseR_Tone1', 0.0),
                ('startMonoStartPhaseR_Tone2', 0.0), ('endMonoStartPhaseR_Tone2', 0.0),
                ('startMonoPhaseOscFreqR', 0.0), ('endMonoPhaseOscFreqR', 0.0),
                ('startMonoPhaseOscRangeR', 0.0), ('endMonoPhaseOscRangeR', 0.0),
                ('startMonoPhaseOscPhaseOffsetR', 0.0), ('endMonoPhaseOscPhaseOffsetR', 0.0),
                ('initial_offset', 0.0), ('duration', 0.0), ('transition_curve', 'linear')
            ]
        }
    }
    # --- End of param_definitions ---

    base_func_key = internal_func_key_map.get(func_name_from_combo, func_name_from_combo)

    # Fallback: if key not found, try snake_case conversion using known definitions
    if base_func_key not in param_definitions:
        potential_key = base_func_key.lower().replace(" ", "_")
        if potential_key in param_definitions:
            base_func_key = potential_key

    ordered_params = OrderedDict()
    definition_set = param_definitions.get(base_func_key)

    # If func_name_from_combo itself is a transition func (e.g. "binaural_beat_transition")
    # then is_transition_mode might be overridden by this fact.
    # The self.transition_check usually dictates the mode.
    effective_func_name_for_lookup = base_func_key # Prefer resolved key for lookups
    effective_is_transition = is_transition_mode

    # If the selected combo item ALREADY implies transition (e.g., "func_transition")
    # then we should look for its base definition and use "transition" params.
    if effective_func_name_for_lookup.endswith("_transition"):
        potential_base_key = effective_func_name_for_lookup[:-len("_transition")]
        if potential_base_key in param_definitions:
            definition_set = param_definitions.get(potential_base_key)
            effective_is_transition = True # Force transition mode if function name implies it
        else: # No base key found, try the full name if it exists
            definition_set = param_definitions.get(effective_func_name_for_lookup)
    else: # Not a name ending with _transition
        definition_set = param_definitions.get(effective_func_name_for_lookup)


    if not definition_set:
        print(f"Warning: No parameter definitions found for function '{effective_func_name_for_lookup}' (derived from combo '{func_name_from_combo}').")
        # Try to get params by direct introspection as a fallback
        ordered_params = _introspect_params(effective_func_name_for_lookup)
        return ordered_params

    # Choose param set based on transition mode
    selected_mode_key = "transition" if effective_is_transition else "standard"
    if selected_mode_key not in definition_set:
        # Fallback: If transition requested but not available, use standard
        if selected_mode_key == "transition" and "transition" not in definition_set:
            print(f"Warning: No 'transition' parameters for '{base_func_key}'. Using 'standard' parameters instead even if transition is checked.")
            selected_mode_key = "standard"
        else:
            print(f"Warning: No parameters found for '{base_func_key}' in mode '{selected_mode_key}'.")
            ordered_params = _introspect_params(effective_func_name_for_lookup)
            return ordered_params

    # Create OrderedDict for selected mode
    for name, default_val in definition_set[selected_mode_key]:
        ordered_params[name] = default_val

    # Introspect the synth function to pick up any parameters that were added
    # after the static definitions were last updated.  This keeps the UI in sync
    # with the actual synth function signatures while still relying on the
    # curated defaults above for functions (like ``isochronic_tone``) that use
    # ``**params``.
    introspected_params = _introspect_params(effective_func_name_for_lookup)

    if not ordered_params:
        ordered_params = introspected_params
    else:
        for name, default_val in introspected_params.items():
            if name not in ordered_params:
                ordered_params[name] = default_val

    return ordered_params
