"""Explicit-SOFA asset inspector for the SAM workbench."""
from __future__ import annotations
import copy
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (QComboBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QPushButton, QSpinBox, QDoubleSpinBox, QTextEdit, QVBoxLayout, QWidget)

class SamHrtfLab(QWidget):
    """Asset selection, policies, metadata, and text spherical coverage view."""
    paramsChanged = pyqtSignal(dict)
    def __init__(self, parent=None):
        super().__init__(parent); self._params = {}; self._loading = False
        layout = QVBoxLayout(self)
        form = QFormLayout(); layout.addLayout(form)
        self.renderer = QComboBox(); self.renderer.addItems(["abstract_pm", "geometric", "hrtf"]); form.addRow("Renderer mode", self.renderer)
        asset_row = QHBoxLayout(); self.asset = QLabel("No explicit SOFA asset selected"); self.asset.setWordWrap(True)
        browse = QPushButton("Select SOFA…"); browse.clicked.connect(self._browse); asset_row.addWidget(self.asset, 1); asset_row.addWidget(browse); form.addRow("HRTF asset", asset_row)
        self.interpolation = QComboBox(); self.interpolation.addItems(["nearest", "logmag_delay"]); form.addRow("Interpolation", self.interpolation)
        self.delay_policy = QComboBox(); self.delay_policy.addItems(["bake_delay_into_ir", "preserve_external_delay"]); form.addRow("Data.Delay policy", self.delay_policy)
        self.crossfade = QDoubleSpinBox(); self.crossfade.setRange(0, 100); self.crossfade.setValue(10); self.crossfade.setSuffix(" ms"); form.addRow("Filter crossfade", self.crossfade)
        self.control_interval = QSpinBox(); self.control_interval.setRange(1, 4096); self.control_interval.setValue(128); self.control_interval.setSuffix(" samples"); form.addRow("Direction interval", self.control_interval)
        self.report = QTextEdit(); self.report.setReadOnly(True); self.report.setPlaceholderText("Select an asset to see validation, metadata, delay status, and spherical coverage."); layout.addWidget(self.report, 1)
        compare = QHBoxLayout(); compare.addWidget(QLabel("Matched A/B renderer:"))
        self.compare_a = QComboBox(); self.compare_b = QComboBox()
        for widget, value in ((self.compare_a, "abstract_pm"), (self.compare_b, "hrtf")):
            widget.addItems(["abstract_pm", "geometric", "hrtf"]); widget.setCurrentText(value)
        use_a = QPushButton("Use A"); use_b = QPushButton("Use B")
        use_a.clicked.connect(lambda: self.renderer.setCurrentText(self.compare_a.currentText()))
        use_b.clicked.connect(lambda: self.renderer.setCurrentText(self.compare_b.currentText()))
        compare.addWidget(self.compare_a); compare.addWidget(use_a); compare.addWidget(self.compare_b); compare.addWidget(use_b)
        layout.addLayout(compare)
        for widget in (self.renderer, self.interpolation, self.delay_policy): widget.currentTextChanged.connect(self._changed)
        self.crossfade.valueChanged.connect(self._changed); self.control_interval.valueChanged.connect(self._changed)

    def set_params(self, params):
        self._loading = True; self._params = copy.deepcopy(dict(params))
        self.renderer.setCurrentText(str(params.get("rendererMode", "abstract_pm")))
        path = params.get("hrtfAsset"); self.asset.setText(str(path) if path else "No explicit SOFA asset selected")
        options = dict(params.get("hrtfOptions") or {})
        self.interpolation.setCurrentText(str(options.get("interpolation", "nearest")))
        self.delay_policy.setCurrentText(str(options.get("delayPolicy", "bake_delay_into_ir")))
        self.crossfade.setValue(float(options.get("crossfadeMs", 10))); self.control_interval.setValue(int(options.get("controlIntervalSamples", 128)))
        self._loading = False
        if path: self.inspect_asset()

    def params(self):
        result = {"rendererMode": self.renderer.currentText(), "hrtfAsset": self._params.get("hrtfAsset"),
            "hrtfOptions": {**dict(self._params.get("hrtfOptions") or {}), "schemaVersion": 1,
                "interpolation": self.interpolation.currentText(), "delayPolicy": self.delay_policy.currentText(),
                "crossfadeMs": self.crossfade.value(), "controlIntervalSamples": self.control_interval.value()}}
        if self._params.get("hrtfAssetHash"): result["hrtfAssetHash"] = self._params["hrtfAssetHash"]
        return result

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select explicit SOFA HRTF", "", "SOFA files (*.sofa);;All files (*)")
        if path:
            self._params["hrtfAsset"] = path; self.asset.setText(path); self.inspect_asset(); self._changed()

    def inspect_asset(self):
        path = self._params.get("hrtfAsset")
        if not path: return
        try:
            from src.audio.sam_workbench.hrtf import load_sofa, validate_dataset, cartesian_to_sofa_spherical
            data = load_sofa(path, delay_policy=self.delay_policy.currentText())
            self._params["hrtfAssetHash"] = data.content_hash
            spherical = cartesian_to_sofa_spherical(data.positions_m)
            issues = validate_dataset(data); report = data.metadata_report()
            lines = [f"{key}: {value}" for key, value in report.items()]
            lines += ["", "Spherical coverage:", f"azimuth {spherical[:,0].min():.1f}° … {spherical[:,0].max():.1f}°",
                f"elevation {spherical[:,1].min():.1f}° … {spherical[:,1].max():.1f}°",
                f"distance {spherical[:,2].min():.3f} … {spherical[:,2].max():.3f} m"]
            if issues: lines += ["", "Validation:"] + [f"[{x.severity}] {x.path}: {x.message}" for x in issues]
            self.report.setPlainText("\n".join(lines))
        except Exception as error:
            self.report.setPlainText(f"Asset unavailable or invalid:\n{error}\n\nNon-HRTF renderers remain available.")

    def _changed(self, *_):
        if not self._loading: self.paramsChanged.emit(self.params())
