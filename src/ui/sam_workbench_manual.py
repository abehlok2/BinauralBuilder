"""Searchable, in-application manual for the SAM/HRTF workbench."""

from __future__ import annotations

import html

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence, QTextDocument
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from src.audio.sam_workbench.parameters import SAM2_FIELDS

__all__ = ["SamWorkbenchManualDialog", "manual_html"]


def _parameter_rows() -> str:
    rows = []
    for field in SAM2_FIELDS:
        if field.kind in {"float", "int"}:
            limits = "unbounded"
            if field.minimum is not None or field.maximum is not None:
                limits = f"{field.minimum if field.minimum is not None else '−∞'} to {field.maximum if field.maximum is not None else '∞'}"
        elif field.choices:
            limits = ", ".join(field.choices)
        else:
            limits = "structured path data"
        transition = {
            "pair": "Start/end values on transition voices",
            "shared": "One value for the whole transition",
            "timing": "Transition voices only",
        }[field.transition]
        rows.append(
            "<tr>"
            f"<td><b>{html.escape(field.label)}</b><br><code>{field.name}</code></td>"
            f"<td>{html.escape(field.mode.title())}</td>"
            f"<td>{html.escape(str(field.default))}</td>"
            f"<td>{html.escape(limits)} {html.escape(field.unit)}</td>"
            f"<td>{html.escape(field.tooltip)}<br><i>{transition}; "
            f"{'automatable' if field.automatable else 'not automatable'}.</i></td>"
            "</tr>"
        )
    return "".join(rows)


def manual_html() -> str:
    """Return the manual as self-contained rich text.

    Parameter rows come from the same registry as the editor, preventing the
    manual from silently falling behind when a bound, default, or field changes.
    """

    return f"""
<html><head><style>
body {{ font-family: sans-serif; line-height: 1.35; }}
h1 {{ color: #3b78b4; }} h2 {{ color: #4f91cd; margin-top: 24px; }}
h3 {{ margin-top: 18px; }} code {{ background: #e8e8e8; color: #202020; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #888; padding: 6px; vertical-align: top; }}
th {{ background: #d9e7f4; color: #202020; }}
.note {{ border-left: 4px solid #d49b22; padding-left: 10px; }}
</style></head><body>
<h1 id="top">SAM/HRTF Workbench manual</h1>
<p>The workbench edits one Spatial Angle Modulation (SAM2) voice inside the
existing BinauralBuilder project. It works on a copy: <b>Apply</b> commits and
keeps the window open, <b>OK</b> commits and closes it, and <b>Cancel</b> discards
work since the last Apply.</p>
<p class="note"><b>Headphones and hearing safety:</b> begin at a low playback
level. Peak reporting and the limiter protect the digital file from clipping;
they do not measure sound pressure at your ears or make long listening sessions
safe. SAM describes acoustic channel relationships. The application makes no
medical, therapeutic, mental-state, or localization guarantee.</p>

<h2 id="contents">Contents</h2><ul>
<li><a href="#workflow">Recommended workflow</a></li>
<li><a href="#header">Renderer, disclosure, preview and saving</a></li>
<li><a href="#parameters">Parameters and every exposed field</a></li>
<li><a href="#path">Path &amp; Geometry</a></li>
<li><a href="#path3d">Three-dimensional paths</a></li>
<li><a href="#hrtf">HRTF Lab</a></li>
<li><a href="#scene">Scene: stages, modulation, routing and cost</a></li>
<li><a href="#analysis">Analysis and compatibility</a></li>
<li><a href="#shortcuts">Keyboard and troubleshooting</a></li>
</ul>

<h2 id="workflow">Recommended workflow</h2><ol>
<li>Choose a <b>Renderer</b>. Start with Abstract PM for an exact baseline or
Geometric for a simple physical head model. HRTF requires an explicit SOFA file.</li>
<li>Stay in <b>Basic</b> disclosure and set amplitude, carrier, motion rate,
motion depth and the path. Render a short preview at a comfortable level.</li>
<li>Open <b>Path &amp; Geometry</b> to inspect the coordinate convention and path
metrics. Use the visual designer for a custom trajectory.</li>
<li>For HRTF, use <b>HRTF Lab → Tests &amp; import</b> to compare candidates, then
select the winning asset on Asset. Audition several directions before editing cues.</li>
<li>Use Advanced or Expert only when the baseline works. Add stages, modulation,
buses, or bands one change at a time and compare previews.</li>
<li>Inspect Analysis for peak, waveform, spectrum, ITD and ILD. Resolve errors in
the status line and Compatibility report, then Apply or OK. Normal project export
uses the committed voice and canonical Python renderer.</li>
</ol>

<h2 id="header">Renderer, disclosure, preview and saving</h2>
<h3>Renderers</h3><ul>
<li><b>Abstract phase modulation:</b> sends opposed phase modulation to the ears.
It needs no HRTF and is the reproducible SAM reference.</li>
<li><b>Geometric binaural:</b> derives interaural delay and level from source
position and a geometric listener model.</li>
<li><b>HRTF (explicit SOFA):</b> convolves with the selected measured or simulated
head-related impulse response. Missing/invalid assets stop rendering rather than
silently substituting another algorithm.</li>
</ul>
<p><b>Disclosure</b> changes only which registered controls are visible: Basic is
the everyday set, Advanced adds path/ear/transition detail, and Expert exposes
ear polarity. Hidden and unknown values are preserved.</p>
<p><b>Start at</b> is the absolute project time sampled by automation. <b>Preview
length</b> controls the audition duration (0.1–60 s). <b>Render preview</b> runs in
a worker so the UI remains responsive; <b>Play</b> uses QtMultimedia and <b>Stop</b>
ends playback. A new render is required after edits. The status line reports
validation, peak, limiting, truncation, and unavailable audio output.</p>

<h2 id="parameters">Parameters</h2>
<p>Units, defaults, limits, automation policy, and transition behavior below are
read directly from the editor's parameter registry. A transition voice shows
start/end controls for paired fields. Reset restores that field's default.</p>
<table><tr><th>Control / saved key</th><th>Disclosure</th><th>Default</th>
<th>Range or choices</th><th>Meaning and behavior</th></tr>{_parameter_rows()}</table>
<h3>Path types and shapes</h3><p><b>Open</b> travels back and forth; <b>closed</b>
loops; <b>discontinuous</b> quantizes motion into the selected number of steps;
<b>custom</b> uses the stored path profile. Sinusoidal is smooth, triangle is
constant-slope, ramp/saw reset at an edge, and square alternates endpoints.
Discontinuous or resetting paths can sound abrupt; audition them carefully.</p>
<p><b>Ear polarity</b> is a compatibility choice. Legacy is left-minus/right-plus;
canonical exact SAM is left-plus/right-minus; same-polarity applies the positive
term to both ears. It is not an electrical polarity inversion.</p>

<h2 id="path">Path &amp; Geometry</h2>
<p>This tab separates path geometry from traversal. It summarizes the active
geometry, coordinate space, length, closure and speed variation. The canonical
listener frame is right-handed: +x forward, +y left, +z up; positive azimuth is
left and physical distance is metres.</p>
<p><b>Open visual path designer</b> edits a custom path. Choose a primitive or
mathematical path, move/add points, set open/closed traversal and smoothing, and
accept to return it to the voice. Legacy pixel paths retain their original data
but are converted through explicit centre, axis, scale, normalization and metres-
per-scene-unit metadata before physical rendering. Smoothing passes round corners;
the ratio controls how strongly each pass cuts them.</p>

<h2 id="path3d">Three-dimensional paths</h2>
<p><b>Open 3D path designer</b> edits the full path: where the source is in
height and depth as well as around you. A position is stored as
<code>[x, y, z]</code> metres in the listener frame, and the renderer derives
azimuth, elevation and distance from it. That is the only stored format -
azimuth/elevation/distance entry is converted as you type it, so there is never
a second version of the path to disagree with the first.</p>

<h3>Why four views</h3>
<p>Depth is ambiguous in a single perspective view: a point dragged across the
screen has moved along a ray, and which point on that ray it landed on is a
guess. So there are four. <b>Perspective</b> shows what the path looks like and
orbits when dragged. <b>Top</b> (forward/left), <b>Front</b> (left/height) and
<b>Side</b> (forward/height) each edit exactly the two axes they show and leave
the third alone. Selecting a point in any view selects it in all of them, in the
table, and in the numeric editor. The listener is drawn with a nose, ears, a
head-height plane and a vertical axis, and can show the HRTF coverage shell -
the sphere a dataset measures on. A path outside it is being extrapolated.</p>

<h3>Geometry and traversal are separate</h3>
<p><b>Geometry</b> is where the source can be. <b>Traversal</b> is how it moves
along that shape over time. A circle stays one circle whether it is walked at
constant speed, eased, reversed, or driven from the stage timeline, so those live
on their own tab and changing them never reshapes the path.</p>
<p>The speed law is worth choosing deliberately. <i>Constant linear speed</i>
advances by physical distance, so the source covers metres evenly. <i>Curve
parameter speed</i> advances along the curve's own parameter, which on a circle
is constant angular speed instead.</p>

<h3>Primitives</h3>
<p>Point kinds - polyline, spline, bezier and the rest - are edited by dragging.
The three-dimensional primitives are edited by their own numbers, so a dome
traversal stays a dome when the project is reopened rather than becoming a spline
through samples of one. They cover horizontal, vertical, tilted and spherical
orbits, rising and falling arcs, overhead sweeps, front-to-back elevation sweeps,
dome traversal, a three-dimensional figure-eight, pendulum, toroidal paths and a
seeded random walk in a volume. <b>Keyframes</b> holds timestamped positions and
is what a CSV or JSON import lands in.</p>

<h3>What each renderer does with it</h3>
<ul>
<li><b>HRTF</b> queries the dataset with azimuth <i>and</i> elevation at every
control interval, and uses distance for gain and propagation delay. This is the
mode that can actually place a source above or below you.</li>
<li><b>Geometric</b> derives interaural time and level difference, propagation
delay and distance attenuation from the path. Elevation colours the sound but
does not localize it - that needs HRTF filtering.</li>
<li><b>Abstract phase modulation</b> can take the path as a control source -
azimuth to interaural phase, elevation to carrier or modulation depth, distance
to amplitude. <span class="note">These are creative mappings, not
spatialization. Phase modulation alone cannot produce reliable height.</span></li>
<li><b>Hybrid</b> runs Source &#8594; SAM &#8594; 3D trajectory &#8594; HRTF
interpolation &#8594; Cue modification &#8594; Output. The order matters:
phase manipulation before binaural filtering and after it are different
renders.</li>
</ul>

<h3>Coverage warnings</h3>
<p>Many published HRTF datasets are dense around the horizontal plane and thin or
empty above and below it, so an overhead or below-head path can ask for
directions that were never measured. The workbench warns when the path leaves
measured coverage, repeatedly crosses sparse regions, needs a hemisphere the
dataset barely samples, falls back to nearest-neighbour selection, or turns
faster than the control interval and crossfade can track. None of these stops a
render; they say which parts of it are extrapolated rather than reproduced.</p>

<h2 id="hrtf">HRTF Lab</h2>
<h3>Asset</h3><p>Point the SONICOM library at a folder, then filter by dataset,
source (measured/KEMAR/synthetic), processing, sample rate and subject, or select
any SOFA explicitly. The coverage view and report validate metadata, receiver
order, directions, delays and sampling. Quality may be normal, suspected outlier,
or missing resources.</p>
<p><b>Interaural delay</b> declares whether delay is embedded in HRIR samples or
kept in SOFA Data.Delay. <b>Interpolation</b> selects nearest or an aligned blend.
Nearest preserves individual transients but can jump; aligned modes move more
smoothly. <b>Filter crossfade</b> fades direction changes (rendering clamps it to
5–20 ms), and <b>direction interval</b> is the sample spacing between updates.
Optional headphone correction is post-render and must match the headphones; it
is never selected automatically.</p>
<h3>Audition, Ranking, and Tests &amp; import</h3><p>Choose azimuth, elevation,
signal and clip length, render, then play. Rate localization, externalization and
other displayed criteria from 1–5 under a named profile. Ranking aggregates only
that profile. The blinded localization test scores angular and front/back errors;
the listening experiment compares reproducible conditions. Measurement/import is
advanced: measurement needs a calibrated rig, while Mesh2HRTF results are made
outside the app and validated on import.</p>
<h3>Modify</h3><p>Modify decomposes and scales ITD, ILD, pinna detail and distance.
Plots compare measured and derived cues. Reset to measured is neutral. Audition
before saving; a derived SOFA is written as a new asset with source provenance and
verification—the selected source is never overwritten. The spatial-anchor tools
can add broadband/harmonic information when a low-frequency carrier cannot carry
pinna cues. Treat every perceptual rating as listener-specific.</p>

<h2 id="scene">Scene</h2>
<h3>Stages</h3><p>Stages name one step or a group of consecutive project steps;
step timing remains authoritative. Select the number of steps per stage and a
crossfade, then Build. A selected stage exposes name, start, duration, transition
in/out, notes and parameter overrides. Each override has a target value and curve.</p>
<h3>Modulation matrix</h3><p>Rows are modulators and columns are target parameters.
Add a modulator, search the registry for a target, add its column, select a cell,
then set depth in the target's units, polarity, shaping curve, and enabled state.
Apply creates/updates the route; Clear removes it. Disabled routes remain stored.
Dependency cycles are refused and named.</p>
<h3>Routing &amp; cost</h3><p>Buses and sources expose gain in dB, mute and solo;
sources choose a destination bus. The master bus cannot be removed. Multiband
crossovers divide the signal, filter order controls steepness, and each band has
gain and enable. The cost estimator uses source/band counts, HRIR taps, block size,
interpolation and sample rate. Measure this machine benchmarks actual convolution.
A cost above the real-time budget suggests fewer sources/bands/taps or offline export.</p>

<h2 id="analysis">Analysis and compatibility</h2>
<p>After Render preview, Analysis displays a numeric summary, stereo waveform,
spectrum and binaural cue measurements. Peak is full-scale amplitude; ITD is ear
timing difference and ILD is ear level difference. These are acoustic measurements,
not proof of a perceived direction or biological outcome.</p>
<p>Compatibility reports renderer, sample rate, static/transition status, schema,
unknown preserved keys and every validation issue. Unversioned voices retain legacy
ear orientation. Explicit HRTF migration is opt-in and does not rewrite a legacy
slab/KEMAR preset. Errors disable Apply/OK; warnings describe risks without erasing data.</p>

<h2 id="shortcuts">Keyboard and troubleshooting</h2><ul>
<li><b>F1</b> opens this manual; Ctrl+F focuses its search. Enter finds next,
Shift+Enter finds previous, and Home returns to the contents.</li>
<li><b>Ctrl+1…6</b> selects workbench tabs; <b>Ctrl+R</b> renders, <b>Ctrl+P</b>
plays, and <b>Ctrl+.</b> stops.</li>
<li><b>Cannot apply:</b> read the named path in the status/Compatibility view.
Common causes are carrier above Nyquist, invalid transition timing, or missing SOFA.</li>
<li><b>No sound:</b> render first, confirm the status says Preview ready, and check
QtMultimedia/output-device availability. Analysis still works without an output device.</li>
<li><b>HRTF library empty:</b> confirm folder, source, processing and sample-rate
filters. Select SOFA directly to distinguish a naming/filter issue from a file issue.</li>
<li><b>Preview differs after a hidden edit:</b> select Expert and inspect
Compatibility. Hidden registered and extension keys are intentionally preserved.</li>
</ul><p><a href="#top">Back to top</a></p>
</body></html>"""


class SamWorkbenchManualDialog(QDialog):
    """Modeless-capable manual with anchor navigation and incremental search."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("SAM/HRTF Workbench Manual")
        self.resize(900, 720)
        self.setMinimumSize(620, 420)

        layout = QVBoxLayout(self)
        tools = QHBoxLayout()
        home = QPushButton("Contents")
        home.clicked.connect(lambda: self.browser.scrollToAnchor("contents"))
        tools.addWidget(home)
        tools.addWidget(QLabel("Find:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("parameter or feature")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.returnPressed.connect(self.find_next)
        tools.addWidget(self.search_edit, 1)
        previous = QPushButton("Previous")
        previous.clicked.connect(self.find_previous)
        tools.addWidget(previous)
        following = QPushButton("Next")
        following.clicked.connect(self.find_next)
        tools.addWidget(following)
        layout.addLayout(tools)

        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        self.browser.setHtml(manual_html())
        self.browser.setAccessibleName("SAM workbench manual contents")
        layout.addWidget(self.browser, 1)

        self.search_status = QLabel("")
        self.search_status.setAccessibleName("Manual search status")
        layout.addWidget(self.search_status)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

        self.search_edit.setAccessibleName("Find in manual")
        self.search_edit.setShortcutEnabled(True)

    def _find(self, backwards: bool = False) -> None:
        query = self.search_edit.text().strip()
        if not query:
            self.search_status.setText("Enter text to find.")
            return
        flags = QTextDocument.FindBackward if backwards else QTextDocument.FindFlags()
        if not self.browser.find(query, flags):
            cursor = self.browser.textCursor()
            cursor.movePosition(cursor.End if backwards else cursor.Start)
            self.browser.setTextCursor(cursor)
            if not self.browser.find(query, flags):
                self.search_status.setText(f'No match for "{query}".')
                return
        self.search_status.setText(f'Found "{query}".')

    def find_next(self) -> None:
        self._find(False)

    def find_previous(self) -> None:
        self._find(True)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.matches(QKeySequence.Find):
            self.search_edit.setFocus(Qt.ShortcutFocusReason)
            self.search_edit.selectAll()
            return
        super().keyPressEvent(event)
