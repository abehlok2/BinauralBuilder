"""Knowing what a SONICOM dataset looks like, without depending on having one.

The dataset is large, licensed, and lives outside the repository. Nothing here
downloads or bundles it. What this module provides is the vocabulary needed to
*talk* about it in the GUI - subjects, processing variants, sample rates, the
KEMAR collection - and a tolerant discovery pass over a folder the user points
at.

The filename parse is deliberately forgiving. Release layouts change, and a
user who has renamed a file should still be able to select it: an unparsed
name yields an entry with the fields it could determine and the rest blank,
never an error and never a guess presented as fact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

__all__ = [
    "HD650",
    "KEMAR_VARIANTS",
    "PROCESSING_VARIANTS",
    "SONICOM",
    "SONICOM_SAMPLE_RATES_HZ",
    "HeadphoneAsset",
    "SofaAssetEntry",
    "describe_subject",
    "discover_assets",
    "find_headphone_assets",
    "is_kemar_subject",
    "parse_asset_name",
]

SONICOM = "SONICOM"

#: Processing variants SONICOM publishes for each subject.
PROCESSING_VARIANTS: tuple[str, ...] = (
    "Windowed",
    "FreeFieldComp",
    "FreeFieldCompMinPhase",
)

#: Sample rates the dataset is published at.
SONICOM_SAMPLE_RATES_HZ: tuple[int, ...] = (44_100, 48_000, 96_000)

#: The KEMAR collection, which is what makes SONICOM useful as a *reference*
#: rather than only as a set of individual subjects. Pinna size, microphone
#: type, measurement point, and head-versus-torso rotation are separable here,
#: which is exactly what is needed to validate a coordinate convention or to
#: hear what a pinna difference does.
KEMAR_VARIANTS: dict[str, str] = {
    "KB0060": "KEMAR, small ears",
    "KB0061": "KEMAR, small ears (second configuration)",
    "KB0065": "KEMAR, large ears",
    "KB0066": "KEMAR, large ears (second configuration)",
}

#: SONICOM's own headphone transfer measurement.
HD650 = "Sennheiser HD650"

# `\b` is useless here: an underscore is a word character, so "P0001_Windowed"
# has no boundary after the identifier. Bound on alphanumerics instead.
_SUBJECT_PATTERN = re.compile(
    r"(?<![0-9A-Za-z])(P\d{3,4}|KB\d{3,4})(?![0-9A-Za-z])", re.IGNORECASE
)
_RATE_PATTERN = re.compile(r"(\d{2,3})\s*k(?:Hz)?", re.IGNORECASE)
_EXPLICIT_RATE_PATTERN = re.compile(r"(?<!\d)(44100|48000|96000)(?!\d)")


def is_kemar_subject(subject: str) -> bool:
    """True for the KEMAR identifiers, which are dummy heads rather than people."""

    return str(subject).upper() in KEMAR_VARIANTS


def describe_subject(subject: str) -> str:
    """A human description of a subject identifier."""

    name = str(subject).upper()
    if name in KEMAR_VARIANTS:
        return f"{name} - {KEMAR_VARIANTS[name]}"
    if name.startswith("P"):
        return f"{name} - human subject"
    return name or "unknown subject"


@dataclass(frozen=True)
class SofaAssetEntry:
    """One SOFA file found on disk, with whatever could be read from its name."""

    path: Path
    subject: str = ""
    processing: str = ""
    sample_rate_hz: int | None = None
    dataset: str = ""

    @property
    def is_kemar(self) -> bool:
        return is_kemar_subject(self.subject)

    @property
    def label(self) -> str:
        """What to show in a selector."""

        parts = [self.subject or self.path.stem]
        if self.processing:
            parts.append(self.processing)
        if self.sample_rate_hz:
            parts.append(f"{self.sample_rate_hz // 1000} kHz")
        return "  ·  ".join(parts)

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "subject": self.subject,
            "processing": self.processing,
            "sample_rate_hz": self.sample_rate_hz,
            "dataset": self.dataset,
        }


@dataclass(frozen=True)
class HeadphoneAsset:
    """A headphone transfer measurement, for the correction stage."""

    path: Path
    model: str = HD650
    dataset: str = SONICOM

    def to_dict(self) -> dict[str, object]:
        return {"path": str(self.path), "model": self.model, "dataset": self.dataset}


def parse_asset_name(path: str | Path) -> SofaAssetEntry:
    """Read what a filename says about a SOFA asset.

    Understands the SONICOM-style ``<subject>_<processing>_<rate>`` naming and
    degrades gracefully: fields it cannot determine are left empty rather than
    guessed.
    """

    location = Path(path)
    text = f"{location.parent.name}/{location.stem}"

    # The filename is the more specific identifier: when a file sits in a
    # folder named for a different subject, the file's own name wins.
    subject_match = _SUBJECT_PATTERN.search(location.stem) or _SUBJECT_PATTERN.search(
        location.parent.name
    )
    subject = subject_match.group(1).upper() if subject_match else ""

    processing = ""
    lowered = text.lower()
    # Longest first, so "FreeFieldCompMinPhase" is not read as "FreeFieldComp".
    for variant in sorted(PROCESSING_VARIANTS, key=len, reverse=True):
        if variant.lower() in lowered:
            processing = variant
            break

    sample_rate: int | None = None
    explicit = _EXPLICIT_RATE_PATTERN.search(text)
    if explicit:
        sample_rate = int(explicit.group(1))
    else:
        kilohertz = _RATE_PATTERN.search(text)
        if kilohertz:
            candidate = int(kilohertz.group(1)) * 1_000
            # "44kHz" means 44.1 kHz in this dataset's naming.
            sample_rate = 44_100 if candidate == 44_000 else candidate

    dataset = SONICOM if (subject or processing) else ""
    return SofaAssetEntry(
        path=location,
        subject=subject,
        processing=processing,
        sample_rate_hz=sample_rate,
        dataset=dataset,
    )


def discover_assets(
    root: str | Path, *, patterns: Iterable[str] = ("*.sofa",), limit: int | None = None
) -> list[SofaAssetEntry]:
    """Find SOFA assets under ``root``, sorted by subject then processing.

    Returns an empty list for a missing folder rather than raising: "the
    dataset is not installed" is a normal state the GUI reports, not an error.
    """

    location = Path(root).expanduser()
    if not location.is_dir():
        return []

    def _walk() -> Iterator[Path]:
        for pattern in patterns:
            yield from location.rglob(pattern)

    seen: set[Path] = set()
    entries: list[SofaAssetEntry] = []
    for candidate in sorted(_walk()):
        resolved = candidate.resolve()
        if resolved in seen or not candidate.is_file():
            continue
        seen.add(resolved)
        entries.append(parse_asset_name(candidate))
        if limit is not None and len(entries) >= limit:
            break

    entries.sort(key=lambda entry: (entry.subject or "~", entry.processing, entry.path.name))
    return entries


def find_headphone_assets(root: str | Path) -> list[HeadphoneAsset]:
    """Find headphone transfer measurements under ``root``.

    SONICOM publishes HD650 measurements; a file whose name says so is labelled
    accordingly, and anything else is offered as an unnamed measurement rather
    than assumed to be HD650.
    """

    location = Path(root).expanduser()
    if not location.is_dir():
        return []

    assets: list[HeadphoneAsset] = []
    for candidate in sorted(location.rglob("*.sofa")):
        name = candidate.stem.lower()
        if "hp" not in name and "headphone" not in name:
            continue
        model = HD650 if ("hd650" in name or "hd_650" in name) else ""
        assets.append(HeadphoneAsset(path=candidate, model=model))
    return assets
