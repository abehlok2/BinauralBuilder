"""The HRTF subsystem: SOFA assets, direction lookup, and interpolation.

Nothing in here is imported when BinauralBuilder starts, and nothing bundles
HRTF data. A SOFA backend is needed only when a user actually selects an
asset, and its absence is reported as an actionable message rather than an
import failure - so every non-HRTF feature works on a machine with no HRTF
extras installed.

The pipeline, in the order the data moves:

.. code-block:: text

    sofa_io      read the file, hash it, keep what it declares
    validation   report what is in it, fatally or otherwise
    coordinates  convert its frame into the canonical one
    dataset      apply the delay policy, resample, freeze it
    cache        do all of that once
    selection    find the directions near a query
    decomposition  delay, log magnitude, minimum phase
    interpolation  the five modes, in the order they were built
    headphones   correction after the binaural stage, never before
    subject_test choosing a subject by listening, and keeping the scores
"""

from __future__ import annotations

from .cache import HrtfCache, cache_key_for, default_cache
from .coordinates import angular_distance_deg, positions_to_canonical, unit_vectors
from .dataset import (
    BAKE_DELAY,
    DELAY_POLICIES,
    PRESERVE_DELAY,
    HrtfAssetInfo,
    PreparedHrtf,
    prepare_dataset,
    prepare_from_path,
)
from .datasets import (
    HD650,
    KEMAR_VARIANTS,
    PROCESSING_VARIANTS,
    SONICOM,
    SONICOM_SAMPLE_RATES_HZ,
    SofaAssetEntry,
    describe_subject,
    discover_assets,
    find_headphone_assets,
    is_kemar_subject,
    parse_asset_name,
)
from .decomposition import (
    align_pair,
    broadband_delay_samples,
    common_differential_delay,
    common_differential_log_magnitude,
    minimum_phase,
    minimum_phase_from_log_magnitude,
    shift_response,
)
from .headphones import (
    CORRECTION_MODES,
    OFF,
    SONICOM_HD650,
    USER_EQUALIZATION_CURVE,
    USER_IMPULSE_RESPONSE,
    HeadphoneCorrection,
    apply_correction,
)
from .interpolation import (
    DELAY_MAGNITUDE,
    INTERPOLATION_MODES,
    NEAREST,
    SPHERICAL_HARMONIC,
    SPHERICAL_TRIANGULAR,
    THREE_NEIGHBOR,
    HrirSelection,
    HrtfInterpolator,
)
from .selection import DirectionIndex, DirectionWeights
from .sofa_io import (
    HrtfDependencyError,
    SofaFile,
    SofaLoadError,
    available_backend,
    file_sha256,
    load_sofa,
    missing_dependency_message,
)
from .storage import data_root, datasets_root, set_data_root, storage_report
from .subject_test import (
    AUDITION_AZIMUTHS_DEG,
    AUDITION_ELEVATIONS_DEG,
    RATING_CRITERIA,
    SIGNAL_TYPES,
    RatingStore,
    SubjectRating,
    audition_directions,
    make_test_signal,
    rank_subjects,
)
from .validation import HrtfReport, validate_sofa_file

__all__ = [
    "AUDITION_AZIMUTHS_DEG",
    "AUDITION_ELEVATIONS_DEG",
    "BAKE_DELAY",
    "CORRECTION_MODES",
    "DELAY_MAGNITUDE",
    "DELAY_POLICIES",
    "HD650",
    "INTERPOLATION_MODES",
    "KEMAR_VARIANTS",
    "NEAREST",
    "OFF",
    "PRESERVE_DELAY",
    "PROCESSING_VARIANTS",
    "RATING_CRITERIA",
    "SIGNAL_TYPES",
    "SONICOM",
    "SONICOM_HD650",
    "SONICOM_SAMPLE_RATES_HZ",
    "SPHERICAL_HARMONIC",
    "SPHERICAL_TRIANGULAR",
    "THREE_NEIGHBOR",
    "USER_EQUALIZATION_CURVE",
    "USER_IMPULSE_RESPONSE",
    "DirectionIndex",
    "DirectionWeights",
    "HeadphoneCorrection",
    "HrirSelection",
    "HrtfAssetInfo",
    "HrtfCache",
    "HrtfDependencyError",
    "HrtfInterpolator",
    "HrtfReport",
    "PreparedHrtf",
    "RatingStore",
    "SofaAssetEntry",
    "SofaFile",
    "SofaLoadError",
    "SubjectRating",
    "align_pair",
    "angular_distance_deg",
    "apply_correction",
    "audition_directions",
    "available_backend",
    "broadband_delay_samples",
    "cache_key_for",
    "common_differential_delay",
    "common_differential_log_magnitude",
    "data_root",
    "datasets_root",
    "default_cache",
    "describe_subject",
    "discover_assets",
    "file_sha256",
    "find_headphone_assets",
    "is_kemar_subject",
    "load_sofa",
    "make_test_signal",
    "minimum_phase",
    "minimum_phase_from_log_magnitude",
    "missing_dependency_message",
    "parse_asset_name",
    "positions_to_canonical",
    "prepare_dataset",
    "prepare_from_path",
    "rank_subjects",
    "set_data_root",
    "shift_response",
    "storage_report",
    "unit_vectors",
    "validate_sofa_file",
]
