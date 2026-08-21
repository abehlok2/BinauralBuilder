"""Explicit-SOFA HRTF ingestion and interpolation.

This package is deliberately optional: importing the SAM core does not import
``sofar``, ``pyfar`` or ``h5py``.  Those dependencies are resolved only when a
SOFA asset is opened, so abstract and geometric renderers remain available in
minimal installations.
"""

from .cache import HRTFCache, default_hrtf_cache
from .coordinates import cartesian_to_sofa_spherical, sofa_positions_to_cartesian
from .interpolation import interpolate_log_magnitude_delay, nearest_indices
from .sofa_io import DelayPolicy, HRTFDataset, SofaDependencyError, load_sofa, resolve_sofa_path
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
    real_spherical_harmonics,
)
from .selection import DirectionIndex, DirectionWeights
from .storage import cache_root, data_root, datasets_root, hrtf_root, ratings_path, storage_report
from .subject_test import (
    AUDITION_AZIMUTHS_DEG,
    AUDITION_ELEVATIONS_DEG,
    RATING_CRITERIA,
    SIGNAL_TYPES,
    RatingStore,
    SubjectRating,
    SubjectScore,
    audition_directions,
    make_test_signal,
    rank_subjects,
)
from .coordinates import angular_distance_deg, canonical_to_spherical_deg, unit_vectors
from .validation import HRTFValidationIssue, validate_dataset

__all__ = [
    "AUDITION_AZIMUTHS_DEG",
    "AUDITION_ELEVATIONS_DEG",
    "CORRECTION_MODES",
    "DELAY_MAGNITUDE",
    "DelayPolicy",
    "HD650",
    "HRTFCache",
    "HRTFDataset",
    "HRTFValidationIssue",
    "HeadphoneCorrection",
    "HrirSelection",
    "HrtfInterpolator",
    "INTERPOLATION_MODES",
    "KEMAR_VARIANTS",
    "NEAREST",
    "OFF",
    "PROCESSING_VARIANTS",
    "RATING_CRITERIA",
    "RatingStore",
    "SIGNAL_TYPES",
    "SONICOM",
    "SONICOM_HD650",
    "SONICOM_SAMPLE_RATES_HZ",
    "SPHERICAL_HARMONIC",
    "SPHERICAL_TRIANGULAR",
    "SofaAssetEntry",
    "SofaDependencyError",
    "SubjectRating",
    "SubjectScore",
    "THREE_NEIGHBOR",
    "USER_EQUALIZATION_CURVE",
    "USER_IMPULSE_RESPONSE",
    "align_pair",
    "apply_correction",
    "audition_directions",
    "broadband_delay_samples",
    "cartesian_to_sofa_spherical",
    "common_differential_delay",
    "default_hrtf_cache",
    "describe_subject",
    "discover_assets",
    "find_headphone_assets",
    "interpolate_log_magnitude_delay",
    "is_kemar_subject",
    "load_sofa",
    "make_test_signal",
    "minimum_phase",
    "minimum_phase_from_log_magnitude",
    "nearest_indices",
    "parse_asset_name",
    "rank_subjects",
    "real_spherical_harmonics",
    "resolve_sofa_path",
    "shift_response",
    "sofa_positions_to_cartesian",
    "validate_dataset",
]
