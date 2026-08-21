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
from .validation import HRTFValidationIssue, validate_dataset

__all__ = [
    "DelayPolicy", "HRTFCache", "HRTFDataset", "HRTFValidationIssue",
    "SofaDependencyError", "cartesian_to_sofa_spherical", "default_hrtf_cache",
    "interpolate_log_magnitude_delay", "load_sofa", "nearest_indices", "resolve_sofa_path",
    "sofa_positions_to_cartesian", "validate_dataset",
]
