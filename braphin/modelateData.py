"""
Deprecated module name. Import from ``braphin.connectivity`` instead.

This shim is retained for backward compatibility and will be removed in a
future major release.
"""
import warnings

warnings.warn(
    "braphin.modelateData is deprecated and will be removed in a future release. "
    "Use 'from braphin.connectivity import ModelBRAPHINConnectivityData, "
    "BRAPHINConnectivityBundle' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .connectivity import *  # noqa: F401, F403
from .connectivity import BRAPHINConnectivityBundle, ModelBRAPHINConnectivityData  # noqa: F401
