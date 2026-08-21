"""
rheed_core — domain logic for RHEED data analysis.

This package has no dependency on Flask, MCP, or any other consumer.
It knows about RHEED physics (wavelength, camera constant, lattice
spacing), file formats (.h5, .npy, video, kSA .img), and the in-memory
representation of a loaded dataset (RheedSession). Both rheed_webapp
and rheed_mcp import from here rather than reimplementing this logic.
"""

from .session import RheedSession
from .analysis_store import AnalysisStore, StoredInferenceRun

__all__ = ["AnalysisStore", "RheedSession", "StoredInferenceRun"]
