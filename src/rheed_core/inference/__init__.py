"""Framework-neutral, local AI inference support for Auto RHEED.

The core package defines stable model/result contracts. Optional adapters own
framework-specific imports such as PyTorch, Transformers, or Ultralytics.
"""

from .analysis import ClusteringResult, cluster_embeddings
from .base import AdapterDescriptor, ModelAdapter, create_adapter, list_adapters
from .embedding_analysis import (
    EmbeddingAnalysisAdapter,
    EmbeddingAnalysisDescriptor,
    create_embedding_analysis_adapter,
    list_embedding_analysis_adapters,
    run_embedding_analysis,
    run_embedding_analysis_with_artifacts,
)
from .manifests import ModelManifestRegistry
from .runner import InferenceCancelled, run_inference
from .types import InferenceRunResult, ModelSpec, PreprocessingContext

__all__ = [
    "ClusteringResult",
    "AdapterDescriptor",
    "InferenceCancelled",
    "InferenceRunResult",
    "EmbeddingAnalysisAdapter",
    "EmbeddingAnalysisDescriptor",
    "ModelAdapter",
    "ModelManifestRegistry",
    "ModelSpec",
    "PreprocessingContext",
    "cluster_embeddings",
    "create_embedding_analysis_adapter",
    "create_adapter",
    "list_adapters",
    "list_embedding_analysis_adapters",
    "run_embedding_analysis",
    "run_embedding_analysis_with_artifacts",
    "run_inference",
]
