"""Local dimensionality reduction and clustering of frame embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .types import InferenceRunResult


@dataclass
class ClusteringResult:
    pca_scores: np.ndarray
    explained_variance_ratio: np.ndarray
    cluster_labels: np.ndarray
    cluster_centers_pca: np.ndarray
    representative_positions: np.ndarray
    parameters: dict[str, Any]

    def to_dict(self, frame_indices: np.ndarray | None = None,
                timestamps: np.ndarray | None = None) -> dict[str, Any]:
        representatives = self.representative_positions
        if frame_indices is not None:
            representatives = frame_indices[representatives]
        result = {
            "pca_scores": self.pca_scores.tolist(),
            "explained_variance_ratio": self.explained_variance_ratio.tolist(),
            "cluster_labels": self.cluster_labels.tolist(),
            "cluster_centers_pca": self.cluster_centers_pca.tolist(),
            "representative_frames": representatives.tolist(),
            "parameters": dict(self.parameters),
        }
        if frame_indices is not None:
            result["frame_indices"] = frame_indices.tolist()
        if timestamps is not None:
            result["timestamps"] = timestamps.tolist()
        return result


def cluster_embeddings(
    result: InferenceRunResult,
    *,
    pca_components: int | float = 10,
    n_clusters: int | str = 3,
    max_clusters: int = 10,
    random_state: int = 0,
    normalize: bool = True,
    whiten: bool = False,
) -> ClusteringResult:
    if result.embeddings is None:
        raise ValueError("This inference result does not contain embeddings")
    x = np.asarray(result.embeddings, dtype=np.float64)
    n_samples, n_features = x.shape
    if n_samples < 2:
        raise ValueError("PCA and clustering require at least two frames")

    if isinstance(pca_components, (bool, np.bool_)):
        raise ValueError("pca_components must be an integer or a fraction between 0 and 1")
    try:
        requested_pca = float(pca_components)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "pca_components must be an integer or a fraction between 0 and 1"
        ) from exc
    if not np.isfinite(requested_pca):
        raise ValueError("pca_components must be finite")
    if 0.0 < requested_pca < 1.0:
        pca_argument: int | float = requested_pca
        pca_selection = "explained_variance"
        pca_requested: int | float = requested_pca
    elif requested_pca >= 1.0 and requested_pca.is_integer():
        pca_argument = min(int(requested_pca), n_samples, n_features)
        pca_selection = "components"
        pca_requested = int(requested_pca)
    else:
        raise ValueError(
            "pca_components must be a positive integer or a fraction between 0 and 1"
        )

    auto_clusters = isinstance(n_clusters, str)
    if auto_clusters:
        if n_clusters.strip().lower() != "auto":
            raise ValueError("n_clusters must be an integer or 'auto'")
        try:
            requested_max_clusters = int(max_clusters)
        except (TypeError, ValueError) as exc:
            raise ValueError("max_clusters must be an integer of at least 1") from exc
        if requested_max_clusters < 1:
            raise ValueError("max_clusters must be at least 1")
        evaluated_max_clusters = min(requested_max_clusters, n_samples - 1)
        requested_clusters: int | str = "auto"
    else:
        if isinstance(n_clusters, (bool, np.bool_)):
            raise ValueError("n_clusters must be an integer or 'auto'")
        try:
            cluster_value = float(n_clusters)
        except (TypeError, ValueError) as exc:
            raise ValueError("n_clusters must be an integer or 'auto'") from exc
        if not np.isfinite(cluster_value) or not cluster_value.is_integer():
            raise ValueError("n_clusters must be an integer or 'auto'")
        selected_clusters = int(cluster_value)
        if not 2 <= selected_clusters <= n_samples:
            raise ValueError(f"n_clusters must be between 2 and {n_samples}")
        requested_clusters = selected_clusters
        requested_max_clusters = None
        evaluated_max_clusters = None

    if normalize:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / np.maximum(norms, 1e-12)

    try:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.metrics import silhouette_score
    except ImportError as exc:
        raise RuntimeError(
            "PCA/K-means requires the optional AI dependencies; run "
            "`uv sync --extra ai`"
        ) from exc

    pca = PCA(
        n_components=pca_argument,
        whiten=bool(whiten),
        svd_solver="full" if pca_selection == "explained_variance" else "auto",
        random_state=int(random_state),
    )
    scores = pca.fit_transform(x)

    def fit_kmeans(k: int):
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            random_state=int(random_state),
        )
        return model, model.fit_predict(scores)

    silhouette_scores: dict[str, float] = {}
    selected_silhouette: float | None = None
    if auto_clusters:
        # Silhouette is undefined for K=1. Treat one cluster as a zero-score
        # no-separation baseline, then replace it only with a valid positive
        # silhouette candidate. This also gives two-frame datasets and an
        # explicit Maximum K of 1 a useful deterministic result.
        selected_clusters = 1
        selection_score = 0.0
        kmeans, labels = fit_kmeans(1)
        for candidate in range(2, evaluated_max_clusters + 1):
            candidate_model, candidate_labels = fit_kmeans(candidate)
            n_labels = len(np.unique(candidate_labels))
            if not 2 <= n_labels < n_samples:
                continue
            score = float(silhouette_score(scores, candidate_labels))
            if not np.isfinite(score):
                continue
            silhouette_scores[str(candidate)] = score
            if score > selection_score:
                selected_clusters = candidate
                selected_silhouette = score
                selection_score = score
                kmeans = candidate_model
                labels = candidate_labels
    else:
        kmeans, labels = fit_kmeans(selected_clusters)
        n_labels = len(np.unique(labels))
        if 2 <= n_labels < n_samples:
            selected_silhouette = float(silhouette_score(scores, labels))

    representative_positions = []
    for cluster_id, center in enumerate(kmeans.cluster_centers_):
        members = np.flatnonzero(labels == cluster_id)
        distances = np.linalg.norm(scores[members] - center, axis=1)
        representative_positions.append(int(members[int(np.argmin(distances))]))

    return ClusteringResult(
        pca_scores=scores.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        cluster_labels=labels.astype(np.int32),
        cluster_centers_pca=kmeans.cluster_centers_.astype(np.float32),
        representative_positions=np.asarray(representative_positions, dtype=np.int64),
        parameters={
            "pca_components": int(scores.shape[1]),
            "pca_components_requested": pca_requested,
            "pca_selection": pca_selection,
            "explained_variance_total": float(
                np.nansum(pca.explained_variance_ratio_)
            ),
            "n_clusters": selected_clusters,
            "n_clusters_requested": requested_clusters,
            "cluster_selection": "silhouette" if auto_clusters else "fixed",
            "silhouette_score": selected_silhouette,
            "silhouette_scores": silhouette_scores,
            "single_cluster_baseline": 0.0 if auto_clusters else None,
            "max_clusters_requested": (
                requested_max_clusters if auto_clusters else None
            ),
            "max_clusters_evaluated": evaluated_max_clusters,
            "random_state": int(random_state),
            "normalize": bool(normalize),
            "whiten": bool(whiten),
            "inertia": float(kmeans.inertia_),
        },
    )
