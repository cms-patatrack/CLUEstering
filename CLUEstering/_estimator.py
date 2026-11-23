"""
scikit-learn compatible estimator for the CLUEstering algorithm.
"""

from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import validate_data

# The tests import CLUEstering.py as a top-level module through sys.path, so the relative
# import fails there. Both cases go away when the implementation moves to private modules.
try:
    from .CLUEstering import (_check_backend_available, _choose_default_backend,
                              _validate_backend, clue_kernels, cpu_omp, cpu_serial, cpu_tbb,
                              gpu_cuda, gpu_hip)
except ImportError:
    from CLUEstering import (_check_backend_available, _choose_default_backend,
                             _validate_backend, clue_kernels, cpu_omp, cpu_serial, cpu_tbb,
                             gpu_cuda, gpu_hip)

# keep in sync with the dimensions compiled in BindingModules/Run.hpp
_MAX_FEATURES = 4

_BACKEND_MODULES = {
    "cpu serial": lambda: cpu_serial,
    "cpu openmp": lambda: cpu_omp,
    "cpu tbb": lambda: cpu_tbb,
    "gpu cuda": lambda: gpu_cuda,
    "gpu hip": lambda: gpu_hip,
}

# name -> (factory, number of parameters, default parameters)
_KERNELS = {
    "flat": (clue_kernels.FlatKernel, 1, (0.5,)),
    "exponential": (clue_kernels.ExponentialKernel, 2, None),
    "gaussian": (clue_kernels.GaussianKernel, 2, None),
}

# name -> (factory, keyword of metric_params with the per-feature values, or None)
_METRICS = {
    "euclidean": (clue_kernels.EuclideanMetric, "weights"),
    "manhattan": (clue_kernels.ManhattanMetric, None),
    "chebyshev": (clue_kernels.ChebyshevMetric, "weights"),
    "periodic_euclidean": (clue_kernels.PeriodicEuclideanMetric, "periods"),
}


class CLUEstering(ClusterMixin, BaseEstimator):
    """
    Density-based weighted clustering with the CLUEstering algorithm.

    CLUEstering computes the local density of each point, links each point to its
    nearest point with higher density and promotes to seeds the points that are far
    from any denser point. Every other point follows its nearest higher until a seed,
    or is marked as an outlier.

    :param density_radius: Radius of the region used to compute the local density.
    :type density_radius: float
    :param min_density: Minimum density of a seed. Points with a lower density that are
        far from any denser point are outliers.
    :type min_density: float
    :param outlier_distance: Maximum distance for the nearest-higher search of the points
        below ``min_density``. If None, ``density_radius`` is used.
    :type outlier_distance: float or None
    :param seeding_distance: Minimum distance from any denser point for a point to become a
        seed. If None, ``density_radius`` is used.
    :type seeding_distance: float or None
    :param kernel: Convolutional kernel used for the local density: ``"flat"``,
        ``"exponential"`` or ``"gaussian"``.
    :type kernel: str
    :param kernel_params: Parameters of the kernel. If None, the default ones are used,
        which exist only for the flat kernel (``(0.5,)``).
    :type kernel_params: tuple or None
    :param metric: Distance metric: ``"euclidean"``, ``"manhattan"``, ``"chebyshev"`` or
        ``"periodic_euclidean"``.
    :type metric: str
    :param metric_params: Extra parameters of the metric: ``{"weights": [...]}`` for the
        euclidean and chebyshev metrics, ``{"periods": [...]}`` for the periodic euclidean
        one, with one value per feature.
    :type metric_params: dict or None
    :param wrapped: Flags marking the periodic features, one per feature. If None, no
        feature is periodic.
    :type wrapped: array-like of bool or None
    :param backend: Backend used for the computation, e.g. ``"cpu serial"`` or
        ``"gpu cuda"``. If None, the fastest available backend is chosen.
    :type backend: str or None
    :param device_id: Index of the device used by the backend.
    :type device_id: int

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster index of each point. Outliers are labelled -1. Clusters are numbered in
        order of their first point, so the labels don't depend on the backend.
    n_clusters_ : int
        Number of clusters found.
    seed_indices_ : ndarray of shape (n_clusters_,)
        Index of the seed of each cluster, so that ``labels_[seed_indices_[k]] == k``.
    rho_ : ndarray of shape (n_samples,)
        Local density of each point.
    delta_ : ndarray of shape (n_samples,)
        Distance of each point from its nearest higher, ``inf`` for the points without one.
    nearest_higher_ : ndarray of shape (n_samples,)
        Index of the nearest point with higher density within ``outlier_distance``, -1 if
        there is none (the seeds and the outliers).
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of the features seen during fit, defined only if X has string feature names.
    """

    def __init__(self,
                 density_radius=1.0,
                 min_density=1.0,
                 outlier_distance=None,
                 seeding_distance=None,
                 kernel="flat",
                 kernel_params=None,
                 metric="euclidean",
                 metric_params=None,
                 wrapped=None,
                 backend=None,
                 device_id=0):
        self.density_radius = density_radius
        self.min_density = min_density
        self.outlier_distance = outlier_distance
        self.seeding_distance = seeding_distance
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.metric = metric
        self.metric_params = metric_params
        self.wrapped = wrapped
        self.backend = backend
        self.device_id = device_id

    def fit(self, X, y=None, sample_weight=None, sample_sizes=None):
        """
        Cluster the points of X.

        :param X: Coordinates of the points. The clustering runs in single precision: a float32
            array in Fortran order (``np.asfortranarray``) is used without copies, otherwise it
            is converted with one copy.
        :type X: array-like of shape (n_samples, n_features)
        :param y: Ignored, present for API consistency.
        :param sample_weight: Weight of each point. If None, all the points have weight 1.
            A contiguous float32 array is used without copies.
        :type sample_weight: array-like of shape (n_samples,) or None
        :param sample_sizes: Sizes of independent samples stored one after the other in X,
            which are clustered separately in a single call (batched clustering). If None,
            X is clustered as a single sample.
        :type sample_sizes: array-like of int or None

        :returns: The fitted estimator.
        :rtype: CLUEstering
        """
        # The bindings use the coordinates in place as a structure of arrays, which is the Fortran
        # order of X, so a float32 Fortran-ordered X is not copied. The kernels and the metrics are
        # bound only for float, so the clustering runs in single precision.
        X = validate_data(self, X, dtype=np.float32, order="F", ensure_min_samples=1)
        n_samples, n_features = X.shape
        dtype = X.dtype.type
        weights = self._check_sample_weight(sample_weight, n_samples, dtype)
        if n_features > _MAX_FEATURES:
            raise ValueError(f"CLUEstering supports up to {_MAX_FEATURES} features, "
                             f"got {n_features}.")

        density_radius = self._check_positive("density_radius", self.density_radius)
        min_density = self._check_positive("min_density", self.min_density, strict=False)
        outlier_distance = self._check_positive(
            "outlier_distance",
            density_radius if self.outlier_distance is None else self.outlier_distance)
        seeding_distance = self._check_positive(
            "seeding_distance",
            density_radius if self.seeding_distance is None else self.seeding_distance)

        kernel = self._make_kernel()
        metric = self._make_metric(n_features)
        wrapped = self._check_wrapped(n_features)
        sizes = self._check_sample_sizes(sample_sizes, n_samples)
        module = self._get_backend_module()
        if not isinstance(self.device_id, Integral) or self.device_id < 0:
            raise ValueError(f"device_id must be a non-negative integer, got {self.device_id!r}.")

        labels = np.zeros(n_samples, dtype=np.int32)
        rho = np.empty(n_samples, dtype=dtype)
        delta = np.empty(n_samples, dtype=dtype)
        nearest_higher = np.empty(n_samples, dtype=np.int32)

        module.mainRunSeparate(dtype(density_radius),
                               dtype(min_density),
                               dtype(outlier_distance),
                               dtype(seeding_distance),
                               wrapped,
                               X,
                               weights,
                               labels,
                               rho,
                               delta,
                               nearest_higher,
                               kernel,
                               sizes,
                               int(self.device_id),
                               metric)

        self.labels_ = _canonical_labels(labels)
        self.n_clusters_ = int(self.labels_.max()) + 1
        self.rho_ = rho
        self.delta_ = delta
        self.nearest_higher_ = nearest_higher
        # the seeds are the clustered points without a nearest higher, sorted by cluster
        seeds = np.flatnonzero((nearest_higher == -1) & (self.labels_ >= 0))
        self.seed_indices_ = seeds[np.argsort(self.labels_[seeds])]
        return self

    def fit_predict(self, X, y=None, sample_weight=None, sample_sizes=None):
        """
        Cluster the points of X and return the cluster labels.

        :param X: Coordinates of the points.
        :type X: array-like of shape (n_samples, n_features)
        :param y: Ignored, present for API consistency.
        :param sample_weight: Weight of each point. If None, all the points have weight 1.
        :type sample_weight: array-like of shape (n_samples,) or None
        :param sample_sizes: Sizes of independent samples stored one after the other in X.
        :type sample_sizes: array-like of int or None

        :returns: Cluster index of each point, -1 for the outliers.
        :rtype: ndarray of shape (n_samples,)
        """
        return self.fit(X, sample_weight=sample_weight, sample_sizes=sample_sizes).labels_

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = False
        tags.input_tags.allow_nan = False
        return tags

    @staticmethod
    def _check_positive(name, value, strict=True):
        if not isinstance(value, Real) or isinstance(value, bool):
            raise TypeError(f"{name} must be a real number, got {type(value).__name__}.")
        if value < 0 or (strict and value == 0):
            bound = "positive" if strict else "non-negative"
            raise ValueError(f"{name} must be {bound}, got {value}.")
        return float(value)

    def _make_kernel(self):
        if self.kernel not in _KERNELS:
            raise ValueError(f"Invalid kernel {self.kernel!r}. Allowed choices are: "
                             f"{', '.join(_KERNELS)}.")
        factory, n_params, default = _KERNELS[self.kernel]
        params = default if self.kernel_params is None else tuple(self.kernel_params)
        if params is None or len(params) != n_params:
            raise ValueError(f"The {self.kernel} kernel requires {n_params} parameter(s) in "
                             f"kernel_params, got {params}.")
        return factory(*params)

    def _make_metric(self, n_features):
        if self.metric not in _METRICS:
            raise ValueError(f"Invalid metric {self.metric!r}. Allowed choices are: "
                             f"{', '.join(_METRICS)}.")
        factory, key = _METRICS[self.metric]
        params = {} if self.metric_params is None else dict(self.metric_params)
        if key is None:
            if params:
                raise ValueError(f"The {self.metric} metric takes no metric_params.")
            return factory()
        unknown = set(params) - {key}
        if unknown:
            raise ValueError(f"Unknown metric_params for the {self.metric} metric: "
                             f"{', '.join(sorted(unknown))}. Allowed: {key}.")
        if key not in params:
            if self.metric == "periodic_euclidean":
                raise ValueError("The periodic_euclidean metric requires "
                                 "metric_params={'periods': [...]}.")
            return factory()
        values = [float(v) for v in params[key]]
        if len(values) != n_features:
            raise ValueError(f"metric_params['{key}'] must have one value per feature "
                             f"({n_features}), got {len(values)}.")
        return factory(values)

    def _check_wrapped(self, n_features):
        if self.wrapped is None:
            return [0] * n_features
        wrapped = [int(bool(w)) for w in self.wrapped]
        if len(wrapped) != n_features:
            raise ValueError(f"wrapped must have one flag per feature ({n_features}), "
                             f"got {len(wrapped)}.")
        return wrapped

    @staticmethod
    def _check_sample_weight(sample_weight, n_samples, dtype):
        if sample_weight is None:
            return np.ones(n_samples, dtype=dtype)
        weights = np.asarray(sample_weight, dtype=dtype)
        if weights.ndim == 0:
            return np.full(n_samples, weights, dtype=dtype)
        if weights.shape != (n_samples,):
            raise ValueError(f"sample_weight must have shape ({n_samples},), "
                             f"got {weights.shape}.")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("sample_weight must contain finite non-negative values.")
        if not np.any(weights):
            raise ValueError("sample_weight must not be all zero.")
        # no copy if the weights are already contiguous and of the same type as X
        return np.ascontiguousarray(weights)

    @staticmethod
    def _check_sample_sizes(sample_sizes, n_samples):
        if sample_sizes is None:
            return None
        sizes = np.asarray(sample_sizes)
        if sizes.ndim != 1 or sizes.size == 0 or not np.issubdtype(sizes.dtype, np.integer):
            raise ValueError("sample_sizes must be a non-empty 1-D array of integers.")
        if np.any(sizes < 0) or sizes.sum() != n_samples:
            raise ValueError(f"sample_sizes must be non-negative and sum to the number of "
                             f"points ({n_samples}), got a sum of {sizes.sum()}.")
        return np.ascontiguousarray(sizes, dtype=np.uint32)

    def _get_backend_module(self):
        backend = _choose_default_backend() if self.backend is None else self.backend
        _validate_backend(backend)
        _check_backend_available(backend)
        return _BACKEND_MODULES[backend]()


def _canonical_labels(labels):
    """
    Renumber the clusters in order of their first point.

    The parallel backends number the clusters in a non-deterministic order, while the
    partition of the points is the same. Renumbering makes the labels reproducible.
    """
    labels = labels.astype(np.intp, copy=False)
    clustered = labels >= 0
    if not np.any(clustered):
        return np.full(labels.shape, -1, dtype=np.intp)
    ids, first = np.unique(labels[clustered], return_index=True)
    mapping = np.full(int(ids.max()) + 1, -1, dtype=np.intp)
    mapping[ids[np.argsort(first)]] = np.arange(ids.size, dtype=np.intp)
    out = np.full(labels.shape, -1, dtype=np.intp)
    out[clustered] = mapping[labels[clustered]]
    return out
