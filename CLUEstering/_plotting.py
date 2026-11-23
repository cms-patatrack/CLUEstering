"""
Plots of the results of a fitted CLUEstering estimator.
"""

import numpy as np
from sklearn.utils.validation import check_is_fitted


def plot_decision_graph(estimator, *, ax=None, log_rho=False):
    """
    Plot the decision graph of a fitted CLUEstering estimator.

    Each point is drawn at its local density ``rho_`` and at the distance ``delta_`` from its
    nearest higher. The seeds are the points above ``seeding_distance`` with a density of at least
    ``min_density``, the outliers are the points above ``outlier_distance`` with a lower density.
    The dashed lines show these thresholds, so the graph helps choosing ``min_density`` and
    ``seeding_distance``.

    The nearest higher is searched only within ``seeding_distance`` for the points with a density
    of at least ``min_density``, and within ``outlier_distance`` for the others. The points without
    a nearest higher in that distance, which include all the seeds, have an infinite ``delta_`` and
    are drawn with triangles on the top edge. To see how far the seeds are from each other, fit
    once with larger ``seeding_distance`` and ``outlier_distance``.

    The densities depend on ``density_radius``, so the graph has to be drawn again after changing
    it.

    :param estimator: A fitted CLUEstering estimator.
    :type estimator: CLUEstering
    :param ax: The axes to draw on. If None, the current axes are used.
    :type ax: matplotlib.axes.Axes or None
    :param log_rho: Whether to use a logarithmic scale for the densities.
    :type log_rho: bool

    :returns: The axes with the decision graph.
    :rtype: matplotlib.axes.Axes

    Example::

        est = clue.CLUEstering(density_radius=20., min_density=10.).fit(X)
        ax = clue.plot_decision_graph(est)
        ax.figure.savefig("decision_graph.png")
    """
    check_is_fitted(estimator, ("rho_", "delta_", "nearest_higher_", "seed_indices_"))
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    if ax is None:
        ax = plt.gca()

    params = estimator.get_params()
    min_density = params["min_density"]
    seeding_distance = (params["density_radius"] if params["seeding_distance"] is None
                        else params["seeding_distance"])
    outlier_distance = (params["density_radius"] if params["outlier_distance"] is None
                        else params["outlier_distance"])

    rho = estimator.rho_
    delta = estimator.delta_
    infinite = np.isinf(delta)
    finite_max = delta[~infinite].max() if np.any(~infinite) else 0.
    # the points with an infinite delta are drawn on the top edge, above all the others
    top = 1.1 * max(finite_max, seeding_distance, outlier_distance)
    y = np.where(infinite, top, delta)

    is_seed = np.zeros(rho.shape, dtype=bool)
    is_seed[estimator.seed_indices_] = True
    is_outlier = estimator.labels_ == -1
    # The points without a nearest higher are either seeds or outliers. The outliers with a finite
    # delta have a nearest higher, but their chain of nearest highers ends in an outlier.
    groups = (("followers", ~is_seed & ~is_outlier, "tab:gray", "o"),
              ("outliers (following an outlier)", is_outlier & ~infinite, "tab:blue", "o"),
              (r"outliers ($\delta = \infty$)", is_outlier & infinite, "tab:blue", "^"),
              (r"seeds ($\delta = \infty$)", is_seed, "tab:red", "^"))
    for label, mask, color, marker in groups:
        if np.any(mask):
            ax.scatter(rho[mask], y[mask], s=12, c=color, marker=marker, label=label)

    ax.axvline(min_density, color="black", linestyle="--", linewidth=1,
               label=f"min_density = {min_density:g}")
    ax.axhline(seeding_distance, color="tab:red", linestyle="--", linewidth=1,
               label=f"seeding_distance = {seeding_distance:g}")
    if outlier_distance != seeding_distance:
        ax.axhline(outlier_distance, color="tab:blue", linestyle="--", linewidth=1,
                   label=f"outlier_distance = {outlier_distance:g}")

    if log_rho:
        ax.set_xscale("log")
    ax.set_ylim(0., 1.05 * top)
    ax.set_xlabel(r"$\rho$ (local density)")
    ax.set_ylabel(r"$\delta$ (distance to the nearest higher)")
    # the densities depend on the density radius, so the graph is valid only for its value
    ax.set_title(f"Decision graph (density_radius = {params['density_radius']:g})")
    ax.legend(fontsize="small")
    return ax
