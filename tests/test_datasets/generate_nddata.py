import numpy as np
import pandas as pd
import CLUEstering as clue

def create_centers(n_clusters: int, n_dim: int, limits: tuple, seed: int=0):
    """
    Returns the coordinates of the centers of a number of randomly generated clusters

    Parameters
    ----------
    n_clusters : int
        Number of clusters in the dataset
        Since the positions of the clusters are randomly generated, they might overlap, so
        the real number of clusters might be different
    n_dim : int
        Number of features (dimensions) for each point of the dataset
    limits : tuple
        Range of values that the spatial coordinates can take
    seed : int
        Seed used in the random generation
    """

    centers = []
    for i in range(n_clusters):
        np.random.seed(seed + i)
        centers.append([np.random.uniform(limits[0], limits[1]) for _ in range(n_dim)])

    return centers

def generate_data(n_clusters: int,
                  n_points: int,
                  n_dim: int,
                  std: float,
                  seed: int=0,
                  noise: int=0) -> pd.DataFrame:
    """
    Create some random data for clustering

    Generates N-dimensional clusters of gaussianely distributed.
    A seed can be passed in order to generate the dataset in a reproducible way.

    Parameters
    ----------
    n_clusters : int
        Number of clusters in the dataset
        Since the positions of the clusters are randomly generated, they might overlap, so
        the real number of clusters might be different
    n_points : int
        Number of points in each cluster
    n_dim : int
        Number of features (dimensions) for each point of the dataset
    std : float
        Standard deviation of the distribution that generates the points coordinates
    seed : int
        Seed used in the random generation
    noise : int
        Number of points randomly distributed accross the space which represent the noise

    Returns
    -------
    pd.Dataframe
        Dataframe containing the N-dimensional points
    """
    space_limits = (0, 50) # Change this to change the dimension of the phase space
    centers = create_centers(n_clusters, n_dim, space_limits)

    # Initialize data dictionary
    data = {}
    for dim in range(n_dim):
        data['x' + str(dim)] = []
    data['weight'] = []

    # Fill with the cluster points coordinates
    for cluster_center in centers:
        for dim in range(n_dim):
            np.random.seed(seed + dim)
            data['x' + str(dim)] += list(np.random.normal(cluster_center[dim], std, n_points))
        data['weight'] += list(np.full(n_points, 1))

    # Add some noise
    for i in range(noise):
        for dim in range(n_dim):
            np.random.seed(seed + i + dim)
            data['x' + str(dim)] += [np.random.uniform(space_limits[0], space_limits[1])]
    data['weight'] += list(np.full(noise, 1))

    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_data(3, 1000, 2, 0.1, 0, 100)
    c = clue.clusterer(0.4, 5., 1.3)
    c.read_data(df)
    c.input_plotter()
