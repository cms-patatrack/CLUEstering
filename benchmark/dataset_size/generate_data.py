
import numpy as np
import pandas as pd


def generate_data(n_points: int,
                  n_dim: int,
                  n_clusters: int,
                  domain: tuple,
                  std: float = 1.,
                  seed: int = 0):
    # Generate random data
    np.random.seed(seed)
    points_per_cluster = int((.9 * n_points) / n_clusters)
    centers = np.random.uniform(domain[0], domain[1], size=(n_clusters, n_dim))
    clusters = []
    for cl in range(n_clusters):
        clusters.append(np.random.normal(centers[cl],
                        [std for _ in range(n_dim)],
                        size=(points_per_cluster, n_dim)))

    # concatenate clusters data
    data = np.concatenate([cl for cl in clusters])
    # add random noise
    noise = np.array([[np.random.uniform(domain[0], domain[1]) for _ in range(n_dim)] for _ in range(len(data), n_points)])
    data = np.concatenate((data, noise))

    # generate dataframe and save to file
    df = pd.DataFrame(data, columns=[f'x{i}' for i in range(n_dim)])
    df = pd.concat((df, pd.DataFrame(np.full(n_points, 1.), columns=['weight'])), axis=1)

    return df


if __name__ == '__main__':
    generate_data(1000, 2, 10, (-20, 20))
