'''
Generates the datasets used in the tests of the merging of the clusters (test_merge_clusters.cpp)

- blob_pairs.csv: three pairs of gaussian blobs (sigma = 1, 500 points each), far from each other,
  with the centers of the two blobs at a distance of 2.5, 4 and 8 sigma. The density valley
  between the blobs of a pair gets deeper with the distance, and the blobs at 8 sigma do not
  touch. The points of pair k are the rows from 1000 * k to 1000 * (k + 1) - 1.
- uniform_noise.csv: 20000 points uniformly distributed in the unit square. With a small
  density radius CLUE finds hundreds of small clusters, all touching, which produce many
  cluster pairs in the merging.

The datasets are generated once and stored, because the random distributions of the C++
standard library give different numbers on different implementations.
'''

import numpy as np
import pandas as pd

SEPARATIONS = (2.5, 4.0, 8.0)
POINTS_PER_BLOB = 500


def blob_pairs(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    blobs = []
    for k, distance in enumerate(SEPARATIONS):
        for side in (-0.5, 0.5):
            center = [side * distance, 20. * k]
            blobs.append(rng.normal(center, 1., size=(POINTS_PER_BLOB, 2)))
    return np.vstack(blobs)


def uniform_noise(n_points: int = 20000, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0., 1., size=(n_points, 2))


def to_csv(points: np.ndarray, file_name: str) -> None:
    df = pd.DataFrame(points, columns=['x0', 'x1'])
    df['weight'] = 1.
    df.to_csv(file_name, index=False, float_format='%.6f')


if __name__ == '__main__':
    to_csv(blob_pairs(), 'blob_pairs.csv')
    to_csv(uniform_noise(), 'uniform_noise.csv')
