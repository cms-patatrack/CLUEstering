
import numpy as np
import pandas as pd

def check_result(output_file, truth_file):
    '''
    Checks if the output file is the same as the truth file
    '''

    output = pd.read_csv(output_file)
    truth = pd.read_csv(truth_file)

    # Check if the number of clusters is the same
    n_clusters_o = len(truth['cluster_ids'].unique())
    n_clusters_t = len(truth['cluster_ids'].unique())
    if n_clusters_o != n_clusters_t:
        return False

    # Check if the number of points per cluster is the same
    points_per_cluster_o = np.asarray(output.groupby('cluster_ids').size())
    points_per_cluster_t = np.asarray(truth.groupby('cluster_ids').size())
    if not np.array_equal(points_per_cluster_o, points_per_cluster_t):
        return False

    return True
