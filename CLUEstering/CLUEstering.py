"""
Density based clustering algorithm developed at CERN.
"""

import sys
from dataclasses import dataclass
from glob import glob
import random as rnd
from math import sqrt
import time
import types
from typing import Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from os.path import dirname, exists, join
path = dirname(__file__)
sys.path.insert(1, join(path, 'lib'))
import CLUE_Convolutional_Kernels as clue_kernels
import CLUE_CPU_Serial as cpu_serial

backends = ["cpu serial"]
tbb_found = exists(str(*glob(join(path, 'lib/CLUE_CPU_TBB*.so'))))
if tbb_found:
    import CLUE_CPU_TBB as cpu_tbb
    backends.append("cpu tbb")
omp_found = exists(str(*glob(join(path, 'lib/CLUE_CPU_OMP*.so'))))
if omp_found:
    import CLUE_CPU_OMP as cpu_omp
    backends.append("cpu openmp")
cuda_found = exists(str(*glob(join(path, 'lib/CLUE_GPU_CUDA*.so'))))
if cuda_found:
    import CLUE_GPU_CUDA as gpu_cuda
    backends.append("gpu cuda")
hip_found = exists(str(*glob(join(path, 'lib/CLUE_GPU_HIP*.so'))))
if hip_found:
    import CLUE_GPU_HIP as gpu_hip
    backends.append("gpu hip")


def is_tbb_available():
    """
    Check if the library is compiled with TBB support.

    :returns: True if TBB is available, False otherwise.
    :rtype: bool
    """
    return tbb_found


def is_openmp_available():
    """
    Check if the library is compiled with OpenMP support.

    :returns: True if OpenMP is available, False otherwise.
    :rtype: bool
    """
    return omp_found


def is_cuda_available():
    """
    Check if the library is compiled with CUDA support.

    :returns: True if CUDA is available, False otherwise.
    :rtype: bool
    """
    return cuda_found


def is_hip_available():
    """
    Check if the library is compiled with HIP support.

    :returns: True if HIP is available, False otherwise.
    :rtype: bool
    """
    return hip_found


def test_blobs(n_samples: int, n_dim: int, n_blobs: int = 4, mean: float = 0,
               sigma: float = 0.5, x_max: float = 30, y_max: float = 30) -> pd.DataFrame:
    """
    Generate random 2D or 3D gaussian blobs for testing purposes.

    This function is useful for creating a random dataset to test the library.

    :param n_samples: Number of points in the dataset.
    :type n_samples: int
    :param n_dim: Number of dimensions (2 or 3).
    :type n_dim: int
    :param n_blobs: Number of blobs to generate. Defaults to 4.
    :type n_blobs: int, optional
    :param mean: Mean of the gaussian distribution for z-values. Defaults to 0.
    :type mean: float, optional
    :param sigma: Standard deviation of the gaussian distribution for z-values. Defaults to 0.5.
    :type sigma: float, optional
    :param x_max: Maximum x-coordinate for blobs. Defaults to 30.
    :type x_max: float, optional
    :param y_max: Maximum y-coordinate for blobs. Defaults to 30.
    :type y_max: float, optional

    :raises ValueError: If n_blobs < 0, sigma < 0, or n_dim > 3.

    :returns: DataFrame containing the generated blobs and their weights.
    :rtype: pd.DataFrame
    """
    if n_blobs < 0:
        raise ValueError('Wrong parameter value. The number of blobs must be positive.')
    if sigma < 0.:
        raise ValueError("Wrong parameter value. The mean and sigma of the blobs"
                         + " cannot be negative.")
    if n_dim > 3:
        raise ValueError("Wrong number of dimensions. Blobs can only be generated"
                         + " in 2 or 3 dimensions.")
    centers = []
    if n_dim == 2:
        data = {'x0': np.array([]), 'x1': np.array([]), 'weight': np.array([])}
        centers = [[x_max * rnd.random(),
                    y_max * rnd.random()] for _ in range(n_blobs)]
        blob_data = make_blobs(n_samples=n_samples,
                               centers=np.array(centers))[0]

        data['x0'] = blob_data.T[0]
        data['x1'] = blob_data.T[1]
        data['weight'] = np.full(shape=len(blob_data.T[0]), fill_value=1)

        return pd.DataFrame(data)
    if n_dim == 3:
        data = {'x0': [], 'x1': [], 'x2': [], 'weight': []}
        sqrt_samples = int(sqrt(n_samples))
        z_values = np.random.normal(mean, sigma, sqrt_samples)
        centers = [[x_max * rnd.random(),
                    y_max * rnd.random()] for _ in range(n_blobs)]

        for value in z_values:  # for every z value, a layer is generated.
            blob_data = make_blobs(n_samples=sqrt_samples,
                                   centers=np.array(centers))[0]
            data['x0'] = np.concatenate([data['x0'], blob_data.T[0]])
            data['x1'] = np.concatenate([data['x1'], blob_data.T[1]])
            data['x2'] = np.concatenate([data['x2'],
                                         np.full(shape=sqrt_samples,
                                                 fill_value=value)])
            data['weight'] = np.concatenate([data['weight'],
                                             np.full(shape=sqrt_samples,
                                                     fill_value=1)])

        return pd.DataFrame(data)


@dataclass()
class ClusteringDataSoA:
    """
    Container for input coordinates and clustering results using Structure of Arrays (SoA).

    :param coords: Input coordinates including weights.
    :type coords: np.ndarray
    :param results: Clustering results including cluster IDs.
    :type results: np.ndarray
    :param n_dim: Number of dimensions.
    :type n_dim: int
    :param n_points: Number of points.
    :type n_points: int
    """

    coords: np.ndarray
    results: np.ndarray
    n_dim : int
    n_points : int

    def __init__(self, coords=None, results=None, n_dim=None, n_points=None):
        self.coords = coords
        self.results = results
        self.n_dim = n_dim
        self.n_points = n_points


@dataclass(eq=False)
class cluster_properties:
    """
    Container for results produced by the clustering algorithm.

    :param n_clusters: Number of clusters constructed.
    :type n_clusters: int
    :param cluster_ids: Cluster ID for each point.
    :type cluster_ids: np.ndarray
    :param cluster_points: Lists of point IDs belonging to each cluster.
    :type cluster_points: np.ndarray
    :param points_per_cluster: Number of points per cluster.
    :type points_per_cluster: np.ndarray
    :param output_df: DataFrame containing the cluster_ids.
    :type output_df: pd.DataFrame
    """

    n_clusters : int
    cluster_ids : np.ndarray
    cluster_points : np.ndarray
    points_per_cluster : np.ndarray
    output_df : pd.DataFrame

    def __eq__(self, other):
        if self.n_clusters != other.n_clusters:
            return False
        if not (self.cluster_ids == other.cluster_ids).all():
            return False
        return True


class clusterer:
    """
    Wrapper class for performing clustering using the CLUE algorithm.

    :param dc: Spatial parameter controlling the region for local density calculation.
    :type dc: float
    :param rhoc: Density threshold separating seeds from outliers.
    :type rhoc: float
    :param dm: Spatial parameter controlling the region for follower search.
    :type dm: float
    :param ppbin: Average number of points per tile.
    :type ppbin: int
    :param kernel: Kernel used to calculate local density.
    :type kernel: clue_kernels.Algo.kernel
    :param clust_data: Container for input data.
    :type clust_data: ClusteringDataSoA
    :param clust_prop: Container for clustering results.
    :type clust_prop: cluster_properties
    :param elapsed_time: Execution time of the algorithm in nanoseconds.
    :type elapsed_time: float
    """

    def __init__(self, dc: float, rhoc: float, dm: [float, None] = None, seed_dc: [float, None] = None, ppbin: int = 128):
        self._dc = dc
        self._rhoc = rhoc
        self._dm = dm
        if dm is None:
            self._dm = dc
        self._seed_dc = seed_dc
        if seed_dc is None:
            self._seed_dc = dc
        self._ppbin = ppbin

        # Initialize attributes
        ## Data containers
        self.clust_data = None

        ## Kernel for calculation of local density
        self._kernel = clue_kernels.FlatKernel(0.5)

        ## Array specifyng which coordinates are periodic (wrapped)
        self.wrapped = None

        ## Output attributes
        self.clust_prop = None
        self._elapsed_time = 0.

    def set_params(self, dc: float, rhoc: float,
                   dm: [float, None] = None, seed_dc: [float, None] = None, ppbin: int = 128) -> None:
        """
        Set parameters for the clustering algorithm.

        :param dc: Spatial parameter for density calculation.
        :type dc: float
        :param rhoc: Density threshold.
        :type rhoc: float
        :param dm: Follower search region. Defaults to dc if None.
        :type dm: float or None
        :param seed_dc: Seed search region. Defaults to dc if None.
        :type seed_dc: float or None
        :param ppbin: Average points per tile.
        :type ppbin: int
        """
        self._dc = dc
        self._rhoc = rhoc
        if dm is not None:
            self._dm = dm
        else:
            self._dm = dc
        if seed_dc is not None:
            self._seed_dc = seed_dc
        else:
            self._seed_dc = dc
        self._ppbin = ppbin

    def _read_array(self, input_data: Union[list, np.ndarray]) -> None:
        """
        Read data from lists or np.ndarrays and initialize clustering data.

        :param input_data: Coordinates and weights of data points.
        :type input_data: list or np.ndarray

        :raises ValueError: If input data format is invalid.

        :returns: None
        """
        # [[x0, x1, x2, ...], [y0, y1, y2, ...], ... , [weights]]
        if len(input_data) < 2 or len(input_data) > 11:
            raise ValueError("Inadequate data. The supported dimensions are between" +
                             "1 and 10.")
        npoints = len(input_data[-1])
        ndim = len(input_data[:-1])
        coords = np.vstack([input_data[:-1],      # coordinates SoA
                            input_data[-1]],      # weights
                            dtype=np.float32)
        coords = np.ascontiguousarray(coords, dtype=np.float32)
        results = np.zeros(npoints, dtype=np.int32)    # cluster ids
        self.clust_data = ClusteringDataSoA(coords,
                                            results,
                                            ndim,
                                            npoints)

    def _read_string(self, input_data: str) -> Union[pd.DataFrame, None]:
        """
        Read input data from a CSV file.

        :param input_data: Path to the CSV file containing the input data.
        :type input_data: str

        :raises ValueError: If the file is not a CSV.

        :returns: DataFrame containing the input data.
        :rtype: pd.DataFrame
        """
        if not input_data.endswith('.csv'):
            raise ValueError('Wrong type of file. The file is not a csv file.')
        df_ = pd.read_csv(input_data, dtype=np.float32)
        return df_


    def _read_dict_df(self, input_data: Union[dict, pd.DataFrame]) -> pd.DataFrame:
        """
        Read input data from a dictionary or pandas DataFrame.

        :param input_data: Dictionary or DataFrame containing coordinates and weights.
        :type input_data: dict or pd.DataFrame

        :returns: DataFrame containing the input data.
        :rtype: pd.DataFrame
        """
        df_ = pd.DataFrame(input_data, copy=False, dtype=np.float32)
        return df_


    def _handle_dataframe(self, df_: pd.DataFrame) -> None:
        """
        Initialize the `clust_data` attribute from a DataFrame.

        :param df_: DataFrame produced by `_read_string` or `_read_dict_df`.
        :type df_: pd.DataFrame

        :raises ValueError: If the number of columns is less than 2 or greater than 11.

        :returns: None
        """
        if len(df_.columns) < 2:
            raise ValueError("Inadequate data. The data must contain at least one coordinate and the weight.")
        if len(df_.columns) > 11:
            raise ValueError("Inadequate data. The maximum number of dimensions supported is 10.")

        ndim = len(df_.columns) - 1
        npoints = len(df_.index)
        coords = df_.iloc[:, 0:-1].to_numpy()
        coords = np.vstack([coords.T, df_.iloc[:, -1]], dtype=np.float32)
        coords = np.ascontiguousarray(coords, dtype=np.float32)
        results = np.zeros(npoints, dtype=np.int32)

        self.clust_data = ClusteringDataSoA(coords, results, ndim, npoints)


    def read_data(self,
                  input_data: Union[pd.DataFrame, str, dict, list, np.ndarray],
                  wrapped_coords: Union[list, np.ndarray, None] = None) -> None:
        """
        Read input data and initialize clustering-related attributes.

        :param input_data: Data to read. Can be one of:
            - pandas DataFrame: must contain one column per coordinate plus one column for weight.
            - string: path to a CSV file containing the data.
            - dict: dictionary with coordinates and weights.
            - list or ndarray: list of coordinate lists plus a weight list.
        :type input_data: Union[pd.DataFrame, str, dict, list, np.ndarray]
        :param wrapped_coordinates: List or array indicating which dimensions are periodic.
        :type wrapped_coordinates: list or np.ndarray

        :raises ValueError: If the data format is not supported.

        :returns: None
        """
        if isinstance(input_data, (list, np.ndarray)):
            self._read_array(input_data)

        if isinstance(input_data, str):
            df = self._read_string(input_data)
            self._handle_dataframe(df)

        if isinstance(input_data, (dict, pd.DataFrame)):
            df = self._read_dict_df(input_data)
            self._handle_dataframe(df)

        if wrapped_coords is not None:
            self.wrapped = wrapped_coords
        else:
            self.wrapped = [0] * self.clust_data.n_dim


    def set_wrapped(self, wrapped_coords: Union[list, np.ndarray]) -> None:
        """
        Set which coordinates are periodic (wrapped).

        :param wrapped_coordinates: List or array indicating which dimensions are periodic.
        :type wrapped_coordinates: list or np.ndarray

        :returns: None
        """
        self.wrapped = wrapped_coords

    def choose_kernel(self,
                      choice: str,
                      parameters: Union[list, None] = None,
                      function: types.FunctionType = lambda: 0) -> None:
        """
        Set the kernel for local density calculation.

        The default kernel is a flat kernel with parameter 0.5.

        :param choice: Kernel type to use. Options are: 'flat', 'exp', 'gaus', or 'custom'.
        :type choice: str
        :param parameters: Parameters for the kernel. Required for 'flat', 'exp', 'gaus'.
            Not required for 'custom'.
        :type parameters: list or None
        :param function: Function to use for a custom kernel.
        :type function: function, optional

        :raises ValueError: If the number of parameters is invalid or the kernel choice is invalid.

        :returns: None
        """
        if choice == "flat":
            if len(parameters) != 1:
                raise ValueError("Wrong number of parameters. The flat kernel requires 1 parameter.")
            self._kernel = clue_kernels.FlatKernel(parameters[0])
        elif choice == "exp":
            if len(parameters) != 2:
                raise ValueError("Wrong number of parameters. The exponential kernel requires 2 parameters.")
            self._kernel = clue_kernels.ExponentialKernel(parameters[0], parameters[1])
        elif choice == "gaus":
            if len(parameters) != 3:
                raise ValueError("Wrong number of parameters. The gaussian kernel requires 3 parameters.")
            self._kernel = clue_kernels.GaussianKernel(parameters[0], parameters[1], parameters[2])
        elif choice == "custom":
            if len(parameters) != 0:
                raise ValueError("Wrong number of parameters. Custom kernels requires 0 parameters.")
        else:
            raise ValueError("Invalid kernel. Allowed choices are: flat, exp, gaus, custom.")


    @property
    def coords(self) -> np.ndarray:
        """
        Return the coordinates of the points used for clustering.

        :returns: Coordinates array.
        :rtype: np.ndarray
        """
        return self.clust_data.coords[:-1]


    @property
    def weight(self) -> np.ndarray:
        """
        Return the weights of the points.

        :returns: Weights array.
        :rtype: np.ndarray
        """
        return self.clust_data.coords[-1]


    @property
    def n_dim(self) -> int:
        """
        Return the number of dimensions.

        :returns: Number of dimensions.
        :rtype: int
        """
        return self.clust_data.n_dim


    @property
    def n_points(self) -> int:
        """
        Return the number of points in the dataset.

        :returns: Number of points.
        :rtype: int
        """
        return self.clust_data.n_points


    def list_devices(self, backend: str = "all") -> None:
        """
        List available devices for a given backend.

        :param backend: Backend to list devices for. Options: 'all', 'cpu serial', 'cpu tbb',
                        'cpu openmp', 'gpu cuda', 'gpu hip'. Defaults to 'all'.
        :type backend: str, optional

        :raises ValueError: If the backend is not valid.

        :returns: None
        """
        if backend == "all":
            cpu_serial.listDevices('cpu serial')
            if tbb_found:
                cpu_tbb.listDevices('cpu tbb')
            if omp_found:
                cpu_omp.listDevices('cpu openmp')
            if cuda_found:
                gpu_cuda.listDevices('gpu cuda')
            if hip_found:
                gpu_hip.listDevices('gpu hip')
        elif backend == "cpu serial":
            cpu_serial.listDevices(backend)
        elif backend == "cpu tbb":
            if tbb_found:
                cpu_tbb.listDevices(backend)
            else:
                print("TBB module not found. Please re-compile the library and try again.")
        elif backend == "cpu openmp":
            if omp_found:
                cpu_omp.listDevices(backend)
            else:
                print("OpenMP module not found. Please re-compile the library and try again.")
        elif backend == "gpu cuda":
            if cuda_found:
                gpu_cuda.listDevices(backend)
            else:
                print("CUDA module not found. Please re-compile the library and try again.")
        elif backend == "gpu hip":
            if hip_found:
                gpu_hip.listDevices(backend)
            else:
                print("HIP module not found. Please re-compile the library and try again.")
        else:
            raise ValueError("Invalid backend. Allowed choices are: all, cpu serial, cpu tbb, cpu openmp, gpu cuda, gpu hip.")


    def _partial_dimension_dataset(self, dimensions: list) -> np.ndarray:
        """
        Return a dataset containing only the selected dimensions.

        This method selects a subset of dimensions from the original dataset, useful
        for running CLUE in lower-dimensional spaces.

        :param dimensions: List of dimension indices to keep.
        :type dimensions: list[int]

        :returns: Array containing coordinates of the selected dimensions, with weights appended.
        :rtype: np.ndarray
        """
        coords = [np.copy(self.clust_data.coords[dim]) for dim in dimensions]
        coords.append(np.copy(self.clust_data.coords[-1]))
        return np.ascontiguousarray(coords, dtype=np.float32)


    def run_clue(self,
                 backend: str = "cpu serial",
                 block_size: int = 1024,
                 device_id: int = 0,
                 verbose: bool = False,
                 dimensions: Union[list, None] = None) -> None:
        """
        Execute the CLUE clustering algorithm.

        :param backend: Backend to use for execution. Defaults to 'cpu serial'.
        :type backend: str, optional
        :param block_size: Size of blocks for parallel execution. Defaults to 1024.
        :type block_size: int, optional
        :param device_id: Device ID to run the algorithm on. Defaults to 0.
        :type device_id: int, optional
        :param verbose: If True, prints execution time and number of clusters found.
        :type verbose: bool, optional
        :param dimensions: Optional list of dimensions to consider. Defaults to None.
        :type dimensions: list[int] or None, optional

        :returns: None
        """
        if dimensions is None:
            data = self.clust_data
        else:
            data = ClusteringDataSoA()
            data.coords = self._partial_dimension_dataset(dimensions)
            data.results = np.copy(self.clust_data.results)
            data.n_dim = len(dimensions)
            data.n_points = self.clust_data.n_points

        start = time.time_ns()
        if backend == "cpu serial":
            cluster_id_is_seed = cpu_serial.mainRun(self._dc, self._rhoc, self._dm, self._seed_dc,
                                                    self._ppbin, self.wrapped, data.coords, data.results,
                                                    self._kernel, data.n_dim,
                                                    data.n_points, block_size, device_id)
        elif backend == "cpu tbb":
            if tbb_found:
                cluster_id_is_seed = cpu_tbb.mainRun(self._dc, self._rhoc, self._dm, self._seed_dc,
                                                     self._ppbin, self.wrapped, data.coords, data.results,
                                                     self._kernel, data.n_dim,
                                                     data.n_points, block_size, device_id)
            else:
                print("TBB module not found. Please re-compile the library and try again.")
        elif backend == "cpu openmp":
            if omp_found:
                cluster_id_is_seed = cpu_omp.mainRun(self._dc, self._rhoc, self._dm, self._seed_dc,
                                                     self._ppbin, self.wrapped, data.coords, data.results,
                                                     self._kernel, data.n_dim,
                                                     data.n_points, block_size, device_id)
            else:
                print("OpenMP module not found. Please re-compile the library and try again.")
        elif backend == "gpu cuda":
            if cuda_found:
                cluster_id_is_seed = gpu_cuda.mainRun(self._dc, self._rhoc, self._dm, self._seed_dc,
                                                      self._ppbin, self.wrapped, data.coords, data.results,
                                                      self._kernel, data.n_dim,
                                                      data.n_points, block_size, device_id)
            else:
                print("CUDA module not found. Please re-compile the library and try again.")
        elif backend == "gpu hip":
            if hip_found:
                cluster_id_is_seed = gpu_hip.mainRun(self._dc, self._rhoc, self._dm, self._seed_dc,
                                                     self._ppbin, self.wrapped, data.coords, data.results,
                                                     self._kernel, data.n_dim,
                                                     data.n_points, block_size, device_id)
            else:
                print("HIP module not found. Please re-compile the library and try again.")

        finish = time.time_ns()
        cluster_ids = data.results
        n_clusters = np.max(cluster_ids) + 1

        cluster_points = [[] for _ in range(n_clusters)]
        for i in range(self.clust_data.n_points):
            if cluster_ids[i] != -1:
                cluster_points[cluster_ids[i]].append(i)

        points_per_cluster = np.array([len(clust) for clust in cluster_points])
        output_df = pd.DataFrame({'cluster_ids': cluster_ids})

        self.clust_prop = cluster_properties(n_clusters,
                                             cluster_ids,
                                             np.asarray(cluster_points, dtype=object),
                                             points_per_cluster,
                                             output_df)
        self._elapsed_time = (finish - start) / 1e6
        if verbose:
            print(f'CLUE executed in {self._elapsed_time} ms')
            print(f'Number of clusters found: {self.clust_prop.n_clusters}')

    def fit(self,
            data: Union[pd.DataFrame,str,dict,list,np.ndarray],
            backend: str = "cpu serial",
            block_size: int = 1024,
            device_id: int = 0,
            verbose: bool = False,
            dimensions: Union[list, None] = None) -> 'Clusterer':
        """
        Run the CLUE clustering algorithm on the input data.

        :param data: Input data. Can be a pandas DataFrame, a CSV file path (string),
                     a dictionary with coordinate keys and weight, or a list/array
                     containing coordinates and weights.
        :type data: Union[pd.DataFrame, str, dict, list, np.ndarray]
        :param backend: Backend to use for the algorithm execution.
        :type backend: str, optional
        :param block_size: Block size for parallel execution.
        :type block_size: int, optional
        :param device_id: ID of the device to run the algorithm on.
        :type device_id: int, optional
        :param verbose: If True, prints execution information.
        :type verbose: bool, optional
        :param dimensions: List of dimensions to consider. If None, all are used.
        :type dimensions: list or None, optional

        :return: Returns the clusterer object itself.
        :rtype: Clusterer

        :raises: Various exceptions if input data is invalid or clustering fails.
        """

        self.read_data(data)
        self.run_clue(backend, block_size, device_id, verbose, dimensions)
        return self

    def fit_predict(self,
                    data: [],
                    backend: str = "cpu serial",
                    block_size: int = 1024,
                    device_id: int = 0,
                    verbose: bool = False,
                    dimensions: Union[list, None] = None) -> np.ndarray:
        """
        Run the CLUE clustering algorithm and return the cluster labels.

        :param data: Input data. Can be a pandas DataFrame, a CSV file path (string),
                     a dictionary with coordinate keys and weight, or a list/array
                     containing coordinates and weights.
        :type data: Union[pd.DataFrame, str, dict, list, np.ndarray]
        :param backend: Backend to use for the algorithm execution.
        :type backend: str, optional
        :param block_size: Block size for parallel execution.
        :type block_size: int, optional
        :param device_id: ID of the device to run the algorithm on.
        :type device_id: int, optional
        :param verbose: If True, prints execution information.
        :type verbose: bool, optional
        :param dimensions: List of dimensions to consider. If None, all are used.
        :type dimensions: list or None, optional

        :return: Array containing the cluster index for every point.
        :rtype: np.ndarray

        :raises: Various exceptions if input data is invalid or clustering fails.
        """

        self.read_data(data)
        self.run_clue(backend, block_size, device_id, verbose, dimensions)
        return self.cluster_ids



    @property
    def n_clusters(self) -> int:
        """
        Return the number of clusters found.

        :returns: Number of clusters reconstructed by CLUE.
        :rtype: int
        """
        return self.clust_prop.n_clusters

    @property
    def cluster_ids(self) -> np.ndarray:
        """
        Index of the cluster to which each point belongs.

        :return: Array mapping each point to its cluster.
        :rtype: np.ndarray
        """

        return self.clust_prop.cluster_ids

    @property
    def labels(self) -> np.ndarray:
        """
        Alias for `cluster_ids`.

        :return: Array mapping each point to its cluster.
        :rtype: np.ndarray
        """

        return self.clust_prop.cluster_ids

    @property
    def cluster_points(self) -> np.ndarray:
        """
        List of points for each cluster.

        :return: Array of arrays containing point indices per cluster.
        :rtype: np.ndarray
        """

        return self.clust_prop.cluster_points

    @property
    def points_per_cluster(self) -> np.ndarray:
        """
        Number of points in each cluster.

        :return: Array containing the number of points in each cluster.
        :rtype: np.ndarray
        """

        return self.clust_prop.points_per_cluster

    @property
    def output_df(self) -> pd.DataFrame:
        """
        DataFrame containing cluster_ids.

        :return: Pandas DataFrame with cluster assignments.
        :rtype: pd.DataFrame
        """

        return self.clust_prop.output_df

    def cluster_centroid(self, cluster_index: int) -> np.ndarray:
        """
        Computes the centroid coordinates of a specified cluster.

        :param cluster_id: ID of the cluster.
        :type cluster_index: int
        :return: Coordinates of the cluster centroid.
        :rtype: np.ndarray
        :raises ValueError: If the cluster_id is invalid.
        """

        centroid = np.zeros(self.clust_data.n_dim)
        counter = 0
        for i in range(self.clust_data.n_points):
            if self.clust_prop.cluster_ids[i] == cluster_index:
                centroid += self.clust_data.coords.T[i][:-1]
                counter += 1
        centroid /= counter

        return centroid

    def cluster_centroids(self) -> np.ndarray:
        """
        Computes the centroids of all clusters.

        :return: Array of shape (n_clusters-1, n_dim) containing cluster centroids.
        :rtype: np.ndarray
        """

        centroids = np.zeros((self.clust_prop.n_clusters, self.clust_data.n_dim))
        for i in range(self.clust_data.n_points):
            if self.clust_prop.cluster_ids[i] != -1:
                centroids[self.clust_prop.cluster_ids[i]] += self.clust_data.coords.T[i][:-1]
        print(self.clust_prop.points_per_cluster[:-1].reshape(-1, 1))
        centroids /= self.clust_prop.points_per_cluster.reshape(-1, 1)

        return centroids

    def input_plotter(self, filepath: Union[str, None] = None, plot_title: str = '',
                      title_size: float = 16, x_label: str = 'x', y_label: str = 'y',
                      z_label: str = 'z', label_size: float = 16, pt_size: float = 1,
                      pt_colour: str = 'b', grid: bool = True, grid_style: str = '--',
                      grid_size: float = 0.2, x_ticks=None, y_ticks=None, z_ticks=None,
                      **kwargs) -> None:
        """
        Plots the input points in 1D, 2D, or 3D space.

        :param filepath: Path to save the plot. If None, the plot is shown interactively.
        :type filepath: str or None
        :param plot_title: Title of the plot.
        :type plot_title: str
        :param title_size: Font size of the plot title.
        :type title_size: float
        :param x_label: Label for the x-axis.
        :type x_label: str
        :param y_label: Label for the y-axis.
        :type y_label: str
        :param z_label: Label for the z-axis.
        :type z_label: str
        :param label_size: Font size for axis labels.
        :type label_size: float
        :param pt_size: Size of the points.
        :type pt_size: float
        :param pt_colour: Colour of the points.
        :type pt_colour: str
        :param grid: Whether to display a grid.
        :type grid: bool
        :param grid_style: Line style of the grid.
        :type grid_style: str
        :param grid_size: Line width of the grid.
        :type grid_size: float
        :param x_ticks: Custom tick locations for x-axis.
        :type x_ticks: list or None
        :param y_ticks: Custom tick locations for y-axis.
        :type y_ticks: list or None
        :param z_ticks: Custom tick locations for z-axis (only for 3D plots).
        :type z_ticks: list or None
        :param kwargs: Optional functions for converting coordinates.
        :type kwargs: dict

        :return: None
        :rtype: None
        """

        if self.clust_data.n_dim == 1:
            plt.scatter(self.coords[0],
                        np.zeros(self.clust_data.n_points),
                        s=pt_size,
                        color=pt_colour)

            # Customization of the plot title
            plt.title(plot_title, fontsize=title_size)

            # Customization of axis labels
            plt.xlabel(x_label, fontsize=label_size)
            plt.ylabel(y_label, fontsize=label_size)

            # Customization of the grid
            if grid:
                plt.grid(linestyle=grid_style, linewidth=grid_size)

            # Customization of axis ticks
            if x_ticks is not None:
                plt.xticks(x_ticks)

            if filepath is not None:
                plt.savefig(filepath)
            else:
                plt.show()
        elif self.clust_data.n_dim == 2:
            plt.scatter(self.coords[0],
                        self.coords[1],
                        s=pt_size,
                        color=pt_colour)

            # Customization of the plot title
            plt.title(plot_title, fontsize=title_size)

            # Customization of axis labels
            plt.xlabel(x_label, fontsize=label_size)
            plt.ylabel(y_label, fontsize=label_size)

            # Customization of the grid
            if grid:
                plt.grid(linestyle=grid_style, linewidth=grid_size)

            # Customization of axis ticks
            if x_ticks is not None:
                plt.xticks(x_ticks)
            if y_ticks is not None:
                plt.yticks(y_ticks)

            if filepath is not None:
                plt.savefig(filepath)
            else:
                plt.show()
        else:
            fig = plt.figure()
            ax_ = fig.add_subplot(projection='3d')
            ax_.scatter(self.coords[0],
                        self.coords[1],
                        self.coords[2],
                        s=pt_size,
                        color=pt_colour)

            # Customization of the plot title
            ax_.set_title(plot_title, fontsize=title_size)

            # Customization of axis labels
            ax_.set_xlabel(x_label, fontsize=label_size)
            ax_.set_ylabel(y_label, fontsize=label_size)
            ax_.set_zlabel(z_label, fontsize=label_size)

            # Customization of the grid
            if grid:
                plt.grid(linestyle=grid_style, linewidth=grid_size)

            # Customization of axis ticks
            if x_ticks is not None:
                ax_.set_xticks(x_ticks)
            if y_ticks is not None:
                ax_.set_yticks(y_ticks)
            if z_ticks is not None:
                ax_.set_zticks(z_ticks)

            if filepath is not None:
                plt.savefig(filepath)
            else:
                plt.show()

    def cluster_plotter(self, filepath: Union[str, None] = None, plot_title: str = '',
                        title_size: float = 16, x_label: str = 'x', y_label: str = 'y',
                        z_label: str = 'z', label_size: float = 16, outl_size: float = 10,
                        pt_size: float = 10, grid: bool = True,
                        grid_style: str = '--', grid_size: float = 0.2, x_ticks=None,
                        y_ticks=None, z_ticks=None, **kwargs) -> None:
        """
        Plots clusters with different colors and outliers as gray crosses.

        :param filepath: Path to save the plot. If None, the plot is shown interactively.
        :type filepath: str or None
        :param plot_title: Title of the plot.
        :type plot_title: str
        :param title_size: Font size of the plot title.
        :type title_size: float
        :param x_label: Label for the x-axis.
        :type x_label: str
        :param y_label: Label for the y-axis.
        :type y_label: str
        :param z_label: Label for the z-axis.
        :type z_label: str
        :param label_size: Font size for axis labels.
        :type label_size: float
        :param outl_size: Marker size for outliers.
        :type outl_size: float
        :param pt_size: Marker size for cluster points.
        :type pt_size: float
        :param grid: Whether to display a grid.
        :type grid: bool
        :param grid_style: Line style of the grid.
        :type grid_style: str
        :param grid_size: Line width of the grid.
        :type grid_size: float
        :param x_ticks: Custom tick locations for x-axis.
        :type x_ticks: list or None
        :param y_ticks: Custom tick locations for y-axis.
        :type y_ticks: list or None
        :param z_ticks: Custom tick locations for z-axis (only for 3D plots).
        :type z_ticks: list or None
        :param kwargs: Optional functions for converting coordinates.
        :type kwargs: dict

        :return: None
        :rtype: None
        """

        if self.clust_data.n_dim == 1:
            data = {'x0': self.coords[0],
                    'x1': np.zeros(self.clust_data.n_points),
                    'cluster_ids': self.clust_prop.cluster_ids}
            df_ = pd.DataFrame(data)

            max_clusterid = max(df_["cluster_ids"])

            df_out = df_[df_.cluster_ids == -1] # Outliers
            plt.scatter(df_out.x0, df_out.x1, s=outl_size, marker='x', color='0.4')
            for i in range(0, max_clusterid+1):
                dfi = df_[df_.cluster_ids == i] # ith cluster
                plt.scatter(dfi.x0, dfi.x1, s=pt_size, marker='.')

            # Customization of the plot title
            plt.title(plot_title, fontsize=title_size)

            # Customization of axis labels
            plt.xlabel(x_label, fontsize=label_size)
            plt.ylabel(y_label, fontsize=label_size)

            # Customization of the grid
            if grid:
                plt.grid(linestyle=grid_style, linewidth=grid_size)

            # Customization of axis ticks
            if x_ticks is not None:
                plt.xticks(x_ticks)

            if filepath is not None:
                plt.savefig(filepath)
            else:
                plt.show()
        elif self.clust_data.n_dim == 2:
            data = {'x0': self.coords[0],
                    'x1': self.coords[1],
                    'cluster_ids': self.cluster_ids}
            df_ = pd.DataFrame(data)

            max_clusterid = max(df_["cluster_ids"])

            df_out = df_[df_.cluster_ids == -1] # Outliers
            plt.scatter(df_out.x0, df_out.x1, s=outl_size, marker='x', color='0.4')
            for i in range(0, max_clusterid+1):
                dfi = df_[df_.cluster_ids == i] # ith cluster
                plt.scatter(dfi.x0, dfi.x1, s=pt_size, marker='.')

            # Customization of the plot title
            plt.title(plot_title, fontsize=title_size)

            # Customization of axis labels
            plt.xlabel(x_label, fontsize=label_size)
            plt.ylabel(y_label, fontsize=label_size)

            # Customization of the grid
            if grid:
                plt.grid(linestyle=grid_style, linewidth=grid_size)

            # Customization of axis ticks
            if x_ticks is not None:
                plt.xticks(x_ticks)
            if y_ticks is not None:
                plt.yticks(y_ticks)

            if filepath is not None:
                plt.savefig(filepath)
            else:
                plt.show()
        else:
            data = {'x0': self.coords[0],
                    'x1': self.coords[1],
                    'x2': self.coords[2],
                    'cluster_ids': self.clust_prop.cluster_ids}
            df_ = pd.DataFrame(data)

            max_clusterid = max(df_["cluster_ids"])
            fig = plt.figure()
            ax_ = fig.add_subplot(projection='3d')

            df_out = df_[df_.cluster_ids == -1]
            ax_.scatter(df_out.x0, df_out.x1, df_out.x2, s=outl_size, color = 'grey', marker = 'x')
            for i in range(0, max_clusterid+1):
                dfi = df_[df_.cluster_ids == i]
                ax_.scatter(dfi.x0, dfi.x1, dfi.x2, s=pt_size, marker = '.')

            # Customization of the plot title
            ax_.set_title(plot_title, fontsize=title_size)

            # Customization of axis labels
            ax_.set_xlabel(x_label, fontsize=label_size)
            ax_.set_ylabel(y_label, fontsize=label_size)
            ax_.set_zlabel(z_label, fontsize=label_size)

            # Customization of the grid
            if grid:
                plt.grid(linestyle=grid_style, linewidth=grid_size)

            # Customization of axis ticks
            if x_ticks is not None:
                ax_.set_xticks(x_ticks)
            if y_ticks is not None:
                ax_.set_yticks(y_ticks)
            if z_ticks is not None:
                ax_.set_zticks(z_ticks)

            if filepath is not None:
                plt.savefig(filepath)
            else:
                plt.show()


    def to_csv(self, output_folder: str, file_name: str) -> None:
        """
        Creates a file containing the coordinates of all the points and their cluster_ids.

        :param output_folder: Full path to the desired output folder.
        :type output_folder: str
        :param file_name: Name of the file, with the '.csv' suffix.
        :type file_name: str

        :return: None
        :rtype: None
        """

        if output_folder[-1] != '/':
            output_folder += '/'
        if file_name[-4:] != '.csv':
            file_name += '.csv' 
        out_path = output_folder + file_name
        data = {}
        for i in range(self.clust_data.n_dim):
            data['x' + str(i)] = self.clust_data.coords[i]
        data['weight'] = self.clust_data.coords[-1]
        data['cluster_ids'] = self.clust_prop.cluster_ids

        df_ = pd.DataFrame(data)
        df_.to_csv(out_path, index=False)


    def import_clusterer(self, input_folder: str, file_name: str) -> None:
        """
        Imports the results of a previous clustering.

        :param input_folder: Full path to the folder containing the CSV file.
        :type input_folder: str
        :param file_name: Name of the file, with the '.csv' suffix.
        :type file_name: str

        :raises ValueError: If the file does not exist or cannot be read correctly.

        :return: None
        :rtype: None
        """

        in_path = input_folder + file_name
        df_ = pd.read_csv(in_path, dtype=float)
        cluster_ids = np.asarray(df_["cluster_ids"], dtype=int)

        self._handle_dataframe(df_.iloc[:, :-2])

        n_clusters = np.max(cluster_ids) + 1

        cluster_points = [[] for _ in range(n_clusters)]
        for i in range(self.clust_data.n_points):
            cluster_points[cluster_ids[i]].append(i)

        points_per_cluster = np.array([len(clust) for clust in cluster_points])
        self.clust_prop = cluster_properties(
            n_clusters,
            cluster_ids,
            np.asarray(cluster_points, dtype=object),
            points_per_cluster,
            df_
        )
