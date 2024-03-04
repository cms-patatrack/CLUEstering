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
tbb_found = exists(str(*glob(join(path, 'lib/CLUE_CPU_TBB*.so'))))
if tbb_found:
    import CLUE_CPU_TBB as cpu_tbb
cuda_found = exists(str(*glob(join(path, 'lib/CLUE_CPU_CUDA*.so'))))
if cuda_found:
    import CLUE_GPU_CUDA as gpu_cuda
hip_found = exists(str(*glob(join(path, 'lib/CLUE_CPU_HIP*.so'))))
if hip_found:
    import CLUE_GPU_HIP as gpu_hip

def test_blobs(n_samples: int, n_dim: int , n_blobs: int = 4, mean: float = 0,
               sigma: float = 0.5, x_max: float = 30, y_max: float = 30) -> pd.DataFrame:
    """
    Returns a dataframe containing randomly generated 2-dimensional or 3-dimensional blobs.

    This functions serves as a tool for generating a random dataset to test the library.

    Parameters
    ----------
    n_samples : int
        The number of points in the dataset.
    n_dim : int
        The number of dimensions.
    n_blobs : int, optional
        The number of blobs that should be produced. By default it is set to 4.
    mean : float, optional
        The mean of the gaussian distribution of the z values.
    sigma : float, optional
        The standard deviation of the gaussian distribution of the z values.
    x_max : float, optional
        Limit of the space where the blobs are created in the x direction.
    y_max : float, optional
        Limit of the space where the blobs are created in the y direction.

    Returns
    -------
    Pandas DataFrame
        DataFrame containing n_blobs gaussian blobs.
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
        centers = [[x_max * rnd.random(), y_max * rnd.random()] for _ in range(n_blobs)]
        blob_data = make_blobs(n_samples=n_samples, centers=np.array(centers))[0]

        data['x0'] = blob_data.T[0]
        data['x1'] = blob_data.T[1]
        data['weight'] = np.full(shape=len(blob_data.T[0]), fill_value=1)


        return pd.DataFrame(data)
    if n_dim == 3:
        data = {'x0': [], 'x1': [], 'x2': [], 'weight': []}
        sqrt_samples = int(sqrt(n_samples))
        z_values = np.random.normal(mean, sigma,sqrt_samples)
        centers = [[x_max * rnd.random(), y_max * rnd.random()] for _ in range(n_blobs)]

        for value in z_values: # for every z value, a layer is generated.
            blob_data = make_blobs(n_samples=sqrt_samples, centers=np.array(centers))[0]
            data['x0'] = np.concatenate([data['x0'], blob_data.T[0]])
            data['x1'] = np.concatenate([data['x1'], blob_data.T[1]])
            data['x2'] = np.concatenate([data['x2'], np.full(shape=sqrt_samples,
                                                             fill_value=value)])
            data['weight'] = np.concatenate([data['weight'], np.full(shape=sqrt_samples,
                                                                     fill_value=1)])

        return pd.DataFrame(data)

@dataclass()
class clustering_data:
    """
    Container characterizing the data used for clustering.

    Attributes
    ----------
    coords : np.ndarray
        Spatially normalized data coordinates in the coordinate system used for clustering.
    original_coords : np.ndarray
        Data coordinates in the original coordinate system provided by the user.
    weight : np.ndarray
        Weight values of the data points.
    domain_ranges : list
        List containing the ranges of the domains for every coordinate.
    n_dim : int
        Number of dimensions.
    n_points : int
        Number of points in the clustering data.
    """

    coords : np.ndarray
    original_coords : np.ndarray
    weight : np.ndarray
    n_dim : int
    n_points : int

@dataclass(eq=False)
class cluster_properties:
    """
    Container of the data resulting from the clusterization of the input data.

    Attributes
    ----------
    n_clusters : int
        Number of clusters constructed.
    cluster_ids : np.ndarray
        Array containing the cluster_id of each point.
    is_seed : np.ndarray
        Array of integers containing '1' if a point is a seed and '0' if it isn't
    cluster_points : np.ndarray
        Array containing, for each cluster, the list of point_ids corresponding to the
        clusters bolonging to that cluster.
    points_per_cluster : np.ndarray
        Array containing the number of points belonging to each cluster.
    output_df : pd.DataFrame
        Dataframe containing is_seed and cluster_ids as columns.
    """

    n_clusters : int
    cluster_ids : np.ndarray
    is_seed : np.ndarray
    cluster_points : np.ndarray
    points_per_cluster : np.ndarray
    output_df : pd.DataFrame

    def __eq__(self, other):
        if self.n_clusters != other.n_clusters:
            return False
        if not (self.cluster_ids == other.cluster_ids).all():
            return False
        if not (self.is_seed == other.is_seed).all():
            return False

        return True


class clusterer:
    """
    Class representing a wrapper for the methods using in the process of clustering using
    the CLUE algorithm.

    Attributes
    ----------
    dc_ : float
        Spatial parameter indicating how large is the region over which the local density of
        each point is calculated.
    rhoc : float
        Parameter representing the density threshold value which divides seeds and
        outliers.

        Points with a density lower than rhoc can't be seeds, can only be followers
        or outliers.
    outlier : float
        Multiplicative increment of dc_ for getting the region over which the followers of a
        point are searched.

        While dc_ determines the size of the search box in which the neighbors of a point are
        searched when calculating its local density, when looking for followers while trying
        to find potential seeds the size of the search box is given by dm = dc_ * outlier.
    ppbin : int
        Average number of points to be found in each tile.
    kernel : Algo.kernel
        Convolution kernel used to calculate the local density of the points.
    clust_data : clustering_data
        Container of the data used by the clustering algorithm.
    clust_prop : cluster_properties
        Container of the data produced as output of the algorithm.
    elapsed_time : int
        Execution time of the algorithm, expressed in nanoseconds.
    """

    def __init__(self, dc_: float, rhoc_: float, outlier_: float, ppbin: int = 10):
        self.dc_ = dc_
        self.rhoc = rhoc_
        self.outlier = outlier_
        self.ppbin = ppbin

        # Initialize attributes
        ## Data containers
        self.clust_data = None
        self.scaler = StandardScaler()

        ## Kernel for calculation of local density
        self.kernel = clue_kernels.FlatKernel(0.5)

        ## Output attributes
        self.clust_prop = None
        self.elapsed_time = 0.

    def _read_array(self, input_data: Union[list,np.ndarray]) -> None:
        """
        Reads data provided with lists or np.ndarrays

        Attributes
        ----------
        input_data : list, np.ndarray
            The coordinates and weight values of the data points

        Modified attributes
        -------------------
        clust_data : clustering_data
            Properties of the input data

        Returns
        -------
        None
        """

        # [[x0, x1, x2, ...], [y0, y1, y2, ...], ... , [weights]]
        if isinstance(input_data[0][0], (int, float)):
            if len(input_data) < 2 or len(input_data) > 11:
                raise ValueError("Inadequate data. The supported dimensions are between" +
                                 "1 and 10.")
            self.clust_data = clustering_data(np.asarray(input_data[:-1]).T,
                                              np.copy(np.asarray(input_data[:-1]).T),
                                              np.asarray(input_data[-1]),
                                              len(input_data[:-1]),
                                              len(input_data[-1]))
        # [[[x0, y0, z0, ...], [x1, y1, z1, ...], ...], [weights]]
        else:
            if len(input_data) != 2:
                raise ValueError("Inadequate data. The data must contain a weight value" +
                                 "for each point.")
            self.clust_data = clustering_data(np.asarray(input_data[0]),
                                              np.copy(np.asarray(input_data[0])),
                                              np.asarray(input_data[-1]),
                                              len(input_data[0][0]),
                                              len(input_data[-1]))

    def _read_string(self, input_data: str) -> Union[pd.DataFrame,None]:
        """
        Reads data provided by passing a string containing the path to a csv file

        Attributes
        ----------
        input_data : str
            The path to the csv file containing the input data

        Modified attributes
        -------------------
        None

        Returns
        -------------------
        pd.DataFrame
            Dataframe containing the input data
        """

        if not input_data.endswith('.csv'):
            raise ValueError('Wrong type of file. The file is not a csv file.')
        df_ = pd.read_csv(input_data)
        return df_

    def _read_dict_df(self, input_data: Union[dict,pd.DataFrame]) -> pd.DataFrame:
        """
        Reads data provided using dictionaries or pandas dataframes

        Attributes
        ----------
        input_data : dict, pd.DataFrame
            The coordinates and weight values of the data points

        Modified attributes
        -------------------
        None

        Returns
        -------------------
        pd.DataFrame
            Dataframe containing the input data
        """

        df_ = pd.DataFrame(input_data, copy=False)
        return df_

    def _handle_dataframe(self, df_: pd.DataFrame) -> None:
        """
        Constructs the clust_data attribute from the dataframe produced by the
        _read_string or _read_dict_df methods

        Modified attributes
        -------------------
        clust_data : clustering_data
            Properties of the input data

        Returns
        -------
        None
        """

        # Check that the user provided the weights
        if 'weight' not in df_.columns:
            raise ValueError("Inadequate data. The input dataframe must"
                             + " contain a weight column.")

        coordinate_columns = [col for col in df_.columns if col[0] == 'x']

        # Check that the dimensionality of the dataset is adequate
        if len(df_.columns) < 2:
            raise ValueError("Inadequate data. The data must contain"
                             + " at least one coordinate and the weight.")
        if len(coordinate_columns) > 10:
            raise ValueError("Inadequate data. The maximum number of"
                             + " dimensions supported is 10.")
        n_dim = len(coordinate_columns)
        n_points = len(df_.index)
        coords = df_.iloc[:, 0:-1].to_numpy()

        self.clust_data = clustering_data(coords,
                                          np.copy(coords),
                                          np.asarray(df_['weight']),
                                          n_dim,
                                          n_points)

    def _rescale(self) -> None:
        """
        Normalizes the input data using a standard scaler

        Modified attributes
        -------------------
        clust_data.coords : np.ndarray
            Array containing the coordinates and weight values of the data points

        Returns
        -------
        None
        """

        for dim in range(self.clust_data.n_dim):
            self.clust_data.coords.T[dim] = \
            self.scaler.fit_transform(
                    self.clust_data.coords.T[dim].reshape(-1, 1)).reshape(1, -1)[0]

    def read_data(self,
                  input_data: Union[pd.DataFrame,str,dict,list,np.ndarray]) -> None:
        """
        Reads the data in input and fills the class members containing the coordinates
        of the points, the weight, the number of dimensions and the number of points.

        Parameters
        ----------
        input_data : pandas dataframe
            The dataframe should contain one column for every
            coordinate, each one called 'x*', and one column for the weight.
        input_data : string
            The string should contain the full path to a csv file containing
            the data.
        input_data : dict
        input_data : array_like
            The list or numpy array should contain a list of lists for the
            coordinates and a list for the weight.
        kwargs : tuples
            Tuples corresponding to the domain of any periodic variables. The
            keyword should be the keyword of the corrispoding variable.

        Modified attributes
        -------------------
        coords : ndarray
            Point coordinates used for clustering, spatially normalized.
        original_coords : ndarray
            Point coordinates in the original coordinate system used by the user.
        weight : ndarray
            Weights of all the points.
        domain_ranges : list of Algo.domain_t
            List of the domains for each coordinate.
        n_dim : int
            The number of dimensions in which we are calculating the clusters.
        n_points : int
            The number of points in the dataset.

        Returns
        -------
        None
        """

        # lists and np ndarrays
        if isinstance(input_data, (list, np.ndarray)):
            self._read_array(input_data)

        # path to .csv file or pandas dataframe
        if isinstance(input_data, (str)):
            df = self._read_string(input_data)
            self._handle_dataframe(df)

        if isinstance(input_data, (dict, pd.DataFrame)):
            df = self._read_dict_df(input_data)
            self._handle_dataframe(df)

        # Rescale the coordinates with a standard scaler
        self._rescale()

    def change_coordinates(self, **kwargs: types.FunctionType) -> None:
        """
        Change the coordinate system

        Parameters
        ----------
        kwargs : function objects
            The functions for the change of coordinates.
            The keywords of the arguments are the coordinates names (x0, x1, ...).

        Modifies attributes
        -------------------
        coords : ndarray
            Coordinates used for clustering converted in the chosen coordinate system.

        Returns
        -------
        None
        """

        # Change the coordinate system
        for coord, func in kwargs.items():
            self.clust_data.coords[int(coord[1])] = func(self.clust_data.original_coords)

            # Normalize the coordinate with a standard scaler
            self.clust_data.coords[int(coord[1])] = \
                self.scaler.fit_transform(
                    self.clust_data.coords[int(coord[1])].reshape(-1, 1)
                ).reshape(1, -1)[0]

    def choose_kernel(self,
                      choice: str,
                      parameters: Union[list,None] = None,
                      function: types.FunctionType = lambda: 0) -> None:
        """
        Changes the kernel used in the calculation of local density. The default kernel
        is a flat kernel with parameter 0.5

        Parameters
        ----------
        choice : string
            The type of kernel that you want to choose (flat, exp, gaus or custom).
        parameters : array_like, optional
            List of the parameters needed by the kernels.
            The flat kernel requires one, the exponential requires two
            (amplitude and mean), the gaussian requires three (amplitude,
            mean and standard deviation) and the custom doesn't require any.
        function : function object, optional
            Function that should be used as kernel when the custom kernel is chosen.

        Modified attributes
        -------------------
        kernel : Algo.kernel

        Return
        ------
        None
        """

        if choice == "flat":
            if len(parameters) != 1:
                raise ValueError("Wrong number of parameters. The flat kernel"
                                 + " requires 1 parameter.")
            self.kernel = CLUE_Convolutional_Kernels.FlatKernel(parameters[0])
        elif choice == "exp":
            if len(parameters) != 2:
                raise ValueError("Wrong number of parameters. The exponential"
                                 + " kernel requires 2 parameters.")
            self.kernel = CLUE_Convolutional_Kernels.ExponentialKernel(parameters[0],
                                                                       parameters[1])
        elif choice == "gaus":
            if len(parameters) != 3:
                raise ValueError("Wrong number of parameters. The gaussian" +
                                 " kernel requires 3 parameters.")
            self.kernel = CLUE_Convolutional_Kernels.GaussinKernel(parameters[0],
                                                                   parameters[1],
                                                                   parameters[2])
        elif choice == "custom":
            if len(parameters) != 0:
                raise ValueError("Wrong number of parameters. Custom kernels"
                                 + " requires 0 parameters.")
        else:
            raise ValueError("Invalid kernel. The allowed choices for the"
                             + " kernels are: flat, exp, gaus and custom.")

    def list_devices(self, backend: str = "all") -> None:
        """
        Lists the devices available for the chosen backend.

        Parameters
        ----------
        backend : string, optional
            The backend for which the devices are listed. The allowed values are
            'all', 'cpu serial', 'cpu tbb' and 'gpu cuda'.
            The default value is 'all'.

        Raises
        ------
        ValueError : If the backend is not valid.
        """

        if backend == "all":
            cpu_serial.listDevices('cpu serial')
            if tbb_found:
                cpu_tbb.listDevices('cpu tbb')
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
            raise ValueError("Invalid backend. The allowed choices for the"
                             + " backend are: all, cpu serial, cpu tbb and gpu cuda.")

    def _partial_dimension_dataset(self, dimensions: list):
        """
        Returns a dataset containing only the coordinates of the chosen dimensions.

        This method returns a dataset containing only the coordinates of the chosen
        dimensions when a set of dimensions is chosen in the `run_clue` method. This
        allows to run the algorithm in a lower dimensional space.

        Parameters
        ----------
        dimensions : list
            The list of the dimensions that should be considered.

        Returns
        -------
        np.ndarray
            Array containing the coordinates of the chosen dimensions.

        """

        return np.array([self.clust_data.coords.T[dim] for dim in dimensions]).T

    def run_clue(self,
                 backend: str = "cpu serial",
                 block_size: int = 1024,
                 device_id: int = 0,
                 verbose: bool = False,
                 dimensions: Union[list, None] = None) -> None:
        """
        Executes the CLUE clustering algorithm.

        Parameters
        ----------
        verbose : bool, optional
            The verbose option prints the execution time of runCLUE and the number
            of clusters found.

        Modified attributes
        -------------------
        cluster_ids : ndarray
            Contains the cluster_id corresponding to every point.
        is_seed : ndarray
            For every point the value is 1 if the point is a seed or an
            outlier and 0 if it isn't.
        n_clusters : int
            Number of clusters reconstructed.
        cluster_points : ndarray of lists
            Contains, for every cluster, the list of points associated to id.
        points_per_cluster : ndarray
            Contains the number of points associated to every cluster.

        Return
        ------
        None
        """

        if dimensions is None:
            data = self.clust_data.coords
        else:
            data = self._partial_dimension_dataset(dimensions)
        start = time.time_ns()
        if backend == "cpu serial":
            cluster_id_is_seed = cpu_serial.mainRun(self.dc_, self.rhoc, self.outlier, self.ppbin,
                                                    data, self.clust_data.weight, self.kernel,
                                                    self.clust_data.n_dim, block_size, device_id)
        elif backend == "cpu tbb":
            if tbb_found:
                cluster_id_is_seed = cpu_tbb.mainRun(self.dc_, self.rhoc, self.outlier,
                                                     self.ppbin, data, self.clust_data.weight,
                                                     self.kernel, self.clust_data.n_dim, block_size,
                                                     device_id)
            else:
                print("TBB module not found. Please re-compile the library and try again.")

        elif backend == "gpu cuda":
            if cuda_found:
                cluster_id_is_seed = gpu_cuda.mainRun(self.dc_, self.rhoc, self.outlier,
                                                      self.ppbin, data, self.clust_data.weight,
                                                      self.kernel, self.clust_data.n_dim, block_size,
                                                      device_id)
            else:
                print("CUDA module not found. Please re-compile the library and try again.")

        elif backend == "gpu hip":
            if hip_found:
                cluster_id_is_seed = gpu_hip.mainRun(self.dc_, self.rhoc, self.outlier,
                                                     self.ppbin, data, self.clust_data.weight,
                                                     self.kernel, self.clust_data.n_dim, block_size,
                                                     device_id)
            else:
                print("HIP module not found. Please re-compile the library and try again.")

        finish = time.time_ns()
        cluster_ids = np.array(cluster_id_is_seed[0])
        is_seed = np.array(cluster_id_is_seed[1])
        n_clusters = len(np.unique(cluster_ids))

        cluster_points = [[] for _ in range(n_clusters)]
        for i in range(self.clust_data.n_points):
            cluster_points[cluster_ids[i]].append(i)

        points_per_cluster = np.array([len(clust) for clust in cluster_points])

        data = {'cluster_ids': cluster_ids, 'is_seed': is_seed}
        output_df = pd.DataFrame(data)

        self.clust_prop = cluster_properties(n_clusters,
                                             cluster_ids,
                                             is_seed,
                                             np.asarray(cluster_points, dtype=object),
                                             points_per_cluster,
                                             output_df)

        self.elapsed_time = (finish - start)/(10**6)
        if verbose:
            print(f'CLUE executed in {self.elapsed_time} ms')
            print(f'Number of clusters found: {self.clust_prop.n_clusters}')

    def input_plotter(self, plot_title: str='', title_size: float = 16,
                      x_label: str = 'x', y_label: str = 'y', z_label: str = 'z',
                      label_size: float = 16, pt_size: float = 1, pt_colour: str = 'b',
                      grid: bool = True, grid_style: str = '--', grid_size: float = 0.2,
                      x_ticks=None, y_ticks=None, z_ticks=None,
                      **kwargs) -> None:
        """
        Plots the points in input.

        Parameters
        ----------
        plot_title : string, optional
            Title of the plot.
        title_size : float, optional
            Size of the plot title.
        x_label : string, optional
            Label on x-axis.
        y_label : string, optional
            Label on y-axis.
        z_label : string, optional
            Label on z-axis.
        label_size : int, optional
            Fontsize of the axis labels.
        pt_size : int, optional
            Size of the points in the plot.
        pt_colour : string, optional
            Colour of the points in the plot.
        grid : bool, optional
            If true displays grids in the plot.
        grid_style : string, optional
            Style of the grid.
        grid_size : float, optional
            Linewidth of the plot grid.
        x_ticks : list, optional
            List of ticks for the x axis.
        y_ticks : list, optional
            List of ticks for the y axis.
        z_ticks : list, optional
            List of ticks for the z axis.
        kwargs : function objects, optional
            Functions for converting the used coordinates to cartesian coordinates.
            The keywords of the arguments are the coordinates names (x0, x1, ...).

        Modified attributes
        -------------------
        None

        Returns
        -------
        None
        """

        # Convert the used coordinates to cartesian coordiantes
        cartesian_coords = np.copy(self.clust_data.original_coords.T)
        for coord, func in kwargs.items():
            cartesian_coords[int(coord[1])] = func(self.clust_data.original_coords.T)

        if self.clust_data.n_dim == 1:
            plt.scatter(cartesian_coords[0],
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
            if y_ticks is not None:
                plt.yticks(y_ticks)

            plt.show()
        elif self.clust_data.n_dim == 2:
            plt.scatter(cartesian_coords[0],
                        cartesian_coords[1],
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

            plt.show()
        else:
            fig = plt.figure()
            ax_ = fig.add_subplot(projection='3d')
            ax_.scatter(cartesian_coords[0],
                       cartesian_coords[1],
                       cartesian_coords[2],
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

            plt.show()

    def cluster_plotter(self, plot_title: str = '', title_size: float = 16,
                        x_label: str = 'x', y_label: str = 'y', z_label: str = 'z',
                        label_size: float = 16, outl_size: float = 10, pt_size: float = 10,
                        seed_size: float = 25, grid: bool = True, grid_style: str = '--',
                        grid_size: float = 0.2, x_ticks=None, y_ticks=None, z_ticks=None,
                        **kwargs) -> None:
        """
        Plots the clusters with a different colour for every cluster.

        The points assigned to a cluster are printed as points, the seeds
        as stars and the outliers as little grey crosses.

        Parameters
        ----------
        plot_title : string, optional
            Title of the plot
        title_size : float, optional
            Size of the plot title
        x_label : string, optional
            Label on x-axis
        y_label : string, optional
            Label on y-axis
        z_label : string, optional
            Label on z-axis
        label_size : int, optional
            Fontsize of the axis labels
        outl_size : int, optional
            Size of the outliers in the plot
        pt_size : int, optional
            Size of the points in the plot
        seed_size : int, optional
            Size of the seeds in the plot
        grid : bool, optional
            f true displays grids in the plot
        grid_style : string, optional
            Style of the grid
        grid_size : float, optional
            Linewidth of the plot grid
        x_ticks : list, optional
            List of ticks for the x axis.
        y_ticks : list, optional
            List of ticks for the y axis.
        z_ticks : list, optional
            List of ticks for the z axis.
        kwargs : function objects, optional
            Functions for converting the used coordinates to cartesian coordinates.
            The keywords of the arguments are the coordinates names (x0, x1, ...).

        Modified attributes
        -------------------
        None

        Returns
        -------
        None
        """

        # Convert the used coordinates to cartesian coordiantes
        cartesian_coords = np.copy(self.clust_data.original_coords.T)
        for coord, func in kwargs.items():
            cartesian_coords[int(coord[1])] = func(self.clust_data.original_coords.T)

        if self.clust_data.n_dim == 1:
            data = {'x0': cartesian_coords[0],
                    'x1': np.zeros(self.clust_data.n_points),
                    'cluster_ids': self.clust_prop.cluster_ids,
                    'isSeed': self.clust_prop.is_seed}
            df_ = pd.DataFrame(data)

            max_clusterid = max(df_["cluster_ids"])

            df_out = df_[df_.cluster_ids == -1] # Outliers
            plt.scatter(df_out.x0, df_out.x1, s=outl_size, marker='x', color='0.4')
            for i in range(0, max_clusterid+1):
                dfi = df_[df_.cluster_ids == i] # ith cluster
                plt.scatter(dfi.x0, dfi.x1, s=pt_size, marker='.')
            df_seed = df_[df_.isSeed == 1] # Only Seeds
            plt.scatter(df_seed.x0, df_seed.x1, s=seed_size, color='r', marker='*')

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

            plt.show()
        elif self.clust_data.n_dim == 2:
            data = {'x0': cartesian_coords[0],
                    'x1': cartesian_coords[1],
                    'cluster_ids': self.clust_prop.cluster_ids,
                    'isSeed': self.clust_prop.is_seed}
            df_ = pd.DataFrame(data)

            max_clusterid = max(df_["cluster_ids"])

            df_out = df_[df_.cluster_ids == -1] # Outliers
            plt.scatter(df_out.x0, df_out.x1, s=outl_size, marker='x', color='0.4')
            for i in range(0, max_clusterid+1):
                dfi = df_[df_.cluster_ids == i] # ith cluster
                plt.scatter(dfi.x0, dfi.x1, s=pt_size, marker='.')
            df_seed = df_[df_.isSeed == 1] # Only Seeds
            plt.scatter(df_seed.x0, df_seed.x1, s=seed_size, color='r', marker='*')

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

            plt.show()
        else:
            data = {'x0': cartesian_coords[0],
                    'x1': cartesian_coords[1],
                    'x2': cartesian_coords[2],
                    'cluster_ids': self.clust_prop.cluster_ids,
                    'isSeed': self.clust_prop.is_seed}
            df_ = pd.DataFrame(data)

            max_clusterid = max(df_["cluster_ids"])
            fig = plt.figure()
            ax_ = fig.add_subplot(projection='3d')

            df_out = df_[df_.cluster_ids == -1]
            ax_.scatter(df_out.x0, df_out.x1, df_out.x2, s=outl_size, color = 'grey', marker = 'x')
            for i in range(0, max_clusterid+1):
                dfi = df_[df_.cluster_ids == i]
                ax_.scatter(dfi.x0, dfi.x1, dfi.x2, s=pt_size, marker = '.')

            df_seed = df_[df_.isSeed == 1] # Only Seeds
            ax_.scatter(df_seed.x0, df_seed.x1, df_seed.x2, s=seed_size, color = 'r', marker = '*')

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

            plt.show()

    def to_csv(self, output_folder: str, file_name: str) -> None:
        """
        Creates a file containing the coordinates of all the points, their cluster_ids and is_seed.

        Parameters
        ----------
        output_folder : string
            Full path to the desired ouput folder.
        file_name : string
            Name of the file, with the '.csv' suffix.

        Modified attributes
        -------------------
        None

        Returns
        -------
        None
        """

        out_path = output_folder + file_name
        data = {}
        for i in range(self.clust_data.n_dim):
            data['x' + str(i)] = self.clust_data.coords.T[i]
        data['cluster_ids'] = self.clust_prop.cluster_ids
        data['is_seed'] = self.clust_prop.is_seed

        df_ = pd.DataFrame(data)
        df_.to_csv(out_path,index=False)

if __name__ == "__main__":
    c = clusterer(0.4,5,1.)
    c.read_data('./blob.csv')
    c.input_plotter()
    c.run_clue(backend="cpu serial", verbose=True)
    # c.run_clue(backend="cpu tbb", verbose=True)
    # c.run_clue(backend="gpu cuda", verbose=True)
    # c.run_clue(backend="gpu hip", verbose=True)
    c.cluster_plotter()
    # c.to_csv('./','sissa_output_tbb.csv')
    c.list_devices('cpu serial')
    c.list_devices('cpu tbb')
    c.list_devices('gpu cuda')
    c.list_devices('gpu hip')
    c.list_devices()
