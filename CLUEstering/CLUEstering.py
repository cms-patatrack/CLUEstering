"""
Density based clustering algorithm developed at CERN.

Classes:

    clusterer

Functions:
    normalize_data(int/float/list, int/float, int/float)
    test_blobs() ->
    read_data(object)
    change_coordinates(object)
    choose_kernel(object)
    run_clue(object)
    input_plotter()
    cluster_plotter()
    to_csv(object)
"""

import random as rnd
from math import sqrt
from math import pi
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
import CLUEsteringCPP as Algo

def normalize_data(data, mean, std):
    """
    Function that normalizes the data using the transformation x' = (x - mu)/sigma

    Parameters:
    data (int, float, list): The data value or list of values to normalize
    mean (int, float): Mean of the data values
    std: Standard deviations of the data values
    """

    data -= mean
    if std != 0:
        data /= std

    return data

def test_blobs(n_samples, n_dim, n_blobs=4, mean=0, sigma=0.5, x_max=30, y_max=30):
    """
    Returns a test dataframe containing randomly generated 2-dimensional or 3-dimensional blobs.

    Parameters:
    n_samples (int): The number of points in the dataset.
    Ndim (int): The number of dimensions.
    n_blobs (int): The number of blobs that should be produced. By default it is set to 4.
    mean (float): The mean of the gaussian distribution of the z values.
    sigma (float): The standard deviation of the gaussian distribution of the z values.
    x_max (float): Limit of the space where the blobs are created in the x direction.
    y_max (float): Limit of the space where the blobs are created in the y direction.
    """

    try:
        if x_max < 0. or y_max < 0.:
            raise ValueError('Error: wrong parameter value\nx_max and y_max must be positive')
        if n_blobs < 0:
            raise ValueError('Error: wrong parameter value\nThe number of blobs must be positive')
        if mean < 0. or sigma < 0.:
            raise ValueError('Error: wrong parameter value\nThe mean and sigma of the blobs \
                              cannot be negative')

        centers = []
        if n_dim == 2:
            data = {'x0': [], 'x1': [], 'weight': []}
            for i in range(n_blobs):
                centers.append([x_max * rnd.random(),
                                y_max * rnd.random()])
            blob_data = make_blobs(n_samples=n_samples, centers=np.array(centers))[0]
            for i in range(n_samples):
                data['x0'] += [blob_data[i][0]]
                data['x1'] += [blob_data[i][1]]
                data['weight'] += [1]

            return pd.DataFrame(data)
        if n_dim == 3:
            data = {'x0': [], 'x1': [], 'x2': [], 'weight': []}
            sqrt_samples = int(sqrt(n_samples))
            z_values = np.random.normal(mean,sigma,sqrt_samples)
            for i in range(n_blobs):
                centers.append([x_max * rnd.random(),  # the centers are 2D because we
                                y_max * rnd.random()]) # create them for each layer
            for value in z_values: # for every z value, a layer is generated.
                blob_data = make_blobs(n_samples=sqrt_samples, centers=np.array(centers))[0]
                for i in range(sqrt_samples):
                    data['x0'] += [blob_data[i][0]]
                    data['x1'] += [blob_data[i][1]]
                    data['x2'] += [value]
                    data['weight'] += [1]

            return pd.DataFrame(data)

        # If it gets to the bottom we raise the exception
        raise ValueError('Error: wrong number of dimensions\nBlobs can only be generated \
                          in 2 or 3 dimensions')
    except ValueError as ve_:
        print(ve_)
        sys.exit()

class clusterer:
    """
    Class representing a wrapper for the methods using in the process of clustering using \
    the CLUE algorithm.
    """

    def __init__(self, dc_, rhoc_, outlier_, ppbin=10):
        try:
            if float(dc_) != dc_:
                raise ValueError('Error: wrong parameter type\nThe dc parameter must be a float')
            self.dc = dc_
            if float(rhoc_) != rhoc_:
                raise ValueError('Error: wrong parameter type\nThe rhoc parameter must be a float')
            self.rhoc = rhoc_
            if float(outlier_) != outlier_:
                raise ValueError('Error: wrong parameter type\n \
                                  The outlier parameter must be a float')
            self.outlier = outlier_
            if not isinstance(ppbin, (int)):
                raise ValueError('Error: wrong parameter type\nThe ppbin parameter must be a int')
            self.ppbin = ppbin
        except ValueError as ve_:
            print(ve_)
            sys.exit()
        self.kernel = Algo.flatKernel(0.5)

        # Initialize attributes
        ## Data containers
        self.coords = []
        self.original_coords = []
        self.weight = []
        self.domain_ranges = []
        self.n_dim = 0
        self.n_points = 0

        ## Output attributes
        self.n_clusters = 0
        self.cluster_ids = []
        self.is_seed = []
        self.cluster_points = []
        self.points_per_cluster = []
        self.output_df = pd.DataFrame(None)
        self.elapsed_time = 0.

    def read_data(self, input_data, **kwargs):
        """
        Reads the data in input and fills the class members containing the coordinates
        of the points, the energy weight, the number of dimensions and the number of points.

        Parameters:
        input_data (pandas dataframe): The dataframe should contain one column for every
        coordinate, each one called 'x*', and one column for the weight.
        input_data (string): The string should contain the full path to a csv file containing
        the data.
        input_data (list or numpy array): The list or numpy array should contain a list of \
        lists for the
        coordinates and a list for the weight.
        kwargs (tuples): Tuples corresponding to the domain of any periodic variables. The keyword
        should be the keyword of the corrispoding variable.
        """

        # numpy array
        if isinstance(input_data, np.ndarray):
            try:
                if len(input_data) < 2:
                    raise ValueError('Error: inadequate data\nThe data must contain \
                                      at least one coordinate and the energy.')
                self.coords = input_data[:-1]
                self.weight = input_data[-1]
                if len(input_data[:-1]) > 10:
                    raise ValueError('Error: inadequate data\nThe maximum number of \
                                      dimensions supported is 10')
                self.n_dim = len(self.coords)
                self.n_points = self.weight.size

                # Save the original coordinates before any normalization
                self.original_coords = np.copy(self.coords)

                # Calculate mean and standard deviations in all the coordinates
                means = np.zeros(shape=(self.n_dim, 1))
                st_devs = np.zeros(shape=(self.n_dim, 1))
                for dim in range(self.n_dim):
                    means[dim] = np.mean(self.coords[dim])
                    st_devs[dim] = np.std(self.coords[dim])

                # Normalize all the coordinates as x'_j = (x_j - mu_j) / sigma_j
                for dim in range(self.n_dim):
                    self.coords[dim] = normalize_data(self.coords[dim],
                                                     means[dim],
                                                     st_devs[dim])

                # Construct the domains of all the coordinates
                empty_domain = Algo.domain_t()
                self.domain_ranges = [empty_domain for i in range(self.n_dim)]
                for coord, domain in kwargs.items():
                    self.domain_ranges[int(coord[1])] = Algo.domain_t(
                                                            normalize_data(domain[0],
                                                                           means[int(coord[1])],
                                                                           st_devs[int(coord[1])]),
                                                            normalize_data(domain[1],
                                                                           means[int(coord[1])],
                                                                           st_devs[int(coord[1])])
                                                                     )
            except ValueError as ve_:
                print(ve_)
                sys.exit()

        # lists
        if isinstance(input_data, (list)):
            try:
                if len(input_data) < 2:
                    raise ValueError('Error: inadequate data\nThe data must contain \
                                      at least one coordinate and the energy.')
                self.coords = np.array(input_data[:-1])
                self.weight = np.array(input_data[-1])
                if len(input_data[:-1]) > 10:
                    raise ValueError('Error: inadequate data\nThe maximum number of \
                                      dimensions supported is 10')
                self.n_dim = len(self.coords)
                self.n_points = self.weight.size

                # Save the original coordinates before any normalization
                self.original_coords = np.copy(self.coords)

                # Calculate mean and standard deviations in all the coordinates
                means = np.zeros(shape=(self.n_dim, 1))
                st_devs = np.zeros(shape=(self.n_dim, 1))
                for dim in range(self.n_dim):
                    means[dim] = np.mean(self.coords[dim])
                    st_devs[dim] = np.std(self.coords[dim])

                # Normalize all the coordinates as x'_j = (x_j - mu_j) / sigma_j
                for dim in range(self.n_dim):
                    self.coords[dim] = normalize_data(self.coords[dim],
                                                     means[dim],
                                                     st_devs[dim])

                # Construct the domains of all the coordinates
                empty_domain = Algo.domain_t()
                self.domain_ranges = [empty_domain for i in range(self.n_dim)]
                for coord, domain in kwargs.items():
                    self.domain_ranges[int(coord[1])] = Algo.domain_t(
                                                            normalize_data(domain[0],
                                                                           means[int(coord[1])],
                                                                           st_devs[int(coord[1])]),
                                                            normalize_data(domain[1],
                                                                           means[int(coord[1])],
                                                                           st_devs[int(coord[1])])
                                                                     )
            except ValueError as ve_:
                print(ve_)
                sys.exit()

        # path to .csv file or pandas dataframe
        if isinstance(input_data, (str, pd.DataFrame, dict)):
            if isinstance(input_data, (str)):
                try:
                    if input_data[-3:] != 'csv':
                        raise ValueError('Error: wrong type of file\nThe file is not a csv file.')
                    df_ = pd.read_csv(input_data)

                except ValueError as ve_:
                    print(ve_)
                    sys.exit()
            if isinstance(input_data, (pd.DataFrame)):
                try:
                    if len(input_data.columns) < 2:
                        raise ValueError('Error: inadequate data\nThe data must contain \
                                          at least one coordinate and the energy.')
                    df_ = input_data
                except ValueError as ve_:
                    print(ve_)
                    sys.exit()
            if isinstance(input_data, (dict)):
                try:
                    if len(input_data.keys()) < 2:
                        raise ValueError('Error: inadequate data\nThe data must contain \
                                          at least one coordinate and the energy.')
                    if not 'weight' in input_data.keys():
                        raise ValueError('Error: inadequate data\nThe data must contain \
                                          values for the weights.')

                    df_ = pd.DataFrame(input_data)
                except ValueError as ve_:
                    print(ve_)
                    sys.exit()

            try:
                if not 'weight' in df_.columns:
                    raise ValueError('Error: inadequate data\nThe input dataframe must \
                                      contain a weight column.')

                coordinate_columns = [col for col in df_.columns if col[0] == 'x']
                if len(coordinate_columns) > 10:
                    raise ValueError('Error: inadequate data\nThe maximum number of \
                                      dimensions supported is 10')
                self.n_dim = len(coordinate_columns)
                self.n_points = len(df_.index)
                self.coords = np.zeros(shape=(self.n_dim, self.n_points))
                for dim in range(self.n_dim):
                    self.coords[dim] = np.array(df_.iloc[:,dim])
                self.weight = df_['weight']

                # Save the original coordinates before any normalization
                self.original_coords = np.copy(self.coords)

                # Calculate mean and standard deviations in all the coordinates
                means = np.zeros(shape=(self.n_dim, 1))
                st_devs = np.zeros(shape=(self.n_dim, 1))
                for dim in range(self.n_dim):
                    means[dim] = np.mean(self.coords[dim])
                    st_devs[dim] = np.std(self.coords[dim])

                # Normalize all the coordinates as x'_j = (x_j - mu_j) / sigma_j
                for dim in range(self.n_dim):
                    self.coords[dim] = normalize_data(self.coords[dim],
                                                     means[dim],
                                                     st_devs[dim])

                # Construct the domains of all the coordinates
                empty_domain = Algo.domain_t()
                self.domain_ranges = [empty_domain for i in range(self.n_dim)]
                for coord, domain in kwargs.items():
                    self.domain_ranges[int(coord[1])] = Algo.domain_t(
                                                            normalize_data(domain[0],
                                                                           means[int(coord[1])],
                                                                           st_devs[int(coord[1])]),
                                                            normalize_data(domain[1],
                                                                           means[int(coord[1])],
                                                                           st_devs[int(coord[1])])
                                                                     )
            except ValueError as ve_:
                print(ve_)
                sys.exit()

    def change_coordinates(self, **kwargs):
        """
        Change the coordinate system

        Parameters:
        kwargs (function objects): The functions for the change of coordinates. \
        The keywords of the arguments are the coordinates names (x0, x1, ...)
        """

        # Change the coordinate system
        for coord, func in kwargs.items():
            self.coords[int(coord[1])] = func(self.original_coords)

    def choose_kernel(self, choice, parameters=None, function = lambda: 0):
        """
        Changes the kernel used in the calculation of local density. The default kernel \
        is a flat kernel with parameter 0.5

        Parameters:
        choice (string): The type of kernel that you want to choose (flat, exp, gaus or custom).
        parameters (list or np.array): List of the parameters needed by the kernels. \
        The flat kernel requires one,
        the exponential requires two (amplutude and mean), the gaussian requires three \
        (amplitude, mean and standard deviation)
        and the custom doesn't require any, so an empty list should be passed.
        function (function object): Function that should be used as kernel when the \
        custom kernel is chosen.
        """

        try:
            if choice == "flat":
                if len(parameters) != 1:
                    raise ValueError('Error: wrong number of parameters\nThe flat kernel \
                                      requires 1 parameter')
                self.kernel = Algo.flatKernel(parameters[0])
            elif choice == "exp":
                if len(parameters) != 2:
                    raise ValueError('Error: wrong number of parameters\nThe exponential \
                                      kernel requires 2 parameters')
                self.kernel = Algo.exponentialKernel(parameters[0], parameters[1])
            elif choice == "gaus":
                if len(parameters) != 3:
                    raise ValueError('Error: wrong number of parameters\nThe gaussian \
                                      kernel requires 3 parameters')
                self.kernel = Algo.gaussianKernel(parameters[0], parameters[1], parameters[2])
            elif choice == "custom":
                if len(parameters) != 0:
                    raise ValueError('Error: wrong number of parameters\nCustom kernels \
                                      requires 0 parameters')
                self.kernel = Algo.customKernel(function)
            else:
                raise ValueError('Error: invalid kernel\nThe allowed choices for the \
                                  kernels are: flat, exp, gaus and custom')
        except ValueError as ve_:
            print(ve_)
            sys.exit()

    def run_clue(self, verbose=False):
        """
        Executes the CLUE clustering algorithm.

        Parameters:
        verbose (bool): The verbose option prints the execution time of runCLUE and \
        the number of clusters found

        Output:
        self.cluster_ids (list): Contains the cluster_id corresponding to every point.
        self.isSeed (list): For every point the value is 1 if the point is a seed or \
        an outlier and 0 if it isn't.
        self.NClusters (int): Number of clusters reconstructed.
        self.clusterPoints (list): Contains, for every cluster, the list of points \
        associated to id.
        self.pointsPerCluster (list): Contains the number of points associated to every cluster
        """

        start = time.time_ns()
        cluster_id_is_seed = Algo.mainRun(self.dc,self.rhoc,self.outlier,self.ppbin,
                                          self.domain_ranges,self.kernel,self.coords,
                                          self.weight,self.n_dim)
        finish = time.time_ns()
        self.cluster_ids = np.array(cluster_id_is_seed[0])
        self.is_seed = np.array(cluster_id_is_seed[1])
        self.n_clusters = len(np.unique(self.cluster_ids))

        cluster_points = [[] for i in range(self.n_clusters)]
        for i in range(self.n_points):
            cluster_points[self.cluster_ids[i]].append(i)

        self.cluster_points = cluster_points
        self.points_per_cluster = np.array([len(clust) for clust in cluster_points])

        data = {'cluster_ids': self.cluster_ids, 'is_seed': self.is_seed}
        self.output_df = pd.DataFrame(data)

        self.elapsed_time = (finish - start)/(10**6)
        if verbose:
            print('CLUE run in ' + str(self.elapsed_time) + ' ms')
            print('Number of clusters found: ', self.n_clusters)

    def input_plotter(self, plot_title='', title_size=16,
                      x_label='x', y_label='y', z_label='z', label_size=16,
                      pt_size=1, pt_colour='b',
                      grid=True, grid_style='--', grid_size=0.2,
                      x_ticks=None, y_ticks=None, z_ticks=None,
                      **kwargs):
        """
        Plots the the points in input.
        Parameters:
        plot_title (string): Title of the plot
        title_size (float): Size of the plot title
        x_label (string): Label on x-axis
        y_label (string): Label on y-axis
        z_label (string): Label on z-axis
        label_size (int): Fontsize of the axis labels
        pt_size (int): Size of the points in the plot
        pt_colour (string): Colour of the points in the plot
        grid (bool): If true displays grids in the plot
        grid_style (string): Style of the grid
        grid_size (float): Linewidth of the plot grid
        kwargs (function objects): Functions for converting the used coordinates \
        to cartesian coordinates. The keywords of the arguments \
        are the coordinates names (x0, x1, ...)
        """

        # Convert the used coordinates to cartesian coordiantes
        cartesian_coords = np.copy(self.original_coords)
        for coord, func in kwargs.items():
            cartesian_coords[int(coord[1])] = func(self.original_coords)

        if self.n_dim == 2:
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
        if self.n_dim >= 3:
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

    def cluster_plotter(self, plot_title='', title_size=16,
                        x_label='x', y_label='y', z_label='z', label_size=16,
                        outl_size=10, pt_size=10, seed_size=25,
                        grid=True, grid_style='--', grid_size=0.2,
                        x_ticks=None, y_ticks=None, z_ticks=None,
                        **kwargs):
        """
        Plots the clusters with a different colour for every cluster.

        The points assigned to a cluster are printed as points, the seeds \
        as stars and the outliers as little grey crosses.

        Parameters:
        plot_title (string): Title of the plot
        title_size (float): Size of the plot title
        x_label (string): Label on x-axis
        y_label (string): Label on y-axis
        z_label (string): Label on z-axis
        label_size (int): Fontsize of the axis labels
        outl_size (int): Size of the outliers in the plot
        pt_size (int): Size of the points in the plot
        seed_size (int): Size of the seeds in the plot
        grid (bool): If true displays grids in the plot
        grid_style (string): Style of the grid
        grid_size (float): Linewidth of the plot grid
        kwargs (function objects): Functions for converting the used \
        coordinates to cartesian coordinates. The keywords of the arguments \
        are the coordinates names (x0, x1, ...)
        """

        # Convert the used coordinates to cartesian coordiantes
        cartesian_coords = np.copy(self.original_coords)
        for coord, func in kwargs.items():
            cartesian_coords[int(coord[1])] = func(self.original_coords)

        if self.n_dim == 2:
            data = {'x0': cartesian_coords[0],
                    'x1': cartesian_coords[1],
                    'cluster_ids': self.cluster_ids,
                    'isSeed': self.is_seed}
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
        if self.n_dim == 3:
            data = {'x0': cartesian_coords[0],
                    'x1': cartesian_coords[1],
                    'x2': cartesian_coords[2],
                    'cluster_ids': self.cluster_ids,
                    'isSeed': self.is_seed}
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

    def to_csv(self, output_folder, file_name):
        """
        Creates a file containing the coordinates of all the points, their cluster_ids and isSeed.

        Parameters:
        outputFolder (string): Full path to the desired ouput folder.
        fileName (string): Name of the file, with the '.csv' suffix.
        """

        out_path = output_folder + file_name
        data = {}
        for i in range(self.n_dim):
            data['x' + str(i)] = self.coords[i]
        data['cluster_ids'] = self.cluster_ids
        data['is_seed'] = self.is_seed

        df_ = pd.DataFrame(data)
        df_.to_csv(out_path,index=False)

if __name__ == "__main__":
    # Test 2-dimensional blobs
    a = clusterer(0.3, 5, 1.2)
    a.read_data(test_blobs(1000, 2))
    a.run_clue()
    a.cluster_plotter()

    # Test points with angles distributed at opposite extremes of the domain
    # This test assures that the code works for data with periodic coordinates
    b = clusterer(0.2, 1, 1.5)
    b.read_data('../opposite_angles.csv', x1=(-pi, pi))
    b.input_plotter()
    b.run_clue()
    b.cluster_plotter(x0=lambda x: x[0]*np.cos(x[1]),
                     x1=lambda x: x[0]*np.sin(x[1]))
    b.cluster_plotter()

    # Create circles dataset
    circ_data, labels = make_circles(n_samples=1000, factor=0.4)
    df = {'x0': [], 'x1': [], 'weight': []}
    for j in range(1000):
        df['x0'] += [circ_data[j][0]]
        df['x1'] += [circ_data[j][1]]
        df['weight'] += [1]
    df = pd.DataFrame(df)

    # Convert it to polar coordinates
    new_data = {}
    new_data['x0'] = np.sqrt(df['x0']**2 + df['x1']**2)
    new_data['x1'] = np.arctan2(df['x1'], df['x0'])
    new_data['weight'] = [1 for k in range(len(new_data['x0']))]
    new_df = pd.DataFrame(new_data)

    # Test circles dataset
    c = clusterer(1, 5, 5)
    c.read_data(new_df, x1=(-pi, pi))
    c.input_plotter(x0=lambda x: x[0]*np.cos(x[1]),
                   x1=lambda x: x[0]*np.sin(x[1]))
    c.run_clue()
    c.cluster_plotter(x0=lambda x: x[0]*np.cos(x[1]),
                     x1=lambda x: x[0]*np.sin(x[1]))
