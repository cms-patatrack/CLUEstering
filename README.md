# CLUEstering 
The CLUE algorithm is a clustering algorithm written at CERN (https://www.frontiersin.org/articles/10.3389/fdata.2020.591315/full).

The original algorithm was designed to work in 2 dimensions, with the data distributed in parallel layers.
Unlike other clustering algorithms, CLUE takes the coordinates of the points and also their weight, which represents their energy, and calculates the energy density of each point.
This energy density is used to find the seeds for each cluster, their followers and the outliers, which are dismissed as noise.
CLUE takes 4 parameters in input: 
* `dc`, which is the side of the box inside of which the density of a point is calculated;
* `rhoc`, which is the minimum energy density that a point must have to not be considered an outlier,
* `dm`, which is the side of the box inside of which the followers of a point are searched;
* `pointsPerBin`, which is the average number of points that are to be found inside a bin. This value allows to control the size of the bins.

This library generalizes the original algorithm, making it N-dimensional, and turns it into a general purpose algorithm, usable by any user and applicaple to a wider range of applications, in particular outside particle physics.

The C++ code is binded using PyBind11, and the module is created locally during the installation of the library.

In the library is defined the `clusterer` class, which contains the methods for reading the data, running the algorithm, plotting the data both in input and output, and others.  
Outside of the class is also defined the function test_blobs, which takes the number of points and the number of dimensions, and is a way to test quickly the library, producing some N-dimensional blobs.

Below is shown a basic example of how the library can be used:
```
import CLUEstering as clue

clust = clue.clusterer(.4, 2., 1.5)
clust.read_data(clue.test_blobs(1000,2))
clust.run_clue()
clust.cluster_plotter()
```

<p align="center">
  <img width="380" height="380" src="https://raw.githubusercontent.com/cms-patatrack/CLUEstering/main/images/blobwithnoise.png">
</p>

## Installation
### Dependencies
The main dependencies of CLUEstering are [Boost](https://www.boost.org/) (version 1.75.0+) and [Alpaka](https://github.com/alpaka-group/alpaka).
If alpaka is not found, it will be automatically fetched from the official repository, so it's not mandatory to install it manually.

### From source
To install the library, first clone the repository recursively:
```shell
git clone --recursive https://github.com/cms-patatrack/CLUEstering.git
```
alternatively, clone and update the submodules manually:
```shell
git clone https://github.com/cms-patatrack/CLUEstering.git
git submodule update --init --recursive
```
Then, inside the root directory install the library with pip:
```shell
pip install -v .
```
where the `-v` flag is optional but suggested because provides more details during the compilation process.

### From PyPi
The library is also available on the PyPi repository, and can be installed with:
```shell
pip install -v CLUEstering
```

## Heterogeneous backend support with `Alpaka`
Since version `2.0.0` the pybind module is compiled for all the supported backends using the `Alpaka` portability library (https://github.com/alpaka-group/alpaka).  
Currently the supported backends include:
* CPU serial
* CPU parallel using TBB
* NVIDIA GPUs
* AMD GPUs  

The modules are compiled automatically at the moment of installation, and the user can choose the backend to use when running by passing a parameter to the
`run_clue` method.
```
clust.run_clue("cpu serial")
clust.run_clue("cpu tbb")
clust.run_clue("gpu cuda")
clust.run_clue("gpu hip")
```
If no argument is passed, by default the serial backend is used.  
It is possible to list all the available devices with the `list_devices` method. If no argument is passed, the method lists all the devices for all the backends,
but it's also possible to specify the backend whose devices want to be listed.
```
# list devices for all backends
c.list_devices()
# specify the backend
c.list_devices('cpu serial')
c.list_devices('cpu tbb')
c.list_devices('gpu cuda')
c.list_devices('gpu hip')
```

## The `clusterer` class
The `clusterer` class represents a wrapper class around the method `mainRun`, which is binded from `C++` and that is the method that runs the CLUE algorithm.  
When an instance of this class is created, it requires at least three parameters: `dc`, `rhoc` and `dm`. There is a fourth parameter, `pPBin`, which represents the desired average number of points found in each of the bins that the clustering space is divided into. This parameter has a default value of `10`.  
The parameters `dc`, `rhoc` and `dm` must be `floats` or a type convertible to a `float`. `ppBin`, on the other hand, is an `integer`.

The class has several methods:
* `read_data`, which takes the data in input and inizializes the class members. The data can be in the form of list, numpy array, dictionary, string containing the path to a csv file or pandas DataFrame;
* `change_coordinates`, which allows to change the coordinate system used for clustering;
* `change_domains`, which allows to change the domain ranges of any eventual periodic coordinates;
* `choose_kernel`, which allows to change the convolution kernel used when calculating the local density of each point. The default kernel is a flat kernel with parameter `0.5`, but it can be changed to an exponential or gaussian kernel, or a custom kernel, which is user defined and can be any continuous function;
* `run_clue`, which runs the CLUE algorithm;
* `list_devices`, which lists all the available devices for the supported backends;
* `input_plotter`, which plots all the points in input. This method is useful for getting an idea of the shape of the dataset before clustering. In addition to some plot customizations (like the colour or the size of the points, the addition of a grid, the axis labels and so on) it's also possible to pass the functions for the change of coordinates and change the coordinate system used for plotting.
* `cluster_plotter`, which plots the data using a different colour for each cluster. The seeds are indicated by stars and the outliers by small grey crosses.
* `to_csv`, which takes two strings, the first containing the path to a folder and the second containing the desired name for the csv file (also with the .csv suffix) and produces the csv file containing the cluster informations.


## Reading data
Data is read with the `read_data` method.  
For the data to be acceptable, it must contain the values of at least one coordinate for all the points, and their `weights`. The `weights` of the points represent their relative importance in the dataset, so in most cases they can all be set to 1. There are several accepted formats for providing the data:  
* `string`, where the string contains the relative path to a `csv` file. The file must contain at least one column for the coordinates, which must be named as `x*` (`x0`, `x1`, ecc.) and one column for the `weight`
* `pandas.DataFrame`, where the columns for the coordinates must be named `x*` (`x0`, `x1`, ecc.) and one column should contain the `weight`
* `list` or `np.ndarray`, where the coordinate data should be provided as a list of lists (or array of arrays), and the weights inserted in a second, separate list/array
* `dictionary`, where the coordinate data must be contained in lists, each with key `x*` (`x0`, `x1`, ecc.) and the weights in another list with key `weight`

## Generating a test dataset with `test_blobs`
If a user wants to test the library without using real data, they can easily do so using the `test_blobs` method.
`test_blobs` generates a dataset containing any number of gaussian blobs (4 by default), i.e. regular distributions of points distributed gaussianly in a round shape. These blobs can be generated in 2 or 3 dimensions.  
It is possible to customize:
* the number of points in the dataset, through the parameter `n_samples`
* how spread out the points are, through the `mean` and `sigma` parameters
* the spatial span over which these blobs are placed, throught the parameters `x_max` and `y_max`. By modifying this parameters it's possible to make the blobs more crammed or more well separated, thus making it harder or easier to clusterize them

## Change of kernels for the calculation of local density
Since `version 1.3.0` it is possible to choose what kernel to use when calculating the local density of each point. 
The default choices are `gaussian`, `exponential` and `flat`. Each of the kernels require a list of parameters:
* `flat`, takes a single parameter
* `exp`, takes two parameters, the amplitude and the mean.
* `gaus`, takes three parameters, the amplitude, the mean and the standard deviation.   

The default kernel is a flat kernel with parameter `0.5`.
The different kernels can be chosen using the `choose_kernel` method. It is also possible to use a user-defined kernel, by passing it as a function object. Custom kernels require an empty list to be passed as parameters.  
The functions used to define a custom kernel should take three parameters: 
* The distance `dist_ij` between the two points.
* The id `point_id` of the fixed point.
* The id `j` of the neighbouring points.

```
import CLUEstering as clue

clust = clue.clusterer(1., 5., 1.5) # the default kernel is flat(0.5)

# Change to an exponential kernel 
clust.choose_kernel('exp', [1. 1.5])

# Now use a custom kernel, a linear function
clust.choose_kernel('custom', [], lambda x, y, z: 2 * x)
```

## Input and cluster `plotter` methods
The `input_plotter` and `cluster_plotter` methods two plotting methods based on matplotlib. `input_plotter` is intenteded to be used as a way to observe the data before clustering and getting an idea of the expected result, whereas `cluster_plotter` plots the results of the clustering, plotting the points corresponding to the same cluster with the same colour and the outliers as small grey crosses.  
Both methods allow for a wide range of customizations:
* the title of the plot and its size can be changed with the `plot_title` and `title_size` parameters
* the labels of the axis can be changed with the `x_label`, `y_label` and `z_label` parameters
* the size and colour of the points. In the cluster plotter it is possible to change the size of the three classes of points (normal, seed and outlier) singularly
* it's possible to add a grid to the point with the `grid` boolean parameter, and change its style and size with the `grid_style` and `grid_size` parameters
* the ticks on the axis can be changed with the `x_ticks`, `y_ticks` and `z_ticks` parameters

Since `version 1.4.0` both plotting methods can take function objects as kwargs, where the functions represent equations for the change of coordinates, thus allowing to change the coordinate system used for plotting.

```
import CLUEstering as clue

clust = clue.clusterer(1., 5., 1.5)
clust.read_data(data)

# Plot the data in polar coordinates
clust.input_plotter(x0=lambda x: np.sqrt(x[0]**2 + x[1]**2),
                    x1=lambda x: np.arctan2(x[1], x[0]))
clust.run_clue()
```
This is particularly useful when the data has some kind of simmetry, which allows it to be clustered more easily in another coordinate system.

## Writing the results of the clustering to file
The results of the clustering can be saved to a csv file for later analysis, using the method `to_csv`.  
This method taks as parameter the path to the `output_folder` and the `file_name`, where the name of the file must include the `.csv` suffix.  
The data saved on the file include the entire input, so the points' coordinates and weights, as well as their `cluster_id`s, an integer that indicates to what cluster they belong, and `is_seed`, a boolean value indicating whether a point is a seed or not.
```
import CLUEstering as clue

clust = clue.clusterer(1., 5., 1.5)
clust.read_data(data)
clust.run_clue()
clust.to_csv('./output/', 'data_results.csv')
```
