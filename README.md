# CLUEstering 
The CLUE algorithm is a clustering algorithm written at CERN.

The original algorithm was designed to work in 2 dimensions, with the data distributed in parallel layers.
Unlike other clustering algorithms, CLUE takes the coordinates of the points and also their weight, which represents their energy, and calculater the energy density of each point.
This energy density is used to find the seeds for each cluster, their followers and the outliers, which are dismissed as noise.
CLUE takes 4 parameters in input: 
* dc_, which is the side of the box inside of which the density of a point is calculated;
* rhoc, which is the minimum energy density that a point must have to not be considered an outlier,
* outlierDeltaFactor, that multiplied by dc_ gives dm_, the side of the box inside of which the followers of a point are searched;
* pointsPerBin, which is the average number of points that are to be found inside a bin. This value allows to control the size of the bins.

This library generalizes the original algorithm, making it N-dimensional.

<p align="center">
    <img src="./plot2d.png" width="300" height="300"> <img src="./plot3d.png" width="300" height="300">
</p>

The C++ code is binded using PyBind11, and the module is created locally during the installation of the library.

In this library is defined the clusterer class. The constructor takes the four parameters, dc_, rhoc, outlierDeltaFactor and pPBin. Passing pPBin is optional since by default it is initialized to 10.

The class has several methods:
* readData, which takes the data in input and inizializes the class members. The data can be in the form of lists, numpy arrays, string containing the path to a csv file or pandas DataFrame;
* runCLUE, which takes no parameters and runs the CLUE algorithm;
* inputPlotter, which takes no parameters and plots the data in input;
* clusterPlotter, which takes no parameters and plots the data using a different colour for each cluster. The seeds are indicated by stars and the outliers by small grey crosses.
* toCSV, which takes two strings, the first containing the path to a folder and the second containing the desired name for the csv file (also with the .csv suffix) and produces the csv file containing the cluster informations.

Outside of the class is also defined the function makeBlobs, which takes the number of points and the number of dimensions, and is a way to test quickly the library, producing some N-dimensional blobs.

An expample of how the library should be used is:
```
import CLUEstering as c

clust = c.clusterer(1,5,1.5)
clust.readData(c.makeBlobs(1000,2))
clust.runCLUE()
clust.clusterPlotter()
```
<p align="center">
    <img src="./blobwithnoise.png" width="400" height="400"> 
</p>