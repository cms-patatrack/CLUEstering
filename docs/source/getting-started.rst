Getting Started
===============

In this section we see how to write minimal code that clusters data using CLUEstering.

Using the C++ Interface
-----------------------

Below is a simple C++ code snippet, which can also be found, along with the CMake file for build, in the ``examples`` folder of the repository:

.. code-block:: cpp

    #include <CLUEstering/CLUEstering.hpp>
    
    int main() {
        // Obtain the queue, which is used for allocations and kernel launches.
        auto queue = clue::get_queue(0u);
    
        // Allocate the points on the host and device.
        clue::PointsHost<2> h_points = clue::read_csv<2>(queue, "path-to-data.csv");
        clue::PointsDevice<2> d_points(queue, h_points.size());
    
        // Define the parameters for the clustering and construct the clusterer.
        const float dc = 20.f, rhoc = 10.f, outlier = 20.f;
        clue::Clusterer<2> algo(queue, dc, rhoc, outlier);
    
        // Launch the clustering
        // The results will be stored in the `clue::PointsHost` object
        algo.make_clusters(queue, h_points, d_points);
        // Read the data from the host points
        auto clusters_indexes = h_points.clusterIndexes();
        auto clusters = clue::get_clusters(h_points);
    }

The first step is to create the ``Queue`` object. A ``Queue`` can be thought of as a ``std::thread`` or as a stream of CUDA/HIP, and represents a queue of operations to be executed on a specific device. The queue will be used to allocate memory and launch kernels on the device.

The ``clue::get_queue`` function provides a convenient way to obtain a queue from a specific device:

.. code-block:: cpp

    auto device = clue::get_device(0u);      // Get the device with index 0
    auto queue = clue::get_queue(device);    // Create a queue from the device
    auto another_queue = clue::Queue(device);// Or call Queue constructor directly

Next, create the containers for the device points. CLUEstering provides ``clue::PointsHost`` and ``clue::PointsDevice`` containers, representing data allocated on the host and on the device. Data is read from a CSV file using ``clue::read_csv``, returning a ``PointsHost`` object, then an empty ``PointsDevice`` is created.

The ``clue::Clusterer`` handles internal allocations and contains the algorithm logic. It requires CLUE algorithm parameters to be passed.

Finally, launch the algorithm with ``make_clusters``, which copies input data from host to device, executes on the device, and copies results back to the host. Results can be read from the host points:

- ``clue::PointsHost::clusterIndexes`` → span of integers for cluster index per point  
- ``clue::PointsHost::isSeed`` → boolean array for cluster seeds  

Compiling the Code
------------------

To compile code using CLUEstering:

1. Include the library headers (via source, submodule, CMake ``FetchContent``, or installed path)  
2. Include/link backend-specific libraries/compilers  
3. Specify the Alpaka backend to use  

Example CMake file:

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.16.0)
    project(CLUEsteringExample)

    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)

    find_package(CLUEstering)
    if(NOT CLUEstering_FOUND)
      message(FATAL_ERROR "CLUEstering not found. Please install it.")
    endif()

    if(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
      add_executable(serial.out main.cpp)
      target_compile_definitions(serial.out PRIVATE ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
    endif()

    if(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
      find_package(TBB REQUIRED)
      add_executable(tbb.out main.cpp)
      target_compile_definitions(tbb.out PRIVATE ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
      target_link_libraries(tbb.out PRIVATE TBB::tbb)
    endif()

    if(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
      find_package(OpenMP REQUIRED)
      add_executable(openmp.out main.cpp)
      target_compile_definitions(openmp.out PRIVATE ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
      target_link_libraries(openmp.out PRIVATE OpenMP::OpenMP_CXX)
    endif()

    if(ALPAKA_ACC_GPU_CUDA_ENABLED)
      include(CheckLanguage)
      check_language(CUDA)
      set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
      add_executable(cuda.out main.cpp)
      target_compile_definitions(cuda.out PRIVATE ALPAKA_ACC_GPU_CUDA_ENABLED)
      set_target_properties(cuda.out PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "50;60;61;62;70;80;90")
    endif()

Using the Python Interface
--------------------------

Minimal example:

.. code-block:: python

    import CLUEstering as clue

    clust = clue.clusterer(1., 5., 1.5)
    clust.read_data(data)
    clust.run_clue()
    clust.to_csv('./output/', 'data_results.csv')

- Parameters can be updated with ``clusterer.set_params``  
- Input data can be pandas DataFrame, Python list, NumPy array, dict, or CSV path  
- ``run_clue`` accepts backends: ``"cpu serial"``, ``"cpu tbb"``, ``"cpu openmp"``, ``"gpu cuda"``, ``"gpu hip"``  

Access results:

.. code-block:: python

    clust.n_clusters         # number of clusters
    clust.n_seeds            # number of seeds
    clust.clusters           # list of clusters
    clust.cluster_ids        # array of cluster indexes
    clust.is_seed            # boolean array for seeds
    clust.cluster_points     # nested arrays with cluster points
    clust.points_per_cluster # array with number of points per cluster
    clust.output_df          # dataframe with input + clustering results

Plotting:

- ``clusterer.input_plotter`` → plots input data  
- ``clusterer.cluster_plotter`` → plots clustered data  

.. image:: images/docs/getting-started/input-plotter.png
   :alt: Data plotted with the input plotter
   :width: 500px

.. image:: images/docs/getting-started/output-plotter.png
   :alt: Data plotted with the cluster plotter
   :width: 500px

Data Format
-----------

### CSV Files

Each row contains coordinates followed by weight:

.. .. code-block:: csv

..     x,y,z,weight
..     -9.95,5.17,0.15,1.0
..     -9.43,5.68,0.15,1.0
..     -11.0,7.29,0.15,1.0
..     -10.7,-4.37,0.15,1.0
..     3.5,4.48,0.15,1.0
..     3.0,2.94,0.15,1.0
..     -9.97,4.04,0.15,1.0
..     -10.36,-4.39,0.15,1.0

### Passing Data to ``PointsHost`` and ``PointsDevice``

- Host and device containers expect **SoA format** (coordinates of all points in each dimension adjacent in memory).  
- Data from external containers can be passed via pointers, ``std::span``, or any contiguous container.  
- Two buffer format: coordinates+weights + results  
- Four buffer format: coordinates, weights + results

Python SoA / AoS example:

.. code-block:: python

    data_aos = [[x0, y0, z0], [x1, y1, z1], [x2, y2, z2], [w0, w1, w2]]
    data_soa = [[[x0, x1, x2], [y0, y1, y2], [z0, z1, z2]], [w0, w1, w2]]
