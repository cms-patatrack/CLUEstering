Batched Clustering
==================

Many applications need to cluster a large number of small, independent datasets with the same
parameters: the events of a particle physics experiment, the frames of a video, the images of a
dataset. Clustering them one at a time is inefficient, especially on GPUs, because each dataset is
too small to fill the device and the cost of launching the kernels and allocating the buffers is
paid once per dataset.

Batched clustering solves this problem by running the clustering on all the datasets, called *batch items*,
with a single call. The points of all the batch items are stored in the same points container, and
the clusterer is told how many points belong to each item. Each batch item is clustered
independently: points belonging to different items are never neighbours of each other, so the
result is the same as clustering each item separately.

Preparing the input
-------------------

The input of a batched clustering is:

- a single points container (``clue::PointsHost`` and/or ``clue::PointsDevice`` in C++, a single
  dataset in Python) containing the points of all the batch items;
- an array of unsigned 32-bit integers with the number of points in each batch item.

The points of each batch item must be stored **contiguously** and in the same order as the sizes
array: the first ``sizes[0]`` points belong to the first item, the following ``sizes[1]`` points to
the second, and so on. The sum of the sizes must be equal to the total number of points.

For example, three events with 4, 2 and 3 points are laid out as::

    point index:  0  1  2  3 | 4  5 | 6  7  8
    batch item:   0  0  0  0 | 1  1 | 2  2  2

    sizes = [4, 2, 3]

All the batch items are clustered with the same parameters (density radius, minimum density,
outlier distance, seeding distance, kernel and metric). The tiles are built from the coordinate
extremes of the whole input, so batch items covering very different regions of space are supported,
but they share the same tile grid.

Using the C++ interface
-----------------------

The batched overloads of ``clue::Clusterer::make_clusters`` take the sizes of the batch items as a
``std::span<const uint32_t>`` after the points. As for the non-batched version, the points can be
passed on the host, on the device, or both:

.. code-block:: cpp

    #include <CLUEstering/CLUEstering.hpp>
    #include <vector>

    int main() {
        auto queue = clue::get_queue(0u);

        // The points of all the batch items, stored contiguously
        clue::PointsHost<2> h_points = clue::read_csv<2, float>(queue, "batched-data.csv");
        clue::PointsDevice<2> d_points(queue, h_points.size());

        // Number of points in each batch item. The sum must be equal to h_points.size()
        std::vector<uint32_t> batch_sizes = {1024, 512, 2048, 700};

        const auto density_radius = 1.3f, min_density = 10.f;
        clue::Clusterer<2> algo(queue, density_radius, min_density);

        // Cluster all the batch items with a single call
        algo.make_clusters(queue, h_points, d_points, batch_sizes);

        // Cluster index of each point
        auto cluster_indexes = h_points.clusterIndexes();
        // Map from each batch item to the clusters it contains
        auto batch_to_clusters = algo.getSampleAssociations(queue, h_points);
    }

The available overloads are:

- ``make_clusters(queue, h_points, d_points, sizes)``: the data is copied from the host to the
  device, clustered, and the results are copied back to the host;
- ``make_clusters(queue, h_points, sizes)``: as above, but the device buffers are allocated
  internally;
- ``make_clusters(queue, d_points, sizes)``: the data is already on the device and the results
  are left there. This is the right choice when the points are produced by a previous step running
  on the device.

All of them accept a distance metric and a convolutional kernel as optional arguments, exactly like
the non-batched ``make_clusters``.

Reading the results
~~~~~~~~~~~~~~~~~~~

The cluster indexes are **global**: they are unique across the whole batch, not restarted for each
batch item. The clusters are numbered by batch item, so all the clusters of the first item come
before those of the second, and so on. Outliers have cluster index ``-1``, as usual.

``clue::Clusterer::getSampleAssociations`` returns an ``AssociationMap`` whose keys are the batch
items and whose values are the indexes of the clusters they contain. It can be used to iterate over
the clusters of a single item:

.. code-block:: cpp

    auto batch_to_clusters = algo.getSampleAssociations(queue, h_points);
    for (auto item = 0; item < static_cast<int>(batch_to_clusters.size()); ++item) {
        for (auto cluster : batch_to_clusters[item]) {
            // `cluster` is the global index of a cluster of batch item `item`
        }
    }

The overload taking ``clue::PointsDevice`` returns an ``AssociationMap`` allocated on the device.
``clue::get_clusters`` works as for the non-batched case and returns the map from each cluster to
its points, using the global cluster indexes.

.. note::

   The sample associations are stored in the clusterer and are overwritten by the next call to
   ``make_clusters``.

Using the Python interface
--------------------------

In Python the sizes of the batch items are passed to ``run_clue``, ``fit`` or ``fit_predict``
through the ``batch_sample_sizes`` argument, as a NumPy array of ``uint32``:

.. code-block:: python

    import numpy as np
    import CLUEstering as clue

    # data contains the points of all the batch items, stored contiguously
    batch_sizes = np.array([1024, 512, 2048, 700], dtype=np.uint32)

    c = clue.clusterer(1.3, 10.)
    c.read_data(data)
    c.run_clue(batch_sample_sizes=batch_sizes)

    labels = c.cluster_ids

As in C++, the labels are global over the whole batch. The batch item of each point can be
recovered from the sizes, for example to compute the number of clusters of each item:

.. code-block:: python

    batch_ids = np.repeat(np.arange(len(batch_sizes)), batch_sizes)
    for item in range(len(batch_sizes)):
        item_labels = labels[batch_ids == item]
        n_clusters = len(np.unique(item_labels[item_labels != -1]))

Performance considerations
--------------------------

- Batching is most effective when the batch items are small compared to the size of the device.
  For a single large dataset the non-batched ``make_clusters`` is just as fast.
- The kernels distribute the work over a two-dimensional grid of *batch items* × *points of the
  largest item*. When the sizes are very different, the threads assigned to the smaller items are
  idle, so batches of items with similar sizes use the device more efficiently.
- The ``Clusterer`` reuses its internal buffers between calls, so when processing a stream of batches
  it is convenient to construct it once and call ``make_clusters`` repeatedly.
- On parallel backends the order in which clusters are found is not deterministic. The numbering of
  the clusters within a batch item can therefore change from run to run, while the clusters
  themselves and their grouping by batch item do not.
