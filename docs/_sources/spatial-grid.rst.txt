Reusing the Spatial Grid
========================

To find the neighbours of the points quickly, the ``Clusterer`` divides the clustering space into
a regular grid of cells and associates each cell with the points it contains. Other algorithms
running on the same points after the clustering, for instance a second neighbour search or a
pattern recognition step, may need the same structure. ``clue::Clusterer::getSpatialGrid`` gives
read-only access to the grid built in the last clustering run, so that it can be used or copied
without computing it again.

This feature is available only in the C++ interface.

Getting the grid
----------------

After ``make_clusters``, ``getSpatialGrid`` returns a ``clue::SpatialGrid``, a small struct that
describes the grid and refers to the associations between cells and points:

.. code-block:: cpp

    clue::Clusterer<2> algo(density_radius, min_density);
    algo.make_clusters(queue, h_points);

    const auto grid = algo.getSpatialGrid();

The struct contains:

- ``cells``: a reference to a ``clue::AssociationMap`` on the device, mapping each cell to the
  indexes of its points, in the order in which they were given to ``make_clusters``;
- ``min``, ``max``: the edges of the grid along each dimension;
- ``cell_size``, ``n_cells``: the size and the number of the cells along each dimension;
- ``periodic``: whether each dimension is periodic;
- ``batch_size``: the number of batch items in the last run, 1 for a run that is not batched.

The geometry is copied into the struct, while the cells are not: ``getSpatialGrid`` performs no
copy of the associations and no allocation. Calling it before the first clustering run throws
``std::runtime_error``.

The cell size is chosen internally from the clustering parameters and the number of points, so it
can change from one run to the next. Code using the grid should always read the geometry from the
struct.

Layout of the cells
-------------------

The associations are stored in CSR form, in two device buffers:

- ``keys``, with one more element than the number of cells: the points of cell ``c`` are the
  elements of ``values`` from ``keys[c]`` to ``keys[c + 1] - 1``;
- ``values``, with the indexes of the points, grouped by cell.

The cell of a point along dimension ``d`` is ``(coord - min[d]) / cell_size[d]``, truncated and
clamped to ``[0, n_cells[d] - 1]``, so the points outside of ``[min[d], max[d])`` are in the
border cells. The global index of a cell is row-major, with the last dimension varying fastest.
For example, in two dimensions the cell with indexes ``(i, j)`` has global index
``i * n_cells[1] + j``.

A periodic dimension covers exactly one period, ``[0, period)``, and the coordinates are wrapped
into that interval before computing their cell.

In a batched run all the batch items share the same grid, and their cells are stored one after
the other: the cells of item ``e`` have global indexes from ``e * total`` to
``(e + 1) * total - 1``, where ``total`` is the product of ``n_cells``.

.. note::

   A point lying exactly on the edge between two cells, within rounding error, can be in either of
   the two cells, depending on how the compiler evaluates the division. Code that computes the cell
   of a point again with this geometry may find the neighbouring cell for such points.

Copying the cells
-----------------

The buffers of the association map can be copied with a plain ``memcpy``. Their size is given by
``cells.extents()``, because the buffers can be larger than the data they contain:

.. code-block:: cpp

    const auto extents = grid.cells.extents();
    const auto containers = grid.cells.extract();

    std::vector<int32_t> keys(extents.keys + 1);
    std::vector<int32_t> values(extents.values);
    alpaka::memcpy(queue,
                   alpaka::createView(clue::host, keys.data(), extents.keys + 1),
                   alpaka::createView(queue, containers.keys.data(), extents.keys + 1));
    alpaka::memcpy(queue,
                   alpaka::createView(clue::host, values.data(), extents.values),
                   alpaka::createView(queue, containers.values.data(), extents.values));
    alpaka::wait(queue);

The same copy works into device buffers, for instance into the equivalent structure of another
framework, without going through the host.

Lifetime and synchronization
----------------------------

The ``cells`` member refers to the internal buffers of the ``Clusterer``:

- the next call to ``make_clusters`` overwrites them, so the grid describes only the last run, and
  a copy has to be made before running the clustering again if the grid is needed later;
- they are freed when the ``Clusterer`` is destroyed, so the ``SpatialGrid`` must not outlive it;
- they are filled asynchronously on the queue used for the clustering, so they have to be read on
  the same queue, or after waiting for it.

The full description of the struct is in the :doc:`C++ API reference <cpp/spatial_grid>`.
