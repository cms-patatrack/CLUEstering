Installation
============

CLUEstering has two interfaces, which are distributed in different ways:

- the **C++ interface** is header-only. It can be installed with CMake, included in a CMake project with
  ``add_subdirectory`` or ``FetchContent``, or obtained from Conan, Spack and the AUR;
- the **Python interface** contains compiled modules, one for each backend. It is distributed on PyPI, as
  precompiled wheels and as a source distribution.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Source
     - Interface
     - Notes
   * - :ref:`CMake installation <cpp-install>`
     - C++
     - Installs the headers and the CMake configuration files
   * - :ref:`add_subdirectory <cpp-add-subdirectory>`
     - C++
     - For a copy of the repository inside the project, e.g. a git submodule
   * - :ref:`FetchContent <cpp-fetchcontent>`
     - C++
     - Downloads CLUEstering when the project is configured
   * - :ref:`Conan <cpp-conan>`
     - C++
     - Package ``cluestering`` on ConanCenter, brings alpaka as a dependency
   * - :ref:`Spack <cpp-spack>`
     - C++
     - Package ``cluestering``, brings alpaka as a dependency
   * - :ref:`AUR <cpp-aur>`
     - C++
     - Package ``cluestering`` for Arch Linux
   * - :ref:`PyPI <python-pypi>`
     - Python
     - Precompiled wheels for Linux with the serial and OpenMP backends
   * - :ref:`PyPI, built from source <python-gpu>`
     - Python
     - Required for the CUDA, HIP and TBB backends

The packages are updated after each release, so the version they provide can be behind the latest one.

The C++ Interface
-----------------

Whatever the source, a CMake project uses the C++ interface through the ``CLUEstering::CLUEstering`` target.
CLUEstering is built on `alpaka <https://github.com/alpaka-group/alpaka>`_, which the project has to provide and
link as well (``alpaka::alpaka``). The backend is chosen by the consuming target, by defining one of the alpaka
flags listed in :doc:`dependencies`. For example, for the serial CPU backend:

.. code-block:: cmake

    add_executable(analysis main.cpp)
    target_link_libraries(analysis PRIVATE CLUEstering::CLUEstering alpaka::alpaka)
    target_compile_definitions(analysis PRIVATE ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)

The parallel and GPU backends need some more configuration, such as linking TBB or OpenMP or compiling the
sources as CUDA: see :doc:`getting-started` for a complete ``CMakeLists.txt``.

The sections below show how to make the two targets available. A minimal program to try them is:

.. code-block:: cpp

    #include <CLUEstering/CLUEstering.hpp>

    #include <iostream>

    int main(int argc, char* argv[]) {
      auto queue = clue::get_queue(0u);

      auto h_points = clue::read_csv<2, float>(queue, argv[1]);

      const float density_radius = 20.f, min_density = 10.f, outlier_distance = 20.f;
      clue::Clusterer<2> algo(density_radius, min_density, outlier_distance);
      algo.make_clusters(queue, h_points);

      std::cout << "Found " << h_points.n_clusters() << " clusters\n";
    }

.. _cpp-install:

Installing with CMake
~~~~~~~~~~~~~~~~~~~~~

From a clone of the repository, or from one of the release
`archives <https://github.com/cms-patatrack/CLUEstering/releases>`_:

.. code-block:: shell

    cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/path/to/install
    cmake --install build

Without ``CMAKE_INSTALL_PREFIX`` the files are installed in the default system location, which may require
``sudo``. alpaka can be installed in the same way, from its
`repository <https://github.com/alpaka-group/alpaka>`_.

The project then finds both packages:

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.25)
    project(MyAnalysis LANGUAGES CXX)

    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    find_package(alpaka 2.2 REQUIRED)
    find_package(CLUEstering REQUIRED)

    add_executable(analysis main.cpp)
    target_link_libraries(analysis PRIVATE CLUEstering::CLUEstering alpaka::alpaka)
    target_compile_definitions(analysis PRIVATE ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)

If they were installed in a custom location, pass it when configuring the project:

.. code-block:: shell

    cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/install

.. _cpp-add-subdirectory:

Using add_subdirectory
~~~~~~~~~~~~~~~~~~~~~~

CLUEstering doesn't need to be installed: a copy of the repository inside the project can be added with
``add_subdirectory``. A git submodule keeps it pinned to a release:

.. code-block:: shell

    git submodule add https://github.com/cms-patatrack/CLUEstering.git extern/CLUEstering
    git -C extern/CLUEstering checkout 2.14.2

alpaka can be added in the same way, or found with ``find_package`` as above:

.. code-block:: cmake

    add_subdirectory(extern/alpaka)
    add_subdirectory(extern/CLUEstering)

    add_executable(analysis main.cpp)
    target_link_libraries(analysis PRIVATE CLUEstering::CLUEstering alpaka::alpaka)
    target_compile_definitions(analysis PRIVATE ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)

.. _cpp-fetchcontent:

Using FetchContent
~~~~~~~~~~~~~~~~~~

``FetchContent`` downloads CLUEstering, and if needed alpaka, when the project is configured:

.. code-block:: cmake

    include(FetchContent)
    FetchContent_Declare(
      alpaka
      URL https://github.com/alpaka-group/alpaka/archive/refs/tags/2.2.0.tar.gz)
    FetchContent_Declare(
      CLUEstering
      URL https://github.com/cms-patatrack/CLUEstering/archive/refs/tags/2.14.2.tar.gz)
    FetchContent_MakeAvailable(alpaka CLUEstering)

    add_executable(analysis main.cpp)
    target_link_libraries(analysis PRIVATE CLUEstering::CLUEstering alpaka::alpaka)
    target_compile_definitions(analysis PRIVATE ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)

Using a release archive instead of a git repository makes the download smaller and the build reproducible.

.. _cpp-conan:

Using Conan
~~~~~~~~~~~

CLUEstering is available on `ConanCenter <https://conan.io/center/recipes/cluestering>`_ as ``cluestering``.
The package brings alpaka with it. In a ``conanfile.txt``:

.. code-block:: ini

    [requires]
    cluestering/2.14.2

    [generators]
    CMakeDeps
    CMakeToolchain

Install the dependencies and configure the project with the toolchain generated by Conan:

.. code-block:: shell

    conan install . --output-folder=build --build=missing -s compiler.cppstd=20
    cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=build/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release
    cmake --build build

The ``CMakeLists.txt`` is the same as for a CMake installation, with ``find_package(alpaka)`` and
``find_package(CLUEstering)``. A complete example is in
`examples/conan <https://github.com/cms-patatrack/CLUEstering/tree/main/examples/conan>`_.

.. _cpp-spack:

Using Spack
~~~~~~~~~~~

With `Spack <https://spack.io>`_, CLUEstering is installed together with alpaka:

.. code-block:: shell

    spack install cluestering
    spack load cluestering

``spack load`` adds both packages to ``CMAKE_PREFIX_PATH``, so they are found by ``find_package`` as for a
CMake installation. In a Spack environment, add ``cluestering`` to the specs instead.

.. _cpp-aur:

Using the AUR
~~~~~~~~~~~~~

On Arch Linux, CLUEstering is available in the `AUR <https://aur.archlinux.org/packages/cluestering>`_ and can be
installed with an AUR helper:

.. code-block:: shell

    yay -S cluestering

The package depends on ``alpaka-cpp`` and ``boost``, which the AUR helper installs together with it.
CLUEstering and alpaka are installed in ``/usr``, where ``find_package`` finds them without further
configuration.

The Python Interface
--------------------

.. _python-pypi:

Installing from PyPI
~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

    pip install CLUEstering

To install a specific version:

.. code-block:: shell

    pip install CLUEstering==<version>

On Linux (x86_64 and aarch64) pip installs a precompiled wheel, which includes the serial and OpenMP CPU backends.
On the other platforms pip downloads the source distribution and compiles it, which requires the build
dependencies listed in :doc:`dependencies`.

.. _python-gpu:

Enabling the GPU backends
~~~~~~~~~~~~~~~~~~~~~~~~~

The precompiled wheels don't include the CUDA and HIP backends. To use them, CLUEstering has to be compiled on
the machine where it will run, by forcing pip to build it from source:

.. code-block:: shell

    pip install -v --no-binary CLUEstering CLUEstering

The backends are compiled depending on what is found on the system:

- **CUDA**: if ``nvcc`` is found, in ``PATH`` or through the ``CUDACXX`` environment variable. By default the module
  is compiled for the GPU architecture of the machine where the build runs, so it must be run on a machine with the
  target GPU. Other architectures can be selected with the ``CUDAARCHS`` environment variable or the
  ``CMAKE_CUDA_ARCHITECTURES`` CMake variable (see below), e.g. ``CUDAARCHS="75;86"``.
- **HIP**: if the ROCm compiler is found, in ``PATH`` or through the ``HIPCXX`` environment variable.
- **TBB**: if TBB is found.

If CLUEstering is already installed from a wheel, add ``--force-reinstall`` to replace it.
The available backends can be checked with:

.. code-block:: shell

    python -c "import CLUEstering; print(CLUEstering.backends)"

Installing from source
~~~~~~~~~~~~~~~~~~~~~~

The Python interface can also be compiled and installed from a clone of the repository:

.. code-block:: shell

    git clone --recursive https://github.com/cms-patatrack/CLUEstering.git
    cd CLUEstering
    pip install -v .

This compiles all the backends supported on the system, as described above.

The build can be configured by passing CMake options to pip, for instance
``pip install -v . -C cmake.define.CLUE_USE_SYSTEM_NANOBIND=ON``:

- ``CLUE_NATIVE_ARCH`` (default ``ON``): optimize the CPU code for the machine where the build runs
  (``-march=native``). Turn it off to build packages meant for other machines.
- ``CMAKE_CUDA_ARCHITECTURES`` (default ``native``): the CUDA architectures to compile for.
- ``CLUE_USE_SYSTEM_NANOBIND`` (default ``OFF``): use the nanobind installed in the Python environment instead of
  the bundled submodule. pip builds in an isolated environment by default, so this needs ``--no-build-isolation``.
- ``CLUE_USE_SYSTEM_ALPAKA`` (default ``OFF``): use an installed alpaka (a 2.x version, 2.2.0 or newer, found
  with ``find_package``) instead of downloading it during the build.
