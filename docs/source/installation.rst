How to Install
==============

Installing the C++ Interface
----------------------------

The C++ interface of CLUEstering can be installed using CMake with the following steps:

.. code-block:: shell

    mkdir -p build && cd build
    cmake -B .. -S ..
    sudo cmake --install .

This will install the library and its headers in the default system paths.  

To install in a custom location, pass the `CMAKE_INSTALL_PREFIX` option during configuration:

.. code-block:: shell

    cmake -B .. -S .. -DCMAKE_INSTALL_PREFIX=/path/to/install

Using CLUEstering Without Installing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CLUEstering is header-only, so you can use it directly in a project without installing. The easiest ways are:

### Adding CLUEstering as a Git Submodule

.. code-block:: shell

    cd my-project
    git submodule add https://github.com/cms-patatrack/CLUEstering.git external/CLUEstering --branch branch-name

### Using CMake's FetchContent

If your project uses CMake, CLUEstering can be added as a dependency via `FetchContent`:

.. code-block:: cmake

    include(FetchContent)
    FetchContent_Declare(
        CLUEstering
        URL https://github.com/cms-patatrack/CLUEstering
    )

Installing the Python Interface
-------------------------------

CLUEstering is available on PyPI:

.. code-block:: shell

    pip install CLUEstering

To install a specific version:

.. code-block:: shell

    pip install CLUEstering==<version>

On Linux (x86_64 and aarch64) pip installs a precompiled wheel, which includes the serial and OpenMP CPU backends.
On the other platforms pip downloads the source distribution and compiles it, which requires the build
dependencies listed in :doc:`dependencies`.

Enabling the GPU backends
~~~~~~~~~~~~~~~~~~~~~~~~~

The precompiled wheels don't include the CUDA and HIP backends. To use them, CLUEstering has to be compiled on
the machine where it will run, by forcing pip to build it from source:

.. code-block:: shell

    pip install -v --no-binary CLUEstering CLUEstering

The backends are compiled depending on what is found on the system:

- **CUDA**: if ``nvcc`` is found, in ``PATH`` or through the ``CUDACXX`` environment variable. The module is compiled
  for the GPU architecture of the machine where the build runs, so it must be run on a machine with the target GPU.
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
