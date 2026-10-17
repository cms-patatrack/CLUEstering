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

The Python interface can be installed with pip:

.. code-block:: shell

    cd CLUEstering
    pip install .

Alternatively, CLUEstering is available on PyPI:

.. code-block:: shell

    pip install CLUEstering

To install a specific version:

.. code-block:: shell

    pip install CLUEstering==<version>
