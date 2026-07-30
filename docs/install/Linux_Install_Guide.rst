.. meta::
  :description: hipGRAPH documentation and API reference library
  :keywords: Graph, Graph-algorithms, Graph-analysis, Graph-processing, Complex-networks, rocGraph, hipGraph, cuGraph, NetworkX, GPU, RAPIDS, AMD Data Science

.. _linux-install:

********************************************************************
hipGRAPH installation instructions
********************************************************************

You can install hipGRAPH using the following instructions. There are some prerequisites
that should be installed prior to installing the hipGRAPH library, as described in the
following steps.

Prerequisites
=============

hipGRAPH requires a ROCm-enabled platform as an implementation backend. This documentation assumes that
you have a system with a compatible AMD GPU. hipGRAPH has been tested with the following system requirements:

* Ubuntu version: 22.04
* ROCm version: 6.4.0
* AMD GPUs: AMD Instinct MI210 and AMD Radeon RX7900 GRE

Building hipGRAPH from source for the ROCm backend
==================================================

There currently are no prebuilt packages, and hipGRAPH must be built and installed from source files.

The following compile-time dependencies must be met:

- `AMD ROCm 6.4.0 <https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.4.0/>`_
- `rocGRAPH <https://github.com/AMD-Ecosystem/rocGRAPH>`_
- `git <https://git-scm.com/>`_
- `CMake <https://cmake.org/>`_ 3.5 or later
- `GoogleTest <https://github.com/google/googletest>`_ (optional, for the test suite)
- Python 3.10 (optional for Python modules)

.. note::
   hipGRAPH does not require hipcc and is tested against other compilers like
   g++. It does require the libraries in a ROCm installation.

Download hipGRAPH
-----------------

The hipGRAPH source code is available at the `hipGRAPH GitHub page <https://github.com/AMD-Ecosystem/hipGRAPH>`_.
Download the source code using the following commands:

.. code:: bash

  $ git clone https://github.com/AMD-Ecosystem/hipGRAPH.git
  $ cd hipGRAPH

Using ``install.sh`` to build and install hipGRAPH
--------------------------------------------------

It is recommended that you use the ``install.sh`` script to build and install different packages for the hipGRAPH library, including dependencies needed by hipGRAPH, and clients that use hipGRAPH such as unit tests.

The following table lists common uses of ``install.sh`` from the hipGRAPH source
folder to build and install hipGRAPH and optionally its dependencies and clients.

.. list-table::
    :widths: 3, 9

    * - **Command**
      - **Description**

    * - ``./install.sh -h``
      - Print help information.

    * - ``./install.sh``
      - Build the hipGRAPH library in your local directory. It is assumed that the required dependencies have been previously installed.

    * - ``./install.sh -d``
      - Build the hipGRAPH library and its dependencies in your local directory. The ``-d`` flag only needs to be used once. For subsequent invocations of install.sh it is not necessary to rebuild the dependencies.

    * - ``./install.sh -c``
      - Build the library and client in your local directory. It is assumed dependencies are available.

    * - ``./install.sh -dc``
      - Build the library, dependencies, and client in your local directory. The ``-d`` flag only needs to be used once. For subsequent invocations of install.sh it is not necessary to rebuild the dependencies.

    * - ``./install.sh -i``
      - Build the library, then build and install the hipGRAPH package in ``/opt/rocm/hipgraph``. You will be prompted for sudo access. This will install for all users.

    * - ``./install.sh -idc``
      - Build the library, dependencies, and client, then build and install the hipGRAPH package in ``/opt/rocm/hipgraph``. You will be prompted for sudo access. This will install for all users.

    * - ``./install.sh -ic``
      - Build the library and client, then build and install the hipGRAPH package in ``opt/rocm/hipgraph``. You will be prompted for sudo access. This will install for all users.


Building and installing hipGRAPH with individual commands
---------------------------------------------------------

CMake 3.5 or later is required to build hipGRAPH. GoogleTest is required to build hipGRAPH clients.

Despite its name, `CMAKE_INSTALL_PREFIX` primary usage is pointing CMake towards the package configurations in `lib/cmake` under that prefix.
The Python wheels bundle the libraries after running `auditwheel`.

You can build and install hipGRAPH along with its dependencies and clients using the following commands:

.. code:: bash

  # Install GoogleTest
  $ mkdir -p build/release/deps ; cd build/release/deps
  $ cmake ../../../deps
  $ make -j$(nproc) install

  # Change to build directory
  $ cd ..

  # Default install path is /opt/rocm, use -DCMAKE_INSTALL_PREFIX=<path> to adjust it
  $ cmake ../.. -DBUILD_CLIENTS_TESTS=ON

  # Compile hipGRAPH library
  $ make -j$(nproc)

  # Install hipGRAPH to /opt/rocm
  $ make install

Alternatively, you can build just hipGRAPH without dependencies or clients using the following commands. This assumes that the dependencies or clients have been previously installed.

.. code:: bash

  # Create and change to build directory
  $ mkdir -p build/release ; cd build/release

  # Default install path is /opt/rocm, use -DCMAKE_INSTALL_PREFIX=<path> to adjust it
  $ cmake ../..

  # Compile hipGRAPH library
  $ make -j$(nproc)

  # Install hipGRAPH into build/release/DESTDIR
  $ make install DESTDIR=$(pwd)/build/release/DESTDIR
  # Optionally, install hipGRAPH to a system-wide /opt/rocm
  $ make install

Building the Python modules
---------------------------

The Python modules require much more detailed instructions, which are work
in progress. See ``clients/python``.
