# hipGRAPH

> [!CAUTION]
> This release is an *early-access* software technology preview. Running production workloads is *not* recommended.

> [!NOTE]
> This ROCm&trade; port is derived work based on the NVIDIA RAPIDS&reg; cugraph project's C API. The CUDA back-end is not tested regularly.

hipGRAPH is a graph marshalling library that acts as a wrapper between your application and the 'worker' graph library, `rocGRAPH <https://github.com/ROCm-DS/rocGraph>`.
The hipGraph library contains a collection of graph algorithms, enabling you to build, analyze, and manipulate complex graphs or networks. hipGraph is derived from the cuGraph library that forms part of the NVIDIA RAPIDS open source project. The hipGraph library described here should not be confused with `HIP graphs <https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html>`_ in the HIP runtime API.

## Documentation

Documentation for hipGRAPH is in progress.

## Build hipGRAPH

To build hipGRAPH, you can use our bash helper script (for
Ubuntu, Centos, RHEL, Fedora, SLES, openSUSE-Leap) or you can
perform a manual build (for all supported platforms).

* Bash helper script (`install.sh`):
  This script, which is located in the root of this repository, builds and installs hipGRAPH on Ubuntu
  with a single command. Note that this option doesn't allow much customization and hard-codes
  configurations that can be specified through invoking CMake directly. Some commands in the script
  require sudo access, so it may prompt you for a password.

    ```bash
    ./install.sh -h  # shows help
    ./install.sh -id # builds library, dependencies, then installs (the `-d` flag only needs to be passed once on a system)
    ```

* Manual build:
    If you use a distribution other than Ubuntu, or would like more control over the build process,
    run cmake manually. See the install

    ```bash
    mkdir build
    cd build
    cmake .. -DCMAKE_CXX_COMPILER="${rocm_path}/bin/hipcc" \
        -DCMAKE_C_COMPILER="${rocm_path}/bin/hipcc" \
        -DCMAKE_BUILD_TYPE=Debug
    # Add -DUSE_CUDA=ON to enable the NVIDIA back-end
    ```
## Python modules

The modules overall are a work in progress. `pylibhipgraph` is functional and can be build from the `python` subdirectory with
``` bash
python3 -m build pylibhipgraph/
```
or
``` bash
conda run -n dev python3 -m build pylibhipgraph/
```
once all the prerequisites are installed. Documentation for installing the prerequisites is in progress.
