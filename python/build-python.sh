#!/bin/bash

# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

set -e

# Just for information at the moment.
supported_architectures="gfx90a;gfx908;gfx940;gfx941;gfx942;gfx90a:sramecc+:xnack-;gfx908:sramecc+:xnack-;gfx940:sramecc+:xnack-;gfx941:sramecc+:xnack-;gfx942:sramecc+:xnack-;gfx90a:xnack+;gfx940:xnack+;gfx941:xnack+;gfx942:xnack+"

: ${ROCM_PATH:=/opt/rocm}
: ${ROCM_DS_PATH:=/opt/rocm-ds}
export PATH="${ROCM_DS_PATH}/bin:${ROCM_PATH}/bin:${PATH}"
# In case ldconfig was not run to include these directories...
export LD_LIBRARY_PATH="${ROCM_DS_PATH}/lib:${ROCM_DS_PATH}/lib64:${ROCM_PATH}/lib:${ROCM_PATH}/lib64:${LD_LIBRARY_PATH}"
: ${CMAKE_HIP_ARCHITECTURES:=$(rocminfo|perl -ne 'if (/Name:\s+(gfx\S+)/i) { print $1; exit; }')}
: ${WAVEFRONT_SIZE:=$(rocminfo|perl -ne 'if (/Wavefront Size:\s+(\d+)/i) { print $1; exit; }')}
# Conservative for hyper-threaded processors that lack the memory for multi-GPU
# compilation.
: ${PARALLEL_LEVEL:=$(($(nproc) / 2))}

# Someday we'll standardize on one.
GPU_TARGETS=${CMAKE_HIP_ARCHITECTURES}
AMDGPU_TARGETS=${CMAKE_HIP_ARCHITECTURES}

export ROCM_PATH CMAKE_HIP_ARCHITECTURES GPU_TARGETS AMDGPU_TARGETS WAVEFRONT_SIZE PARALLEL_LEVEL

# Silly console fun.
strong=
normal=
if [[ -v TERM ]] ; then
    strong=$(tput smso)
    normal=$(tput rmso)
    trap "echo ${normal}" EXIT
fi

if [[ "${WAVEFRONT_SIZE}" == 32 ]] ; then
    printf "${strong}NOTE: ROCm-DS only supports 64-wide wavefronts currently.${normal}\n"
    wavefront_arg='--warpsize32'
    SKBUILD_CMAKE_ARGS='-DUSE_WARPSIZE_32=ON;'
fi

mkdir -p install
export HIPGRAPH_PATH="$(realpath install)"

# Ham-handed, but this makes it "easy" to leave .venv around for debugging.
if [ -d .venv ] ; then
    rm -rf .venv
fi
# In more of a real-life scenario, delete on exit:
# trap "rm -r ${VIRTUAL_ENV}" EXIT

python3.10 -m venv .venv
. .venv/bin/activate

# Last two only needed with the -no-isolation build flag, and that also requires
# pre-installing all the other intermediate build requirements.
pip install pyyaml patchelf auditwheel piprepo build setuptools scikit-build-core cython

# Some sub-dependencies have their minimum CMake version set to 3.5, which 4.0 no
# longer supports.
pip install 'cmake<4.0'

if [[ ! -d rocGRAPH || -v FORCE ]] ; then
    : ${rocgraph_url:=git@github.com:ROCm-DS/rocGRAPH.git}
    : ${rocgraph_branch_args:=""}
    [[ -d rocGRAPH ]] || git clone --depth=1 ${rocgraph_branch_args} "${rocgraph_url}" rocGRAPH

    pushd rocGRAPH/
    # The options for both the install.sh scripts here are as follows:
    #
    #   -k :: Build for a release but include debugging symbols. TODO: Separate
    #         these into dbgsym packages.
    #
    #   -c :: Build the unsupported "clients." Currently these consist of the tests.
    #
    #   -p :: Build the packages, so tar.gz, zip, deb, and rpm, These are for
    #         installation atop existing /opt/rocm installations on supported
    #         platforms but may still work on unsupported platforms.
    #
    #   -a :: Select the accelerator architectures to target.
    #
    #   ${wavefront_arg} :: Currently, wavefront sizes of 32 and 64 cannot
    #         co-exist in the same binary builds.
    #
    #   -j${PARALLEL_LEVEL} :: Passed onto the build system for the number of
    #         jobs to be run in parallel.
    ./install.sh -k -c -p -a ${CMAKE_HIP_ARCHITECTURES} ${wavefront_arg} -j${PARALLEL_LEVEL}
    make -C build/release-debug/ install
    rsync -a rocgraph-install/ ${HIPGRAPH_PATH}/
    popd

if [[ ! -d hipGRAPH || -v FORCE ]] ; then
    ${hipgraph_url:=git@github.com:ROCm-DS/hipGRAPH.git}
    ${hipgraph_branch_args:=""}
    [[ -d hipGRAPH ]] || git clone --depth=1 ${hipgraph_branch_args} "${hipgraph_url}" hipGRAPH

    pushd hipGRAPH
    ./install.sh -pkc --rocgraph-path "${HIPGRAPH_PATH}" -j${PARALLEL_LEVEL}
    make -C build/release-debug/ install
    rsync -a build/release-debug/hipgraph-install/ ${HIPGRAPH_PATH}/
    popd
fi

export CMAKE_MODULE_PATH="${HIPGRAPH_PATH}/lib/cmake:/opt/rocm-ds/lib/cmake:/opt/rocm/lib/cmake"

rm -rf wheelhouse
mkdir wheelhouse
export WHL="$(realpath wheelhouse)"
export urlWHL="file://$(echo $WHL | sed -s 's/ /%20/g;')/simple"

# You need to supply your own wf32 wheels.
if [[ "${WAVEFRONT_SIZE}" == 32 ]] ; then
    shopt -s nullglob
    dep=(wheelhouse/amd_cupy-*.whl)
    fail=0
    if [ -z "${dep}" ] ; then
        echo "${strong}Missing amd-cupy wheel.${normal}"
        fail=1
    fi
    dep=(wheelhouse/amd_hipmm-*.whl)
    if [ -z "${dep}" ] ; then
        echo "${strong}Missing amd-hipmm wheel.${normal}"
        fail=$(($fail + 1))
    fi
    dep=(wheelhouse/amd_hipdf-*.whl)
    if [ -n "${dep}" ] ; then
        echo "${strong}Missing amd-hipidf wheels.${normal}"
        fail=$(($fail + 1))
    fi
    if [ "${fail}" -gt 0 ] ; then
        echo "${strong}Missing ${fail} wheels for a wavefront-32 build.${normal}"
        exit ${fail}
    fi
fi

piprepo build "${WHL}"

pushd hipGRAPH/python

# Clean out any existing packages. This is useful for re-running the post-setup pieces.
pip uninstall -y amd-hipdf amd-nx-hipgraph amd-hipmm amd-cupy \
    amd-pylibhipgraph amd-hipgraph amd-nx-hipgraph || /bin/true

# Note that pip does *not* guarantee any ordering between repos. And building
# some of the below modules can fail until pip happens to resolve numba-hip
# before amd-hipmm. This is as much fun as it sounds.
#
# Users may need to re-run this script. Multiple times.

export PIP_EXTRA_INDEX_URL="${urlWHL} https://pypi.amd.com/simple https://test.pypi.org/simple"

# At least attempt to convince pip to use this *first*.
pushd "${WHL}"
pip download 'numba-hip[rocm-6-4-0]@git+https://github.com/rocm/numba-hip.git'
piprepo build .
popd

export VERBOSE=ON
export SKBUILD_CMAKE_ARGS="-DCMAKE_BUILD_TYPE=RelWithDebSymbols;-DHIPGRAPH_PATH=${HIPGRAPH_PATH}"

rm -rf pylibhipgraph/dist pylibhipgraph/_skbuild pylibhipgraph/build
python3 -m build pylibhipgraph

auditwheel show pylibhipgraph/dist/amd_pylibhipgraph-*linux*.whl
cp pylibhipgraph/dist/amd_pylibhipgraph-*linux*.whl "${WHL}"
# Alternatively, "repair" the wheel for the manylinux platform from auditwheel show.
# auditwheel repair --plat manylinux_2_39_x86_64 -w "${WHL}" pylibhipgraph/dist/amd_pylibhipgraph-*linux*.whl
piprepo build "${WHL}"

rm -rf hipgraph/dist hipgraph/_skbuild hipgraph/build
# bug: shouldn't need hipcc, but otherwise raft dies on hip_warp_primitive includes.
CC=${ROCM_PATH}/bin/hipcc CXX=${ROCM_PATH}/bin/hipcc VERBOSE=ON python3 -m build hipgraph

auditwheel show hipgraph/dist/amd_hipgraph-*linux*.whl
cp hipgraph/dist/amd_hipgraph-*linux*.whl "${WHL}"
# Alternatively, "repair" the wheel for the manylinux platform from auditwheel show.
# auditwheel repair --plat manylinux_2_39_x86_64 -w "${WHL}" hipgraph/dist/amd_hipgraph-*linux*.whl
piprepo build "${WHL}"

# Pure python module.
python3 -m build -o "${WHL}" nx-hipgraph
piprepo build "${WHL}"

# Repeat installation of numba-hip to ease pip's dependency resolution pain at this point.
pip install 'numba-hip[rocm-6-4-0]@git+https://github.com/rocm/numba-hip.git' amd-hipgraph amd-nx-hipgraph
