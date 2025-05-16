#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# Author : Yvan Mokwinski

# Terminal control codes
if [ -t 1 -a -n "${TERM}" -a $(tput colors) -ge 8 ]; then
  hBOLD="$(tput bold)"
  hUL="$(tput smul)"
  hunUL="$(tput rmul)"
  hREV="$(tput rev)"
  hBLINK="$(tput blink)" # And the generations shall spit upon you.
  hINVIS="$(tput invis)"
  hSTAND="$(tput smso)"
  hunSTAND="$(tput rmso)"

  hRESET="$(tput sgr0)"

  cBLACK="$(tput setaf 0)"
  cRED="$(tput setaf 1)"
  cGREEN="$(tput setaf 2)"
  cYELLOW="$(tput setaf 3)"
  cBLUE="$(tput setaf 3)"
  cMAGENTA="$(tput setaf 3)" # Master! Dinner is prepared! (Meatloaf? Again?)
  cCYAN="$(tput setaf 3)"
  cWHITE="$(tput setaf 3)"

  cbBLACK="$(tput setab 0)"

  cRESET="$(tput sgr0)"
else
  hBOLD=""
  hUL=""
  hunUL=""
  hREV=""
  hBLINK=""
  hINVIS=""
  hSTAND=""
  hunSTAND=""

  hRESET=""

  cBLACK=""
  cRED=""
  cGREEN=""
  cYELLOW=""
  cBLUE=""
  cMAGENTA=""
  cCYAN=""
  cWHITE=""

  cbBLACK=""

  cRESET=""
fi

reset_colors() {
  printf "${cRESET}"
}

trap reset_colors EXIT

# set -x # echo on, must be *after* the terminal control codes.

# #################################################
# helper functions
# #################################################
function display_help()
{
  echo "hipGRAPH build & installation helper script"
  echo "./install [-h|--help] "
  echo "    [-h|--help] prints this help message"
  echo "    [--prefix] Specify an alternate CMAKE_INSTALL_PREFIX for cmake."
  echo "      This also points CMake to configs under the specified prefix."
  echo "    [-p|--package] build package"
  echo "    [-b|--build-dir] directory to build in"
  echo "    [-s|--source-dir] top-level source directory"
  echo "    [-i|--install] install after build"
  echo "    [-d|--dependencies] install build dependencies"
  echo "    [-r]--relocatable] create a package to support relocatable ROCm"
  echo "    [-c|--clients] build library clients too (combines with -i & -d)"
  echo "    [-g|--debug] -DCMAKE_BUILD_TYPE=Debug (default is =Release)"
  echo "    [-k|--relwithdebinfo] -DCMAKE_BUILD_TYPE=RelWithDebInfo"
  echo "    [--codecoverage] build with code coverage profiling enabled"
  echo "    [--cuda] build library for cuda backend"
  echo "    [--static] build static library"
  echo "    [--address-sanitizer] build with address sanitizer enabled. Uses hipcc to compile"
  echo "    [--matrices-dir] existing client matrices directory"
  echo "    [--matrices-dir-install] install client matrices directory"
  echo "    [--cmake-arg] Forward the given argument to CMake when configuring the build."
  echo "    [--rocm-path] Path to a ROCm install, default /opt/rocm. Overrides ROCM_PATH."
  echo "    [--rocgraph-path] Path to a rocGRAPH installation separate from rocm-path."
  echo "    [-j] Sets the parallelism level, default $(nproc). Overrides PARALLEL_LEVEL."
  echo
  echo "Note: Currently the Python modules are not included in this build script."
  echo "  See clients/python/README.md for more information."
  echo
  echo "Environment variables used:"
  echo "  ROCM_PATH : Path to a ROCm install, defaults to /opt/rocm."
  echo "  ROCM_RPATH : If set, overrides ld.so's RPATH for this build."
  echo "  PARALLEL_LEVEL : Number of parallel jobs run, $(nproc) by default."
  echo "      CMAKE_BUILD_PARALLEL_LEVEL also is accepted for CMake consistency."
}

# This function is helpful for dockerfiles that do not have sudo installed, but the default user is root
# true is a system command that completes successfully, function returns success
# prereq: ${ID} must be defined before calling
supported_distro( )
{
  if [ -z ${ID+foo} ]; then
    printf "supported_distro(): \$ID must be set\n"
    exit 2
  fi

  case "${ID}" in
    debian|ubuntu|centos|rhel|fedora|sles|opensuse-leap)
        true
        ;;
    *)  printf "This script is currently supported on Debian, Ubuntu, CentOS, RHEL, Fedora, SLES, and OpenSUSE-Leap\n"
        exit 2
        ;;
  esac
}

# This function is helpful for dockerfiles that do not have sudo installed, but the default user is root
check_exit_code( )
{
    if (( $1 != 0 )); then
	exit $1
    fi
}

# This function is helpful for dockerfiles that do not have sudo installed, but the default user is root
elevate_if_not_root( )
{
  local uid=$(id -u)

  if (( ${uid} )); then
    sudo $@
    check_exit_code "$?"
  else
    $@
    check_exit_code "$?"
  fi
}

# Take an array of packages as input, and install those packages with 'apt-get' if they are not already installed
install_apt_packages( )
{
  package_dependencies=("$@")
  for package in "${package_dependencies[@]}"; do
    if [[ $(dpkg-query --show --showformat='${db:Status-Abbrev}\n' ${package} 2> /dev/null | grep -q "ii"; echo $?) -ne 0 ]]; then
      printf "${cGREEN}Installing ${cYELLOW}${package}${cGREEN} from distro package manager${cRESET}\n"
      elevate_if_not_root apt-get install -y --no-install-recommends ${package}
    fi
  done
}

# Take an array of packages as input, and install those packages with 'yum' if they are not already installed
install_yum_packages( )
{
  package_dependencies=("$@")
  for package in "${package_dependencies[@]}"; do
    if [[ $(yum list installed ${package} &> /dev/null; echo $? ) -ne 0 ]]; then
      printf "${cGREEN}Installing ${cYELLOW}${package}${cGREEN} from distro package manager${cRESET}\n"
      elevate_if_not_root yum -y --nogpgcheck install ${package}
    fi
  done
}

# Take an array of packages as input, and install those packages with 'dnf' if they are not already installed
install_dnf_packages( )
{
  package_dependencies=("$@")
  for package in "${package_dependencies[@]}"; do
    if [[ $(dnf list installed ${package} &> /dev/null; echo $? ) -ne 0 ]]; then
      printf "${cGREEN}Installing ${cYELLOW}${package}${cGREEN} from distro package manager${cRESET}\n"
      elevate_if_not_root dnf install -y ${package}
    fi
  done
}

# Take an array of packages as input, and install those packages with 'zypper' if they are not already installed
install_zypper_packages( )
{
  package_dependencies=("$@")
  for package in "${package_dependencies[@]}"; do
    if [[ $(rpm -q ${package} &> /dev/null; echo $? ) -ne 0 ]]; then
      printf "${cGREEN}Installing ${cYELLOW}${package}${cGREEN} from distro package manager${cRESET}\n"
      elevate_if_not_root zypper -n --no-gpg-checks install ${package}
    fi
  done
}

# Take an array of packages as input, and delegate the work to the appropriate distro installer
# prereq: ${ID} must be defined before calling
# prereq: ${build_clients} must be defined before calling
install_packages( )
{
    if [ -z ${ID+foo} ]; then
        printf "install_packages(): \$ID must be set\n"
        exit 2
    fi

    if [ -z ${build_clients+foo} ]; then
        printf "install_packages(): \$build_clients must be set\n"
        exit 2
    fi

    # dependencies needed for library and clients to build
    local library_dependencies_debian=( "build-essential" "cmake" "pkg-config" )
    local library_dependencies_ubuntu=${library_dependencies_debian}
    local library_dependencies_centos_6=( "epel-release" "make" "cmake3" "gcc-c++" "rpm-build" )
    local library_dependencies_centos_7=${library_dependencies_centos_6}
    local library_dependencies_centos_8=${library_dependencies_centos_7}
    local library_dependencies_fedora=( "make" "cmake" "gcc-c++" "libcxx-devel" "rpm-build" )
    local library_dependencies_sles=( "make" "cmake" "gcc-c++" "rpm-build" "pkg-config" )

    case "${ID}" in
        debian)
            elevate_if_not_root apt update
            install_apt_packages "${library_dependencies_debian[@]}"
            ;;

        ubuntu)
            elevate_if_not_root apt update
            install_apt_packages "${library_dependencies_ubuntu[@]}"
            ;;

        centos|rhel)
            #     yum -y update brings *all* installed packages up to date
            #     without seeking user approval
            #     elevate_if_not_root yum -y update
            if [[ "${MAJORVERSION}" == 8 ]]; then
                install_yum_packages "${library_dependencies_centos_8[@]}"

                if [[ "${build_clients}" == true ]]; then
                    install_yum_packages "${client_dependencies_centos_8[@]}"
                fi
            elif [[ "${MAJORVERSION}" == 6 ]]; then
                install_yum_packages "${library_dependencies_centos_6[@]}"

                if [[ "${build_clients}" == true ]]; then
                    install_yum_packages "${client_dependencies_centos_6[@]}"
                fi
            else
                install_yum_packages "${library_dependencies_centos_7[@]}"

                if [[ "${build_clients}" == true ]]; then
                    install_yum_packages "${client_dependencies_centos_7[@]}"
                fi
            fi
            ;;

        fedora)
            #     elevate_if_not_root dnf -y update
            install_dnf_packages "${library_dependencies_fedora[@]}"

            if [[ "${build_clients}" == true ]]; then
                install_dnf_packages "${client_dependencies_fedora[@]}"
            fi
            ;;

        sles|opensuse-leap)
            #     elevate_if_not_root zypper -y update
            install_zypper_packages "${library_dependencies_sles[@]}"

            if [[ "${build_clients}" == true ]]; then
                install_zypper_packages "${client_dependencies_sles[@]}"
            fi
            ;;
        *)
            printf "This script is currently supported on Debian, Ubuntu, CentOS, RHEL, Fedora, SLES, and OpenSUSE-Leap\n"
            exit 2
            ;;
    esac
}

# #################################################
# Pre-requisites check
# #################################################
# Exit code 0: all is well
# Exit code 1: problems with getopt
# Exit code 2: problems with supported platforms

# check if getopt command is installed
type getopt > /dev/null
if [[ $? -ne 0 ]]; then
  echo "This script uses getopt to parse arguments; try installing the util-linux package";
  exit 1
fi

# /etc/*-release files describe the system
if [[ -e "/etc/os-release" ]]; then
  source /etc/os-release
elif [[ -e "/usr/lib/os-release" ]]; then
  source /usr/lib/os-release
elif [[ -e "/etc/centos-release" ]]; then
  ID=$(cat /etc/centos-release | awk '{print tolower($1)}')
  VERSION_ID=$(cat /etc/centos-release | grep -oP '(?<=release )[^ ]*' | cut -d "." -f1)
else
  echo "This script depends on the /etc/*-release files"
  exit 2
fi

MAJORVERSION=$(echo $VERSION_ID | cut -f1 -d.)
# The following function exits script if an unsupported distro is detected
supported_distro

# #################################################
# global variables
# #################################################
build_package=false
install_package=false
install_dependencies=false
build_clients=false
build_cuda=false
build_static=false
build_release=true
build_release_debug=false
build_codecoverage=false
build_directory=$(realpath ./build)
source_directory=$(realpath .)
rocm_path="${ROCM_PATH:-/opt/rocm}"
rocm_rpath="${ROCM_RPATH:-${rocm_path}/lib64:${rocm_path}/lib}"
rocgraph_path="${ROCGRAPH_PATH:-''}" # For building against a non-installed rocGRAPH
# TODO: Use DESTDIR for CPack. The below sets CMAKE_INSTALL_PREFIX, and that
# affects the search path for CMake config modules.
# install_prefix="${rocm_path}"
install_prefix="hipgraph-install"
build_relocatable=false
build_address_sanitizer=false
declare -a cmake_common_options
declare -a cmake_client_options
declare -a cmake_build_static_options
# The next is questionable. The nproc default runs out of memory on thread-heavy systems.
parallel_level=${PARALLEL_LEVEL:-${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}}

export CC=${CC:-hipcc}
export CXX=${CC:-hipcc}

# #################################################
# Parameter parsing
# #################################################

# check if we have a modern version of getopt that can handle whitespace and long parameters
getopt -T
if [[ $? -eq 4 ]]; then
  GETOPT_PARSE=$(getopt --name "${0}" --longoptions help,build-dir:,source-dir:,install,package,clients,dependencies,debug,cuda,static,relocatable,codecoverage,relwithdebinfo,address-sanitizer,matrices-dir:,matrices-dir-install:,rm-legacy-include-dir,cmake-arg:,rocm-path:,rocgraph-path: --options hpb:B:s:S:icdgrkj: -- "$@")
else
  echo "Need a new version of getopt"
  exit 1
fi

if [[ $? -ne 0 ]]; then
  echo "getopt invocation failed; could not parse the command line";
  exit 1
fi

eval set -- "${GETOPT_PARSE}"

while true; do
  case "${1}" in
    -h|--help)
        display_help
        exit 0
        ;;
    -p|--package)
        build_package=true
        shift ;;
    -i|--install)
        install_package=true
        build_package=true
        shift ;;
    -B|-b|--build-dir)
        build_directory="${2}"
        shift 2 ;;
    -S|-s|--source-dir)
        source_directory="${2}"
        shift 2 ;;
    -d|--dependencies)
        install_dependencies=true
        shift ;;
    -c|--clients)
        build_clients=true
        shift ;;
    -r|--relocatable)
        build_relocatable=true
        shift ;;
    -g|--debug)
        build_release=false
        shift ;;
    --cuda)
        build_cuda=true
        shift ;;
    --static)
        build_static=true
        shift ;;
    -k|--relwithdebinfo)
        build_release=false
        build_release_debug=true
        shift ;;
    --codecoverage)
        build_codecoverage=true
        shift ;;
    --address-sanitizer)
        build_address_sanitizer=true
        shift ;;
    --matrices-dir)
        matrices_dir=${2}
        if [[ "${matrices_dir}" == "" ]];then
            echo "Missing argument from command line parameter --matrices-dir; aborting"
            exit 1
        fi
        shift 2 ;;
    --matrices-dir-install)
        matrices_dir_install=${2}
        if [[ "${matrices_dir_install}" == "" ]];then
            echo "Missing argument from command line parameter --matrices-dir-install; aborting"
            exit 1
        fi
        shift 2 ;;
    --prefix)
        install_prefix=${2}
        shift 2 ;;
    --cmake-arg)
        cmake_common_options+=("${2}")
        shift 2 ;;
    -j)
        parallel_level=${2}
        shift 2 ;;
    --rocm-path)
        rocm_path=${2}
        shift 2 ;;
    --rocgraph-path)
        rocgraph_path=${2}
        shift 2 ;;
    --) shift ; break ;;
    *)  echo "Unexpected command line parameter received; aborting";
        exit 1
        ;;
  esac
done

if [[ "${build_relocatable}" == true ]]; then
    rocm_rpath=" -Wl,--enable-new-dtags -Wl,--rpath,${rocm_rpath}"
fi

#
# If matrices_dir_install has been set up then install matrices dir and exit.
#
if ! [[ "${matrices_dir_install}" == "" ]];then
    cmake -DCMAKE_CXX_COMPILER="${rocm_path}/bin/hipcc" -DCMAKE_C_COMPILER="${rocm_path}/bin/hipcc"  -DPROJECT_BINARY_DIR=${matrices_dir_install} -DCMAKE_MATRICES_DIR=${matrices_dir_install} -P ./cmake/ClientMatrices.cmake
    exit 0
fi

#
# If matrices_dir has been set up then check if it exists and it contains expected files.
# If it doesn't contain expected file, it will create them.
#
if ! [[ "${matrices_dir}" == "" ]];then
    if ! [ -e ${matrices_dir} ];then
        echo "Invalid dir from command line parameter --matrices-dir: ${matrices_dir}; aborting";
        exit 1
    fi

    # Let's 'reinstall' to the specified location to check if all good
    # Will be fast if everything already exists as expected.
    # This is to prevent any empty directory.
    cmake -DCMAKE_CXX_COMPILER="${rocm_path}/bin/hipcc" -DCMAKE_C_COMPILER="${rocm_path}/bin/hipcc" -DPROJECT_BINARY_DIR=${matrices_dir} -DCMAKE_MATRICES_DIR=${matrices_dir} -P ./cmake/ClientMatrices.cmake
fi

if [ -f "${source_directory}/CMakeLists.txt" ] ; then
    source_dir="${source_directory}"
else
    printf "${cRED}Error: Source directory ${source_directory} does not contain CMakeLists.txt.${cRESET}\n"
    exit -1
fi

if [[ "${build_release}" == true ]]; then
  build_dir="${build_directory}/release"
elif [[ "${build_release_debug}" == true ]]; then
  build_dir="${build_directory}/release-debug"
else
  build_dir="${build_directory}/debug"
fi

printf "${cbBLACK}${cGREEN}Creating project build directory in: ${cYELLOW}${build_dir}${cRESET}\n"

# #################################################
# prep
# #################################################
# Ensure a clean build environment and define the build directory.
rm -rf ${build_dir}

# Default cmake executable is called cmake
cmake_executable=cmake

case "${ID}" in
  centos|rhel)
  cmake_executable=cmake3
  ;;
esac

# We append customary rocm path; if user provides custom rocm path in ${path}, our
# hard-coded path has lesser priority
if [[ "${build_relocatable}" == true ]]; then
    export PATH=${rocm_path}/bin:${PATH}
else
    export PATH=${PATH}:/opt/rocm/bin
fi

pushd .
  # #################################################
  # configure & build
  # #################################################

  # For development, if there is a custom rocGRAPH, use it.
  if [[ ! -z "$rocgraph_path" && "${build_cuda}" == false ]] ; then
      cmake_common_options+=("-DCUSTOM_ROCGRAPH=${rocgraph_path}")
  fi

  # build type
  mkdir -p ${build_dir} && cd ${build_dir}
  if [[ "${build_release}" == true ]]; then
    cmake_common_options+=("-DCMAKE_BUILD_TYPE=Release")
  elif [[ "${build_release_debug}" == true ]]; then
    cmake_common_options+=("-DCMAKE_BUILD_TYPE=RelWithDebInfo")
  else
    cmake_common_options+=("-DCMAKE_BUILD_TYPE=Debug")
  fi

  # code coverage
  if [[ "${build_codecoverage}" == true ]]; then
      if [[ "${build_release}" == true ]]; then
          echo "Code coverage is disabled in Release mode, to enable code coverage select either Debug mode (-g | --debug) or RelWithDebInfo mode (-k | --relwithdebinfo); aborting";
          exit 1
      fi
      cmake_common_options+=("-DBUILD_CODE_COVERAGE=ON")
  fi

  # address sanitizer
  if [[ "${build_address_sanitizer}" == true ]]; then
    cmake_common_options+=("-DBUILD_ADDRESS_SANITIZER=ON")
  fi

  # library type
  if [[ "${build_static}" == true ]]; then
    cmake_common_options+=("-DBUILD_SHARED_LIBS=OFF")
    cmake_build_static_options+=("-no-pie")
  fi

  # clients
  if [[ "${build_clients}" == true ]]; then
    cmake_client_options+=("-DBUILD_CLIENTS_SAMPLES=OFF" "-DBUILD_CLIENTS_TESTS=ON")
    #
    # Add matrices_dir if exists.
    #
    if ! [[ "${matrices_dir}" == "" ]];then
        cmake_client_options+=("-DCMAKE_MATRICES_DIR=${matrices_dir}")
    fi
  fi

  # cpack
  if [[ "${build_relocatable}" == true ]]; then
    cmake_common_options+=("-DCPACK_SET_DESTDIR=OFF" "-DCPACK_PACKAGING_INSTALL_PREFIX=${rocm_path}")
  else
    cmake_common_options+=("-DCPACK_SET_DESTDIR=OFF" "-DCPACK_PACKAGING_INSTALL_PREFIX=/opt/rocm")
  fi

  # cuda or hip
  if [[ "${build_cuda}" == false ]]; then
    cmake_common_options+=("-DUSE_CUDA=OFF")
  else
    cmake_common_options+=("-DUSE_CUDA=ON")
  fi

  # Configure library
  if [[ "${build_relocatable}" == true ]]; then
    env CXX=${CXX} CC=${CC} ${cmake_executable} ${cmake_common_options[*]} ${cmake_client_options[*]} \
      -DCMAKE_INSTALL_PREFIX="${install_prefix}" \
      -DCMAKE_SHARED_LINKER_FLAGS="${rocm_rpath}" \
      -DCMAKE_PREFIX_PATH="${rocm_path}" \
      -DCMAKE_MODULE_PATH="${rocm_path}/lib/cmake ${rocm_path}/hip/cmake" \
      -DCMAKE_EXE_LINKER_FLAGS=" -Wl,--enable-new-dtags -Wl,--rpath,${rocm_path}/lib:${rocm_path}/lib64 ${cmake_build_static_options}" \
      -DROCM_DISABLE_LDCONFIG=ON \
      -DROCM_PATH="${rocm_path}" -B ${build_dir} -S ${source_dir}
  else
    env CXX=${CXX} CC=${CC} ${cmake_executable} \
        -DCMAKE_EXE_LINKER_FLAGS=" ${cmake_build_static_options[*]}" ${cmake_common_options[*]} ${cmake_client_options[*]} \
        -DCMAKE_INSTALL_PREFIX="${install_prefix}" \
        -DROCM_PATH=${rocm_path} -B ${build_dir} -S ${source_dir}
  fi
  check_exit_code "$?"

  # Configure library
  ${cmake_executable} --build .
  check_exit_code "$?"

  # Pseudo-install the library into the build directory
  ${cmake_executable} --build . -j${parallel_level} --target install
  check_exit_code "$?"

  # #################################################
  # build package
  # #################################################
  if [[ "${build_package}" == true ]]; then
    ${cmake_executable} --build . -- package
    check_exit_code "$?"
  fi

  # #################################################
  # install
  # #################################################
  # Check if the user *may* believe they're installing into a "virtual
  # environment." These checks are neither comprehensive nor reliable.
  if [[ -v CONDA_ENV_PATH ]] ; then
      printf "${cbBLACK}${cRED}This script appears to be running in a Conda environment.\n${hSTAND}The packages are installed globally.${hunSTAND}${cRESET}"
  elif [[ -v VIRTUAL_ENV ]] ; then
      printf "${cbBLACK}${cRED}This script appears to be running in a Python virtual environment.\n${hSTAND}The packages are installed globally.${hunSTAND}${cRESET}"
  fi

  # Install through the system's package manager, ideally simplifying removing the package.
  if [[ "${install_package}" == true ]]; then
    case "${ID}" in
      debian|ubuntu)
        elevate_if_not_root dpkg -i hipgraph[-\_]*.deb
      ;;
      centos|rhel)
        elevate_if_not_root yum -y localinstall hipgraph-*.rpm
      ;;
      fedora)
        elevate_if_not_root dnf install hipgraph-*.rpm
      ;;
      sles|opensuse-leap)
        elevate_if_not_root zypper -n --no-gpg-checks install hipgraph-*.rpm
      ;;
    esac
    check_exit_code "$?"
  fi
popd
