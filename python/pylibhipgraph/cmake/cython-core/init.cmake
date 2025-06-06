# Copyright (c) 2023-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

include_guard(GLOBAL)

#[=======================================================================[.rst:
rapids_cython_init
------------------

.. versionadded:: v24.02.00

Perform standard initialization of any CMake build using scikit-build-core to create Python extension modules with Cython.

.. code-block:: cmake

  rapids_cython_init()

.. note::
  Use of the rapids-cython component of rapids-cmake requires scikit-build-core. The behavior of the functions provided by
  this component is undefined if they are invoked outside of a build managed by scikit-build-core.

Result Variables
^^^^^^^^^^^^^^^^
  :cmake:variable:`RAPIDS_CYTHON_INITIALIZED` will be set to TRUE.
  :cmake:variable:`CYTHON_FLAGS` will be set to a standard set of a flags to pass to the command line cython invocation.

#]=======================================================================]
macro(rapids_cython_init)
    list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cython.init")
    # Only initialize once.
    if(NOT DEFINED RAPIDS_CYTHON_INITIALIZED)
        # Verify that we are using scikit-build.
        if(NOT DEFINED SKBUILD)
            message(WARNING "rapids-cython expects scikit-build-core to be active before being used. \
          The SKBUILD variable is not currently set, indicating that scikit-build-core \
          is not active, so builds may behave unexpectedly.")
        else()
            # Access the variable to avoid unused variable warnings."
            message(TRACE "Accessing SKBUILD variable ${SKBUILD}")
        endif()

        find_package(
            Python
            COMPONENTS Interpreter Development.Module
            REQUIRED)
        find_program(CYTHON "cython")

        if(NOT CYTHON_FLAGS)
            set(CYTHON_FLAGS "--directive binding=True,embedsignature=True,always_allow_keywords=True")
        endif()

        # Flag
        set(RAPIDS_CYTHON_INITIALIZED TRUE)
    endif()
    list(POP_BACK CMAKE_MESSAGE_CONTEXT)
endmacro()
