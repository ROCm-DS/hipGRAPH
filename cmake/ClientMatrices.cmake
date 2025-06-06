# SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# ######################################################################################################################
# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# ######################################################################################################################

set(TEST_MATRICES
    SNAP/amazon0312
    Muite/Chebyshev4
    FEMLAB/sme3Dc
    Williams/webbase-1M
    Bova/rma10
    JGD_BIBD/bibd_22_8
    Williams/mac_econ_fwd500
    Williams/mc2depi
    Hamm/scircuit
    Sandia/ASIC_320k
    GHS_psdef/bmwcra_1
    HB/nos1
    HB/nos2
    HB/nos3
    HB/nos4
    HB/nos5
    HB/nos6
    HB/nos7
    DNVS/shipsec1)

set(TEST_MD5HASH
    f567e5f5029d052e3004bc69bb3f13f5
    e39879103dafab21f4cf942e0fe42a85
    a95eee14d980a9cfbbaf5df4a3c64713
    2d4c239daad6f12d66a1e6a2af44cbdb
    a899a0c48b9a58d081c52ffd88a84955
    455d5b699ea10232bbab5bc002219ae6
    f1b0e56fbb75d1d6862874e3d7d33060
    8c8633eada6455c1784269b213c85ea6
    3e62f7ea83914f7e20019aefb2a5176f
    fcfaf8a25c8f49b8d29f138f3c65c08f
    8a3cf5448a4fe73dcbdb5a16b326715f
    b203f7605cb1f20f83280061068f7ec7
    b0f812ffcc9469f0bf9be701205522c4
    f185514062a0eeabe86d2909275fe1dc
    04b781415202db404733ca0c159acbef
    c98e35f1cfd1ee8177f37bdae155a6e7
    c39375226aa5c495293003a5f637598f
    9a6481268847e6cf0d70671f2ff1ddcd
    73372e7d6a0848f8b19d64a924fab73e)

if(NOT CMAKE_MATRICES_DIR)
    message(FATAL_ERROR "Unspecified CMAKE_MATRICES_DIR")
endif()

if(NOT CONVERT_SOURCE)
    set(CONVERT_SOURCE ${CMAKE_SOURCE_DIR}/deps/convert.cpp)
endif()

# convert relative path to absolute
get_filename_component(PROJECT_BINARY_DIR "${PROJECT_BINARY_DIR}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
get_filename_component(CMAKE_MATRICES_DIR "${CMAKE_MATRICES_DIR}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")

file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR})

if(BUILD_ADDRESS_SANITIZER)
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} ${CONVERT_SOURCE} -O3 -fsanitize=address -shared-libasan -o
                            ${PROJECT_BINARY_DIR}/mtx2csr.exe RESULT_VARIABLE STATUS)
else()
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} ${CONVERT_SOURCE} -O3 -o ${PROJECT_BINARY_DIR}/mtx2csr.exe
                    RESULT_VARIABLE STATUS)
endif()

if(STATUS AND NOT STATUS EQUAL 0)
    message(FATAL_ERROR "mtx2csr.exe failed to build, aborting.")
endif()

list(LENGTH TEST_MATRICES len)
math(EXPR len1 "${len} - 1")

foreach(i RANGE 0 ${len1})
    list(GET TEST_MATRICES ${i} m)
    list(GET TEST_MD5HASH ${i} md5)

    string(REPLACE "/" ";" sep_m ${m})
    list(GET sep_m 0 dir)
    list(GET sep_m 1 mat)

    # Download test matrices if not already downloaded
    if(NOT EXISTS "${CMAKE_MATRICES_DIR}/${mat}.bin")
        if(NOT HIPGRAPH_MTX_DIR)
            # First try user specified mirror, if available
            if(DEFINED ENV{HIPGRAPH_TEST_MIRROR} AND NOT $ENV{HIPGRAPH_TEST_MIRROR} STREQUAL "")
                message(
                    "-- Downloading and extracting test matrix ${m}.tar.gz from user specified test mirror: $ENV{HIPGRAPH_TEST_MIRROR}"
                )
                file(
                    DOWNLOAD $ENV{HIPGRAPH_TEST_MIRROR}/${mat}.tar.gz ${CMAKE_MATRICES_DIR}/${mat}.tar.gz
                    INACTIVITY_TIMEOUT 3
                    STATUS DL)

                list(GET DL 0 stat)
                list(GET DL 1 msg)

                if(NOT stat EQUAL 0)
                    message(FATAL_ERROR "-- Timeout has been reached, specified test mirror is not reachable: ${msg}")
                endif()
            else()
                message("-- Downloading and extracting test matrix ${m}.tar.gz")
                file(
                    DOWNLOAD https://sparse.tamu.edu/MM/${m}.tar.gz ${CMAKE_MATRICES_DIR}/${mat}.tar.gz
                    INACTIVITY_TIMEOUT 3
                    STATUS DL)

                list(GET DL 0 stat)
                list(GET DL 1 msg)

                if(NOT stat EQUAL 0)
                    message("-- Timeout has been reached, trying mirror ...")
                    # Try again using ufl links
                    file(
                        DOWNLOAD https://www.cise.ufl.edu/research/sparse/MM/${m}.tar.gz
                        ${CMAKE_MATRICES_DIR}/${mat}.tar.gz
                        INACTIVITY_TIMEOUT 3
                        STATUS DL)

                    list(GET DL 0 stat)
                    list(GET DL 1 msg)

                    if(NOT stat EQUAL 0)
                        message(FATAL_ERROR "${msg}")
                    endif()
                endif()
            endif()

            # Check MD5 hash before continuing
            file(MD5 ${CMAKE_MATRICES_DIR}/${mat}.tar.gz hash)

            # Compare hash
            if(NOT hash STREQUAL md5)
                message(FATAL_ERROR "${mat}.tar.gz is corrupted")
            endif()

            execute_process(
                COMMAND tar xf ${mat}.tar.gz
                RESULT_VARIABLE STATUS
                WORKING_DIRECTORY ${CMAKE_MATRICES_DIR})
            if(STATUS AND NOT STATUS EQUAL 0)
                message(FATAL_ERROR "uncompressing failed, aborting.")
            endif()

            file(RENAME ${CMAKE_MATRICES_DIR}/${mat}/${mat}.mtx ${CMAKE_MATRICES_DIR}/${mat}.mtx)
        else()
            file(RENAME ${HIPGRAPH_MTX_DIR}/${mat}/${mat}.mtx ${CMAKE_MATRICES_DIR}/${mat}.mtx)
        endif()
        execute_process(
            COMMAND ${PROJECT_BINARY_DIR}/mtx2csr.exe ${mat}.mtx ${mat}.bin
            RESULT_VARIABLE STATUS
            WORKING_DIRECTORY ${CMAKE_MATRICES_DIR})
        if(STATUS AND NOT STATUS EQUAL 0)
            message(FATAL_ERROR "mtx2csr.exe failed, aborting.")
        else()
            message(STATUS "${mat} success.")
        endif()
        # TODO: add 'COMMAND_ERROR_IS_FATAL ANY' once cmake supported version is 3.19
        file(REMOVE_RECURSE ${CMAKE_MATRICES_DIR}/${mat}.tar.gz ${CMAKE_MATRICES_DIR}/${mat}
             ${CMAKE_MATRICES_DIR}/${mat}.mtx)

    endif()
endforeach()
