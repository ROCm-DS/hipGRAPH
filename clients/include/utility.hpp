// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/* ************************************************************************
 * Copyright (C) 2018-2020 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once
#ifndef TESTING_UTILITY_HPP
#define TESTING_UTILITY_HPP

#include <algorithm>
#include <assert.h>
#include <complex>
#include <math.h>
#include <random>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include <iostream>

#include <hip/hip_runtime_api.h>
#include <hipgraph/hipgraph.h>

#ifdef GOOGLE_TEST
#include "gtest/gtest.h"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif
std::string hipgraph_exepath();
/*!\file
 * \brief provide data initialization and timing utilities.
 */

#define CHECK_HIP_ERROR(error)                \
    if(error != hipSuccess)                   \
    {                                         \
        fprintf(stderr,                       \
                "error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(error),     \
                error,                        \
                __FILE__,                     \
                __LINE__);                    \
        exit(EXIT_FAILURE);                   \
    }

#if(!defined(CUDART_VERSION) || (CUDART_VERSION >= 11003))

#define CHECK_HIPGRAPH_ERROR_CASE__(token_) \
    case token_:                            \
        fprintf(stderr, #token_);           \
        break

#define CHECK_HIPGRAPH_ERROR(error)                                                 \
    {                                                                               \
        auto local_error = (error);                                                 \
        if(local_error != HIPGRAPH_SUCCESS)                                         \
        {                                                                           \
            fprintf(stderr, "hipGRAPH error: ");                                    \
            switch(local_error)                                                     \
            {                                                                       \
                CHECK_HIPGRAPH_ERROR_CASE__(HIPGRAPH_SUCCESS);                      \
                CHECK_HIPGRAPH_ERROR_CASE__(HIPGRAPH_UNKNOWN_ERROR);                \
                CHECK_HIPGRAPH_ERROR_CASE__(HIPGRAPH_INVALID_HANDLE);               \
                CHECK_HIPGRAPH_ERROR_CASE__(HIPGRAPH_ALLOC_ERROR);                  \
                CHECK_HIPGRAPH_ERROR_CASE__(HIPGRAPH_INVALID_INPUT);                \
                CHECK_HIPGRAPH_ERROR_CASE__(HIPGRAPH_NOT_IMPLEMENTED);              \
                CHECK_HIPGRAPH_ERROR_CASE__(HIPGRAPH_UNSUPPORTED_TYPE_COMBINATION); \
            }                                                                       \
            fprintf(stderr, "\n");                                                  \
            return local_error;                                                     \
        }                                                                           \
    }                                                                               \
    (void)0

#else

#define CHECK_HIPGRAPH_ERROR_CASE__(token_) \
    case token_:                            \
        fprintf(stderr, #token_);           \
        break

#define CHECK_HIPGRAPH_ERROR(error)                                                 \
    {                                                                               \
        auto local_error = (error);                                                 \
        if(local_error != HIPGRAPH_SUCCESS)                                         \
        {                                                                           \
            fprintf(stderr, "hipGRAPH error: ");                                    \
            switch(local_error)                                                     \
            {                                                                       \
                CHECK_HIPGRAPH_ERROR_CASE__(HIPGRAPH_SUCCESS);                      \
                CHECK_HIPGRAPH_ERROR_CASE__(HIPGRAPH_UNKNOWN_ERROR);                \
                CHECK_HIPGRAPH_ERROR_CASE__(HIPGRAPH_INVALID_HANDLE);               \
                CHECK_HIPGRAPH_ERROR_CASE__(HIPGRAPH_ALLOC_ERROR);                  \
                CHECK_HIPGRAPH_ERROR_CASE__(HIPGRAPH_INVALID_INPUT);                \
                CHECK_HIPGRAPH_ERROR_CASE__(HIPGRAPH_NOT_IMPLEMENTED);              \
                CHECK_HIPGRAPH_ERROR_CASE__(HIPGRAPH_UNSUPPORTED_TYPE_COMBINATION); \
            }                                                                       \
            fprintf(stderr, "\n");                                                  \
            return local_error;                                                     \
        }                                                                           \
    }                                                                               \
    (void)0

#endif

/* ============================================================================================ */
/*! \brief  Read matrix from mtx file in COO format */
static inline void read_mtx_value(std::istringstream& is, int& row, int& col, float& val)
{
    is >> row >> col >> val;
}

static inline void read_mtx_value(std::istringstream& is, int& row, int& col, double& val)
{
    is >> row >> col >> val;
}
template <typename T>
int read_mtx_matrix(const char*       filename,
                    int&              nrow,
                    int&              ncol,
                    int&              nnz,
                    std::vector<int>& row,
                    std::vector<int>& col,
                    std::vector<T>&   val)
{
    const char* env = getenv("GTEST_LISTENER");
    if(!env || strcmp(env, "NO_PASS_LINE_IN_LOG"))
    {
        printf("Reading matrix %s...", filename);
        fflush(stdout);
    }

    FILE* f = fopen(filename, "r");
    if(!f)
    {
        fprintf(stderr,
                "Failed to open matrix file %s because it does not exist. Please download the "
                "matrix file using the install script with -c flag.",
                filename);
        return -1;
    }

    char line[1024];

    // Check for banner
    if(!fgets(line, 1024, f))
    {
        return -1;
    }

    char banner[16];
    char array[16];
    char coord[16];
    char data[16];
    char type[16];

    // Extract banner
    if(sscanf(line, "%15s %15s %15s %15s %15s", banner, array, coord, data, type) != 5)
    {
        return -1;
    }

    // Convert to lower case
    for(char* p = array; *p != '\0'; *p = tolower(*p), p++)
        ;
    for(char* p = coord; *p != '\0'; *p = tolower(*p), p++)
        ;
    for(char* p = data; *p != '\0'; *p = tolower(*p), p++)
        ;
    for(char* p = type; *p != '\0'; *p = tolower(*p), p++)
        ;

    // Check banner
    if(strncmp(line, "%%MatrixMarket", 14) != 0)
    {
        return -1;
    }

    // Check array type
    if(strcmp(array, "matrix") != 0)
    {
        return -1;
    }

    // Check coord
    if(strcmp(coord, "coordinate") != 0)
    {
        return -1;
    }

    // Check data
    if(strcmp(data, "real") != 0 && strcmp(data, "integer") != 0 && strcmp(data, "pattern") != 0)
    {
        return -1;
    }

    // Check type
    if(strcmp(type, "general") != 0 && strcmp(type, "symmetric") != 0)
    {
        return -1;
    }

    // Symmetric flag
    int symm = !strcmp(type, "symmetric");

    // Skip comments
    while(fgets(line, 1024, f))
    {
        if(line[0] != '%')
        {
            break;
        }
    }

    // Read dimensions
    int snnz;

    sscanf(line, "%d %d %d", &nrow, &ncol, &snnz);
    nnz = symm ? (snnz - nrow) * 2 + nrow : snnz;

    std::vector<int> unsorted_row(nnz);
    std::vector<int> unsorted_col(nnz);
    std::vector<T>   unsorted_val(nnz);

    // Read entries
    int idx = 0;
    while(fgets(line, 1024, f))
    {
        if(idx >= nnz)
        {
            return true;
        }

        int irow;
        int icol;
        T   ival;

        std::istringstream ss(line);

        if(!strcmp(data, "pattern"))
        {
            ss >> irow >> icol;
            ival = static_cast<T>(1.0);
        }
        else
        {
            read_mtx_value(ss, irow, icol, ival);
        }

        // Assume zero-based indexes.
        --irow;
        --icol;

        unsorted_row[idx] = irow;
        unsorted_col[idx] = icol;
        unsorted_val[idx] = ival;

        ++idx;

        if(symm && irow != icol)
        {
            if(idx >= nnz)
            {
                return true;
            }

            unsorted_row[idx] = icol;
            unsorted_col[idx] = irow;
            unsorted_val[idx] = ival;
            ++idx;
        }
    }
    fclose(f);

    row.resize(nnz);
    col.resize(nnz);
    val.resize(nnz);

    // Sort by row and column index
    std::vector<int> perm(nnz);
    for(int i = 0; i < nnz; ++i)
    {
        perm[i] = i;
    }

    std::sort(perm.begin(), perm.end(), [&](const int& a, const int& b) {
        if(unsorted_row[a] < unsorted_row[b])
        {
            return true;
        }
        else if(unsorted_row[a] == unsorted_row[b])
        {
            return (unsorted_col[a] < unsorted_col[b]);
        }
        else
        {
            return false;
        }
    });

    for(int i = 0; i < nnz; ++i)
    {
        row[i] = unsorted_row[perm[i]];
        col[i] = unsorted_col[perm[i]];
        val[i] = unsorted_val[perm[i]];
    }

    if(!env || strcmp(env, "NO_PASS_LINE_IN_LOG"))
    {
        printf("done.\n");
        fflush(stdout);
    }

    return 0;
}

/* ============================================================================================ */
/*! \brief  Read matrix from binary file in CSR format */
template <typename I, typename J, typename T>
int read_bin_matrix(const char*     filename,
                    J&              nrow,
                    J&              ncol,
                    I&              nnz,
                    std::vector<I>& ptr,
                    std::vector<J>& col,
                    std::vector<T>& val)
{
    const char* env = getenv("GTEST_LISTENER");
    if(!env || strcmp(env, "NO_PASS_LINE_IN_LOG"))
    {
        printf("Reading matrix %s...", filename);
        fflush(stdout);
    }

    FILE* f = fopen(filename, "rb");
    if(!f)
    {
        return -1;
    }

    int err;

    int nrowf, ncolf, nnzf;

    err = fread(&nrowf, sizeof(int), 1, f);
    err |= fread(&ncolf, sizeof(int), 1, f);
    err |= fread(&nnzf, sizeof(int), 1, f);
    if(!err)
    {
        fclose(f);
        return -1;
    }
    nrow = (J)nrowf;
    ncol = (J)ncolf;
    nnz  = (I)nnzf;

    // Allocate memory
    std::vector<int>    ptrf(nrow + 1);
    std::vector<int>    colf(nnz);
    std::vector<double> valf(nnz);
    ptr.resize(nrow + 1);
    col.resize(nnz);
    val.resize(nnz);

    err |= fread(ptrf.data(), sizeof(int), nrow + 1, f);
    err |= fread(colf.data(), sizeof(int), nnz, f);
    err |= fread(valf.data(), sizeof(double), nnz, f);
    if(!err)
    {
        fclose(f);
        return -1;
    }

    fclose(f);

    for(J i = 0; i < nrow + 1; ++i)
    {
        ptr[i] = (I)ptrf[i];
    }

    for(I i = 0; i < nnz; ++i)
    {
        col[i] = (J)colf[i];
        val[i] = static_cast<T>(valf[i]);
    }

    if(!env || strcmp(env, "NO_PASS_LINE_IN_LOG"))
    {
        printf("done.\n");
        fflush(stdout);
    }

    return 0;
}

static inline float testing_neg(float val)
{
    return -val;
}

static inline double testing_neg(double val)
{
    return -val;
}

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================================ */
/*  query for hipgraph version and git commit SHA-1. */
void query_version(char* version);

/* ============================================================================================ */
/*  device query and print out their ID and name */
int query_device_property();

/*  set current device to device_id */
void set_device(int device_id);

/* ============================================================================================ */
/*  timing: HIP only provides very limited timers function clock() and not general;
            hipgraph sync CPU and device and use more accurate CPU timer*/

/*! \brief  CPU Timer(in microsecond): synchronize with the default device and return wall time */
double get_time_us(void);

/*! \brief  CPU Timer(in microsecond): synchronize with given queue/stream and return wall time */
double get_time_us_sync(hipStream_t stream);

#ifdef __cplusplus
}
#endif

/* ============================================================================================ */

/*! \brief Class used to parse command arguments in both client & gtest   */

class Arguments
{
public:
    int norm_check = 0;
    int unit_check = 1;
    int timing     = 0;

    int    numericboost{};
    double boosttol{};
    double boostval{};
    double boostvali{};

    std::string filename = "";

    Arguments& operator=(const Arguments& rhs) = default;
};

inline void missing_file_error_message(const char* filename)
{
    std::cerr << "#" << std::endl;
    std::cerr << "# error:" << std::endl;
    std::cerr << "# cannot open file '" << filename << "'" << std::endl;
    std::cerr << "#" << std::endl;
    std::cerr << "# PLEASE READ CAREFULLY !" << std::endl;
    std::cerr << "#" << std::endl;
    std::cerr << "# What could be the reason of this error: " << std::endl;
    std::cerr << "# You are running the testing application and it expects to find the file "
                 "at the specified location. This means that either you did not download the test "
                 "matrices, or you did not specify the location of the folder containing your "
                 "files. If you want to specify the location of the folder containing your files, "
                 "then you will find the needed information with 'hipgraph-test --help'."
                 "If you need to download matrices, then a cmake script "
                 "'hipgraph_clientmatrices.cmake' is available from the hipgraph client package."
              << std::endl;
    std::cerr << "#" << std::endl;
    std::cerr
        << "# Examples: 'hipgraph_clientmatrices.cmake -DCMAKE_MATRICES_DIR=<path-of-your-folder>'"
        << std::endl;
    std::cerr << "#           'hipgraph-test --matrices-dir <path-of-your-folder>'" << std::endl;
    std::cerr << "# (or        'export "
                 "HIPGRAPH_CLIENTS_MATRICES_DIR=<path-of-your-folder>;hipgraph-test')"
              << std::endl;
    std::cerr << "#" << std::endl;
}

const char* get_hipgraph_clients_matrices_dir();

inline std::string get_filename(const std::string& bin_file)
{
    const char* matrices_dir = get_hipgraph_clients_matrices_dir();
    if(matrices_dir == nullptr)
    {
        matrices_dir = getenv("HIPGRAPH_CLIENTS_MATRICES_DIR");
    }

    std::string r;
    if(matrices_dir != nullptr)
    {
        r = std::string(matrices_dir) + "/" + bin_file;
    }
    else
    {
        r = hipgraph_exepath() + "../matrices/" + bin_file;
    }

    FILE* tmpf = fopen(r.c_str(), "r");
    if(!tmpf)
    {
        missing_file_error_message(r.c_str());
        std::cerr << "exit(HIPGRAPH_UNKNOWN_ERROR)" << std::endl;
        exit(HIPGRAPH_UNKNOWN_ERROR);
    }
    else
    {
        fclose(tmpf);
    }
    return r;
}

#endif // TESTING_UTILITY_HPP
