// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0
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

#include "arg_check.hpp"

#include <hip/hip_runtime_api.h>
#include <hipgraph/hipgraph.h>
#include <iostream>

#ifdef GOOGLE_TEST
#include <gtest/gtest.h>
#endif

#define PRINT_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                \
    {                                                             \
        hipError_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                    \
        {                                                         \
            fprintf(stderr,                                       \
                    "hip error code: %d at %s:%d\n",              \
                    TMP_STATUS_FOR_CHECK,                         \
                    __FILE__,                                     \
                    __LINE__);                                    \
        }                                                         \
    }

void verify_hipgraph_status(hipgraphStatus_t status,
                            hipgraphStatus_t expected_status,
                            const char*      message)
{
#ifdef GOOGLE_TEST
    ASSERT_EQ(status, expected_status);
#else
    if(status != expected_status)
    {
        std::cerr << "hipGRAPH TEST ERROR: status(=" << status
                  << ") != expected_status(= " << expected_status << "), ";
        std::cerr << message << std::endl;
    }
#endif
}

void verify_hipgraph_status_invalid_pointer(hipgraphStatus_t status, const char* message)
{
#ifdef GOOGLE_TEST
    ASSERT_EQ(status, HIPGRAPH_STATUS_INVALID_VALUE);
#else
    if(status != HIPGRAPH_STATUS_INVALID_VALUE)
    {
        std::cerr << "hipGRAPH TEST ERROR: status != HIPGRAPH_STATUS_INVALID_VALUE, ";
        std::cerr << message << std::endl;
    }
#endif
}

void verify_hipgraph_status_invalid_size(hipgraphStatus_t status, const char* message)
{
#ifdef GOOGLE_TEST
    ASSERT_EQ(status, HIPGRAPH_STATUS_INVALID_VALUE);
#else
    if(status != HIPGRAPH_STATUS_INVALID_VALUE)
    {
        std::cerr << "hipGRAPH TEST ERROR: status != HIPGRAPH_STATUS_INVALID_VALUE, ";
        std::cerr << message << std::endl;
    }
#endif
}

void verify_hipgraph_status_invalid_value(hipgraphStatus_t status, const char* message)
{
#ifdef GOOGLE_TEST
    ASSERT_EQ(status, HIPGRAPH_STATUS_INVALID_VALUE);
#else
    if(status != HIPGRAPH_STATUS_INVALID_VALUE)
    {
        std::cerr << "hipGRAPH TEST ERROR: status != HIPGRAPH_STATUS_INVALID_VALUE, ";
        std::cerr << message << std::endl;
    }
#endif
}

void verify_hipgraph_status_zero_pivot(hipgraphStatus_t status, const char* message)
{
#ifdef GOOGLE_TEST
    ASSERT_EQ(status, HIPGRAPH_STATUS_ZERO_PIVOT);
#else
    if(status != HIPGRAPH_STATUS_ZERO_PIVOT)
    {
        std::cerr << "hipGRAPH TEST ERROR: status != HIPGRAPH_STATUS_ZERO_PIVOT, ";
        std::cerr << message << std::endl;
    }
#endif
}

void verify_hipgraph_status_invalid_handle(hipgraphStatus_t status)
{
#ifdef GOOGLE_TEST
    ASSERT_EQ(status, HIPGRAPH_STATUS_INVALID_VALUE);
#else
    if(status != HIPGRAPH_STATUS_INVALID_VALUE)
    {
        std::cerr << "hipGRAPH TEST ERROR: status != HIPGRAPH_STATUS_INVALID_VALUE" << std::endl;
    }
#endif
}

void verify_hipgraph_status_internal_error(hipgraphStatus_t status, const char* message)
{
#ifdef GOOGLE_TEST
    ASSERT_EQ(status, HIPGRAPH_STATUS_INTERNAL_ERROR);
#else
    if(status != HIPGRAPH_STATUS_INTERNAL_ERROR)
    {
        std::cerr << "hipGRAPH TEST ERROR: status != HIPGRAPH_STATUS_INTERNAL_ERROR, ";
        std::cerr << message << std::endl;
    }
#endif
}

void verify_hipgraph_status_not_supported(hipgraphStatus_t status, const char* message)
{
#ifdef GOOGLE_TEST
    ASSERT_EQ(status, HIPGRAPH_STATUS_NOT_SUPPORTED);
#else
    if(status != HIPGRAPH_STATUS_NOT_SUPPORTED)
    {
        std::cerr << "hipGRAPH TEST ERROR: status != HIPGRAPH_STATUS_NOT_SUPPORTED, ";
        std::cerr << message << std::endl;
    }
#endif
}

void verify_hipgraph_status_success(hipgraphStatus_t status, const char* message)
{
#ifdef GOOGLE_TEST
    ASSERT_EQ(status, HIPGRAPH_STATUS_SUCCESS);
#else
    if(status != HIPGRAPH_STATUS_SUCCESS)
    {
        std::cerr << "hipGRAPH TEST ERROR: status != HIPGRAPH_STATUS_SUCCESS, ";
        std::cerr << message << std::endl;
    }
#endif
}
