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
#ifndef ARG_CHECK_HPP
#define ARG_CHECK_HPP

#include <hipgraph/hipgraph.h>

void verify_hipgraph_status(hipgraphStatus_t status,
                            hipgraphStatus_t expected_status,
                            const char*      message);

void verify_hipgraph_status_invalid_pointer(hipgraphStatus_t status, const char* message);

void verify_hipgraph_status_invalid_size(hipgraphStatus_t status, const char* message);

void verify_hipgraph_status_invalid_value(hipgraphStatus_t status, const char* message);

void verify_hipgraph_status_zero_pivot(hipgraphStatus_t status, const char* message);

void verify_hipgraph_status_invalid_handle(hipgraphStatus_t status);

void verify_hipgraph_status_internal_error(hipgraphStatus_t status, const char* message);

void verify_hipgraph_status_not_supported(hipgraphStatus_t status, const char* message);

void verify_hipgraph_status_success(hipgraphStatus_t status, const char* message);

#endif // ARG_CHECK_HPP
