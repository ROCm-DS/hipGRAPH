// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*! \file */
#if !defined(HIPGRAPH_CAPI_NVIDIA_DETAIL_COMMON_)
#define HIPGRAPH_CAPI_NVIDIA_DETAIL_COMMON_
/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
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

/* Definitions common to all the nvidia_detail implementation sources. */

/* Check the cugraph version. */
#include <cugraph/version_config.hpp>
#if CUGRAPH_VERSION_MAJOR < 24 || (CUGRAPH_VERSION_MAJOR == 24 && CUGRAPH_VERSION_MINOR < 6)
#error "cugraph versions < 24.06 are not supported."
#endif

/* Disable the non-prefixed aliases so we can use the cuGraph headers directly. */
#define HIPGRAPH_NO_NONPREFIXED_ALIASES

/* Include the symbol export macro declarations. */
#include "hipgraph/hipgraph-export.h"

#endif
