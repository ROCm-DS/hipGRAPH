/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 */
#if !defined(_hipgraph__/LEGACY/INTERNALS_HPP_)
#define _hipgraph__/LEGACY/INTERNALS_HPP_ 1

#include <cugraph/./legacy/internals.hpp>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "./legacy/internals.hpp"
namespace hipgraph
{
    namespace backend = ::cugraph;
}

// Legacy and future...
#if 0
#if defined(USE_CUDA)
#include <cugraph/./legacy/internals.hpp>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "./legacy/internals.hpp"
namespace hipgraph
{
    namespace backend = ::cugraph;
}
#endif
#else
#include <rocgraph/./legacy/internals.hpp>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "./legacy/internals.hpp"
namespace rocgraph = cugraph; // For now.
namespace hipgraph
{
    namespace backend = ::rocgraph;
}
#endif
#endif
#endif

namespace hipgraph
{
inline namespace compat_v25_02 {
  using namespace ::hipgraph::backend;
}
}

#endif // _hipgraph__/LEGACY/INTERNALS_HPP_

