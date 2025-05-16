#if !defined(HIPGRAPH_HDR___GRAPH_VIEW_HPP_)
#define HIPGRAPH_HDR___GRAPH_VIEW_HPP_ 1
/*
 * SPDX-FileCopyrightText: Modifications Copyright (C) 2024 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 */
/*
 * Copyright (C) 2020-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <utility>

#if defined(USE_CUDA)
#include <cugraph/./graph_view.hpp>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "graph_view.hpp"
namespace hipgraph
{
    namespace backend = ::cuda;
}
#endif
#else
#include <rocgraph/cpp/./graph_view.hpp>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "graph_view.hpp"
namespace hipgraph
{
    namespace backend = ::rocgraph;
}
#endif
#endif

#include "edge_partition_view.hpp"
#include "edge_property.hpp"
#include "partition_manager.hpp"
#include "utilities/error.hpp"
#include "vertex_partition_view.hpp"

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/host_span.hpp>

#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace hipgraph
{
    // Classes
    template <typename vertex_t>
    using partition_t = ::hipgraph::backend::partition_t<vertex_t>;

    using graph_properties_t = ::hipgraph::backend::graph_properties_t;
    template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
    using graph_view_meta_t
        = ::hipgraph::backend::graph_view_meta_t<vertex_t, edge_t, store_transposed, multi_gpu>;

    template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
    using graph_view_meta_t
        = ::hipgraph::backend::graph_view_meta_t<vertex_t, edge_t, store_transposed, multi_gpu>;

    template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
    using graph_view_t
        = ::hipgraph::backend::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>;

    template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
    using graph_view_t
        = ::hipgraph::backend::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>;

} // namespace hipgraph

#endif // HIPGRAPH_HDR___GRAPH_VIEW_HPP_
