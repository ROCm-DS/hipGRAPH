#if !defined(HIPGRAPH_HDR___GRAPH_HPP_)
#define HIPGRAPH_HDR___GRAPH_HPP_ 1
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
#include <cugraph/./graph.hpp>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "graph.hpp"
namespace hipgraph
{
    namespace backend = ::cuda;
}
#endif
#else
#include <rocgraph/cpp/./graph.hpp>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "graph.hpp"
namespace hipgraph
{
    namespace backend = ::rocgraph;
}
#endif
#endif

#include "graph_view.hpp"
#include "utilities/error.hpp"

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <cstddef>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

namespace hipgraph
{
    // Functions
    template <typename... OrigArgs>
    constexpr auto is_valid_vertex = [](auto&&... args) {
        return ::hipgraph::backend::is_valid_vertex<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    // Classes
    template <typename vertex_t, typename edge_t, bool multi_gpu>
    using graph_meta_t = ::hipgraph::backend::graph_meta_t<vertex_t, edge_t, multi_gpu>;

    template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
    using graph_t = ::hipgraph::backend::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>;

    template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
    using graph_t = ::hipgraph::backend::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>;

    template <typename T>
    using invalid_idx = ::hipgraph::backend::invalid_idx<T>;

    template <typename T>
    using invalid_idx = ::hipgraph::backend::invalid_idx<T>;

    template <typename vertex_t>
    using invalid_vertex_id = ::hipgraph::backend::invalid_vertex_id<vertex_t>;

    template <typename edge_t>
    using invalid_edge_id = ::hipgraph::backend::invalid_edge_id<edge_t>;

    template <typename vertex_t>
    using invalid_component_id = ::hipgraph::backend::invalid_component_id<vertex_t>;

} // namespace hipgraph

// Forward declare graph types already instantiated in the compiled library.
#include "eidecl_graph.hpp"

#endif // HIPGRAPH_HDR___GRAPH_HPP_
