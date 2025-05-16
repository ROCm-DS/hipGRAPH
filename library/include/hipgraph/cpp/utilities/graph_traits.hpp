#if !defined(HIPGRAPH_HDR___UTILITIES_GRAPH_TRAITS_HPP_)
#define HIPGRAPH_HDR___UTILITIES_GRAPH_TRAITS_HPP_ 1
/*
 * SPDX-FileCopyrightText: Modifications Copyright (C) 2024 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 */
/*
 * Copyright (C) 2021-2024, NVIDIA CORPORATION.
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
#include <cugraph/./utilities/graph_traits.hpp>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "utilities/graph_traits.hpp"
namespace hipgraph
{
    namespace backend = ::cuda;
}
#endif
#else
#include <rocgraph/cpp/./utilities/graph_traits.hpp>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "utilities/graph_traits.hpp"
namespace hipgraph
{
    namespace backend = ::rocgraph;
}
#endif
#endif

#include <type_traits>

namespace hipgraph
{
    // Classes
    template <typename Src, typename Head, typename... Tail>
    using is_one_of = ::hipgraph::backend::is_one_of<Src, Head, Tail...>;

    template <typename Src>
    using is_one_of = ::hipgraph::backend::is_one_of<Src>;

    template <typename vertex_t, typename edge_t>
    using is_vertex_edge_combo = ::hipgraph::backend::is_vertex_edge_combo<vertex_t, edge_t>;

    template <typename vertex_t, typename edge_t>
    using is_vertex_edge_combo_legacy
        = ::hipgraph::backend::is_vertex_edge_combo_legacy<vertex_t, edge_t>;

    template <typename vertex_t, typename edge_t, typename weight_t>
    using is_candidate = ::hipgraph::backend::is_candidate<vertex_t, edge_t, weight_t>;

    template <typename vertex_t, typename edge_t, typename weight_t>
    using is_candidate_legacy
        = ::hipgraph::backend::is_candidate_legacy<vertex_t, edge_t, weight_t>;

} // namespace hipgraph

#endif // HIPGRAPH_HDR___UTILITIES_GRAPH_TRAITS_HPP_
