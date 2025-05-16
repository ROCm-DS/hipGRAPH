#if !defined(HIPGRAPH_HDR___GRAPH_GENERATORS_HPP_)
#define HIPGRAPH_HDR___GRAPH_GENERATORS_HPP_ 1
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
#include <cugraph/./graph_generators.hpp>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "graph_generators.hpp"
namespace hipgraph
{
    namespace backend = ::cuda;
}
#endif
#else
#include <rocgraph/cpp/./graph_generators.hpp>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "graph_generators.hpp"
namespace hipgraph
{
    namespace backend = ::rocgraph;
}
#endif
#endif

#include <raft/core/handle.hpp>
#include <raft/random/rng_state.hpp>

#include <rmm/device_uvector.hpp>

#include <cstdint>
#include <optional>
#include <tuple>

namespace hipgraph
{
    // Enums
    using generator_distribution_t = ::hipgraph::backend::generator_distribution_t;
    // Functions
    template <typename... OrigArgs>
    constexpr auto generate_rmat_edgelist = [](auto&&... args) {
        return ::hipgraph::backend::generate_rmat_edgelist<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto generate_bipartite_rmat_edgelist = [](auto&&... args) {
        return ::hipgraph::backend::generate_bipartite_rmat_edgelist<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto generate_rmat_edgelists = [](auto&&... args) {
        return ::hipgraph::backend::generate_rmat_edgelists<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto generate_path_graph_edgelist = [](auto&&... args) {
        return ::hipgraph::backend::generate_path_graph_edgelist<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto generate_2d_mesh_graph_edgelist = [](auto&&... args) {
        return ::hipgraph::backend::generate_2d_mesh_graph_edgelist<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto generate_3d_mesh_graph_edgelist = [](auto&&... args) {
        return ::hipgraph::backend::generate_3d_mesh_graph_edgelist<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto generate_complete_graph_edgelist = [](auto&&... args) {
        return ::hipgraph::backend::generate_complete_graph_edgelist<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto generate_erdos_renyi_graph_edgelist_gnp = [](auto&&... args) {
        return ::hipgraph::backend::generate_erdos_renyi_graph_edgelist_gnp<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto generate_erdos_renyi_graph_edgelist_gnm = [](auto&&... args) {
        return ::hipgraph::backend::generate_erdos_renyi_graph_edgelist_gnm<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto symmetrize_edgelist_from_triangular = [](auto&&... args) {
        return ::hipgraph::backend::symmetrize_edgelist_from_triangular<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto scramble_vertex_ids = [](auto&&... args) {
        return ::hipgraph::backend::scramble_vertex_ids<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto combine_edgelists = [](auto&&... args) {
        return ::hipgraph::backend::combine_edgelists<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
} // namespace hipgraph

#endif // HIPGRAPH_HDR___GRAPH_GENERATORS_HPP_
