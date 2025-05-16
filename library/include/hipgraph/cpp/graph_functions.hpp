#if !defined(HIPGRAPH_HDR___GRAPH_FUNCTIONS_HPP_)
#define HIPGRAPH_HDR___GRAPH_FUNCTIONS_HPP_ 1
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
#include <cugraph/./graph_functions.hpp>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "graph_functions.hpp"
namespace hipgraph
{
    namespace backend = ::cuda;
}
#endif
#else
#include <rocgraph/cpp/./graph_functions.hpp>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "graph_functions.hpp"
namespace hipgraph
{
    namespace backend = ::rocgraph;
}
#endif
#endif

#include "edge_property.hpp"
#include "graph.hpp"
#include "graph_view.hpp"

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <raft/random/rng_state.hpp>

#include <rmm/device_uvector.hpp>

#include <memory>
#include <optional>
#include <tuple>
#include <vector>

namespace hipgraph
{
    // Functions
    template <typename... OrigArgs>
    constexpr auto renumber_edgelist = [](auto&&... args) {
        return ::hipgraph::backend::renumber_edgelist<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto renumber_ext_vertices = [](auto&&... args) {
        return ::hipgraph::backend::renumber_ext_vertices<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto unrenumber_local_int_vertices = [](auto&&... args) {
        return ::hipgraph::backend::unrenumber_local_int_vertices<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto unrenumber_int_vertices = [](auto&&... args) {
        return ::hipgraph::backend::unrenumber_int_vertices<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto unrenumber_local_int_edges = [](auto&&... args) {
        return ::hipgraph::backend::unrenumber_local_int_edges<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto renumber_local_ext_vertices = [](auto&&... args) {
        return ::hipgraph::backend::renumber_local_ext_vertices<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto decompress_to_edgelist = [](auto&&... args) {
        return ::hipgraph::backend::decompress_to_edgelist<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto symmetrize_edgelist = [](auto&&... args) {
        return ::hipgraph::backend::symmetrize_edgelist<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto symmetrize_graph = [](auto&&... args) {
        return ::hipgraph::backend::symmetrize_graph<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto transpose_graph = [](auto&&... args) {
        return ::hipgraph::backend::transpose_graph<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto transpose_graph_storage = [](auto&&... args) {
        return ::hipgraph::backend::transpose_graph_storage<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto coarsen_graph = [](auto&&... args) {
        return ::hipgraph::backend::coarsen_graph<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto relabel = [](auto&&... args) {
        return ::hipgraph::backend::relabel<OrigArgs...>(std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto extract_induced_subgraphs = [](auto&&... args) {
        return ::hipgraph::backend::extract_induced_subgraphs<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto create_graph_from_edgelist = [](auto&&... args) {
        return ::hipgraph::backend::create_graph_from_edgelist<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto get_two_hop_neighbors = [](auto&&... args) {
        return ::hipgraph::backend::get_two_hop_neighbors<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto compute_in_weight_sums = [](auto&&... args) {
        return ::hipgraph::backend::compute_in_weight_sums<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto compute_out_weight_sums = [](auto&&... args) {
        return ::hipgraph::backend::compute_out_weight_sums<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto compute_max_in_weight_sum = [](auto&&... args) {
        return ::hipgraph::backend::compute_max_in_weight_sum<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto compute_max_out_weight_sum = [](auto&&... args) {
        return ::hipgraph::backend::compute_max_out_weight_sum<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto compute_total_edge_weight = [](auto&&... args) {
        return ::hipgraph::backend::compute_total_edge_weight<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto select_random_vertices = [](auto&&... args) {
        return ::hipgraph::backend::select_random_vertices<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto renumber_sampled_edgelist = [](auto&&... args) {
        return ::hipgraph::backend::renumber_sampled_edgelist<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto remove_self_loops = [](auto&&... args) {
        return ::hipgraph::backend::remove_self_loops<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto remove_multi_edges = [](auto&&... args) {
        return ::hipgraph::backend::remove_multi_edges<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto shuffle_external_vertices = [](auto&&... args) {
        return ::hipgraph::backend::shuffle_external_vertices<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto shuffle_external_vertex_value_pairs = [](auto&&... args) {
        return ::hipgraph::backend::shuffle_external_vertex_value_pairs<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto shuffle_external_edges = [](auto&&... args) {
        return ::hipgraph::backend::shuffle_external_edges<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    // Classes
    template <typename vertex_t, typename edge_t, bool multi_gpu>
    using renumber_meta_t = ::hipgraph::backend::renumber_meta_t<vertex_t, edge_t, multi_gpu>;

    template <typename vertex_t, typename edge_t, bool multi_gpu>
    using renumber_meta_t = ::hipgraph::backend::renumber_meta_t<vertex_t, edge_t, multi_gpu>;

} // namespace hipgraph

#endif // HIPGRAPH_HDR___GRAPH_FUNCTIONS_HPP_
