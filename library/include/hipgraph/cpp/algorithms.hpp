#if !defined(HIPGRAPH_HDR___ALGORITHMS_HPP_)
#define HIPGRAPH_HDR___ALGORITHMS_HPP_ 1
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
#include <cugraph/./algorithms.hpp>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "algorithms.hpp"
namespace hipgraph
{
    namespace backend = ::cuda;
}
#endif
#else
#include <rocgraph/cpp/./algorithms.hpp>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "algorithms.hpp"
namespace hipgraph
{
    namespace backend = ::rocgraph;
}
#endif
#endif

#include "api_helpers.hpp"
#include "dendrogram.hpp"
#include "edge_property.hpp"
#include "graph.hpp"
#include "graph_view.hpp"
#include "legacy/graph.hpp"
#include "legacy/internals.hpp"

#include <rmm/resource_ref.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <raft/random/rng_state.hpp>

#include <optional>
#include <tuple>

namespace hipgraph
{
    // Namespaces
    namespace ext_raft = ::hipgraph::backend::ext_raft;
    namespace dense    = ::hipgraph::backend::dense;
    namespace subgraph = ::hipgraph::backend::subgraph;
    // Enums
    enum class hipgraph_cc_t
    {
#if defined(USE_CUDA)
        HIPGRAPH_STRONG = static_cast<int>(::cuda::cugraph_cc_t::CUGRAPH_STRONG),
#else
        HIPGRAPH_STRONG = static_cast<int>(::rocgraph::rocgraph_cc_t::ROCGRAPH_STRONG),
#endif
        NUM_CONNECTIVITY_TYPES,
    };
#if defined(USE_CUDA)
    using hipgraph_backend_cc_t = ::cuda::cugraph_cc_t;
#else
    using hipgraph_backend_cc_t = ::rocgraph::rocgraph_cc_t;
#endif
    using k_core_degree_type_t     = ::hipgraph::backend::k_core_degree_type_t;
    using prior_sources_behavior_t = ::hipgraph::backend::prior_sources_behavior_t;
    // Functions
    template <typename... OrigArgs>
    constexpr auto jaccard = [](auto&&... args) {
        return ::hipgraph::backend::jaccard<OrigArgs...>(std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto jaccard_list = [](auto&&... args) {
        return ::hipgraph::backend::jaccard_list<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto overlap = [](auto&&... args) {
        return ::hipgraph::backend::overlap<OrigArgs...>(std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto overlap_list = [](auto&&... args) {
        return ::hipgraph::backend::overlap_list<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto force_atlas2 = [](auto&&... args) {
        return ::hipgraph::backend::force_atlas2<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto betweenness_centrality = [](auto&&... args) {
        return ::hipgraph::backend::betweenness_centrality<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto edge_betweenness_centrality = [](auto&&... args) {
        return ::hipgraph::backend::edge_betweenness_centrality<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };

    //template <typename... OrigArgs> constexpr auto connected_components = [] ( auto &&...args ) { return ::hipgraph::backend::connected_components<OrigArgs...>( std::forward<decltype(args)>( args )...); };
    template <typename VT, typename ET, typename WT>
    void connected_components(::hipgraph::backend::legacy::GraphCSRView<VT, ET, WT> const& graph,
                              hipgraph_cc_t connectivity_type,
                              VT*           labels)
    {
        const auto kind_ = static_cast<hipgraph_backend_cc_t>(connectivity_type);
        ::hipgraph::backend::connected_components(graph, kind_, labels);
    }

    template <typename... OrigArgs>
    constexpr auto hungarian = [](auto&&... args) {
        return ::hipgraph::backend::hungarian<OrigArgs...>(std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto louvain = [](auto&&... args) {
        return ::hipgraph::backend::louvain<OrigArgs...>(std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto flatten_dendrogram = [](auto&&... args) {
        return ::hipgraph::backend::flatten_dendrogram<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto leiden = [](auto&&... args) {
        return ::hipgraph::backend::leiden<OrigArgs...>(std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto ecg = [](auto&&... args) {
        return ::hipgraph::backend::ecg<OrigArgs...>(std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto minimum_spanning_tree = [](auto&&... args) {
        return ::hipgraph::backend::minimum_spanning_tree<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto bfs = [](auto&&... args) {
        return ::hipgraph::backend::bfs<OrigArgs...>(std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto extract_bfs_paths = [](auto&&... args) {
        return ::hipgraph::backend::extract_bfs_paths<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto sssp = [](auto&&... args) {
        return ::hipgraph::backend::sssp<OrigArgs...>(std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto od_shortest_distances = [](auto&&... args) {
        return ::hipgraph::backend::od_shortest_distances<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto pagerank = [](auto&&... args) {
        return ::hipgraph::backend::pagerank<OrigArgs...>(std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto eigenvector_centrality = [](auto&&... args) {
        return ::hipgraph::backend::eigenvector_centrality<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto hits = [](auto&&... args) {
        return ::hipgraph::backend::hits<OrigArgs...>(std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto katz_centrality = [](auto&&... args) {
        return ::hipgraph::backend::katz_centrality<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto extract_ego = [](auto&&... args) {
        return ::hipgraph::backend::extract_ego<OrigArgs...>(std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto random_walks = [](auto&&... args) {
        return ::hipgraph::backend::random_walks<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto uniform_random_walks = [](auto&&... args) {
        return ::hipgraph::backend::uniform_random_walks<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto biased_random_walks = [](auto&&... args) {
        return ::hipgraph::backend::biased_random_walks<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto node2vec_random_walks = [](auto&&... args) {
        return ::hipgraph::backend::node2vec_random_walks<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto weakly_connected_components = [](auto&&... args) {
        return ::hipgraph::backend::weakly_connected_components<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto core_number = [](auto&&... args) {
        return ::hipgraph::backend::core_number<OrigArgs...>(std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto k_core = [](auto&&... args) {
        return ::hipgraph::backend::k_core<OrigArgs...>(std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto uniform_neighbor_sample = [](auto&&... args) {
        return ::hipgraph::backend::uniform_neighbor_sample<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto triangle_count = [](auto&&... args) {
        return ::hipgraph::backend::triangle_count<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto k_truss = [](auto&&... args) {
        return ::hipgraph::backend::k_truss<OrigArgs...>(std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto jaccard_coefficients = [](auto&&... args) {
        return ::hipgraph::backend::jaccard_coefficients<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto sorensen_coefficients = [](auto&&... args) {
        return ::hipgraph::backend::sorensen_coefficients<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto overlap_coefficients = [](auto&&... args) {
        return ::hipgraph::backend::overlap_coefficients<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto jaccard_all_pairs_coefficients = [](auto&&... args) {
        return ::hipgraph::backend::jaccard_all_pairs_coefficients<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto sorensen_all_pairs_coefficients = [](auto&&... args) {
        return ::hipgraph::backend::sorensen_all_pairs_coefficients<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto overlap_all_pairs_coefficients = [](auto&&... args) {
        return ::hipgraph::backend::overlap_all_pairs_coefficients<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto k_hop_nbrs = [](auto&&... args) {
        return ::hipgraph::backend::k_hop_nbrs<OrigArgs...>(std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto maximal_independent_set = [](auto&&... args) {
        return ::hipgraph::backend::maximal_independent_set<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto vertex_coloring = [](auto&&... args) {
        return ::hipgraph::backend::vertex_coloring<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    // Classes
    using centrality_algorithm_metadata_t = ::hipgraph::backend::centrality_algorithm_metadata_t;
} // namespace hipgraph

#endif // HIPGRAPH_HDR___ALGORITHMS_HPP_
