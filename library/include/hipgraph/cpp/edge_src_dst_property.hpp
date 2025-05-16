#if !defined(HIPGRAPH_HDR___EDGE_SRC_DST_PROPERTY_HPP_)
#define HIPGRAPH_HDR___EDGE_SRC_DST_PROPERTY_HPP_ 1
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
#include <cugraph/./edge_src_dst_property.hpp>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "edge_src_dst_property.hpp"
namespace hipgraph
{
    namespace backend = ::cuda;
}
#endif
#else
#include <rocgraph/cpp/./edge_src_dst_property.hpp>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "edge_src_dst_property.hpp"
namespace hipgraph
{
    namespace backend = ::rocgraph;
}
#endif
#endif

#include "utilities/dataframe_buffer.hpp"
#include "utilities/packed_bool_utils.hpp"
#include "utilities/thrust_tuple_utils.hpp"

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/host_span.hpp>
#include <raft/util/cudart_utils.hpp>

#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/optional.h>
#include <thrust/tuple.h>

#include <optional>
#include <type_traits>

namespace hipgraph
{
    // Functions
    template <typename... OrigArgs>
    constexpr auto view_concat = [](auto&&... args) {
        return ::hipgraph::backend::view_concat<OrigArgs...>(std::forward<decltype(args)>(args)...);
    };
    // Classes
    template <typename GraphViewType, typename T>
    using edge_src_property_t = ::hipgraph::backend::edge_src_property_t<GraphViewType, T>;

    template <typename GraphViewType, typename T>
    using edge_dst_property_t = ::hipgraph::backend::edge_dst_property_t<GraphViewType, T>;

    using edge_src_dummy_property_t = ::hipgraph::backend::edge_src_dummy_property_t;
    using edge_dst_dummy_property_t = ::hipgraph::backend::edge_dst_dummy_property_t;
} // namespace hipgraph

#endif // HIPGRAPH_HDR___EDGE_SRC_DST_PROPERTY_HPP_
