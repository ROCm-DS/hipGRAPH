#if !defined(HIPGRAPH_HDR___EDGE_PROPERTY_HPP_)
#define HIPGRAPH_HDR___EDGE_PROPERTY_HPP_ 1
/*
 * SPDX-FileCopyrightText: Modifications Copyright (C) 2024 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 */
/*
 * Copyright (C) 2022-2024, NVIDIA CORPORATION.
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
#include <cugraph/./edge_property.hpp>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "edge_property.hpp"
namespace hipgraph
{
    namespace backend = ::cuda;
}
#endif
#else
#include <rocgraph/cpp/./edge_property.hpp>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "edge_property.hpp"
namespace hipgraph
{
    namespace backend = ::rocgraph;
}
#endif
#endif

#include "utilities/dataframe_buffer.hpp"
#include "utilities/packed_bool_utils.hpp"
#include "utilities/thrust_tuple_utils.hpp"

#include <raft/core/handle.hpp>

#include <thrust/iterator/iterator_traits.h>
#include <thrust/optional.h>

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
    template <typename edge_t, typename ValueIterator, typename value_t>
    using edge_property_view_t
        = ::hipgraph::backend::edge_property_view_t<edge_t, ValueIterator, value_t>;

    using edge_dummy_property_view_t = ::hipgraph::backend::edge_dummy_property_view_t;
    template <typename GraphViewType, typename T>
    using edge_property_t = ::hipgraph::backend::edge_property_t<GraphViewType, T>;

    using edge_dummy_property_t = ::hipgraph::backend::edge_dummy_property_t;
} // namespace hipgraph

#endif // HIPGRAPH_HDR___EDGE_PROPERTY_HPP_
