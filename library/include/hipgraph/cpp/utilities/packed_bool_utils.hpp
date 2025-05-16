#if !defined(HIPGRAPH_HDR___UTILITIES_PACKED_BOOL_UTILS_HPP_)
#define HIPGRAPH_HDR___UTILITIES_PACKED_BOOL_UTILS_HPP_ 1
/*
 * SPDX-FileCopyrightText: Modifications Copyright (C) 2024 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 */
/*
 * Copyright (C) 2023-2024, NVIDIA CORPORATION.
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
#include <cugraph/./utilities/packed_bool_utils.hpp>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "utilities/packed_bool_utils.hpp"
namespace hipgraph
{
    namespace backend = ::cuda;
}
#endif
#else
#include <rocgraph/cpp/./utilities/packed_bool_utils.hpp>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "utilities/packed_bool_utils.hpp"
namespace hipgraph
{
    namespace backend = ::rocgraph;
}
#endif
#endif

#include "thrust_tuple_utils.hpp"
#include <raft/util/cudart_utils.hpp>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/tuple.h>

#include <limits>
#include <type_traits>
#include <utility>

namespace hipgraph
{
    // Functions
    template <typename... OrigArgs>
    constexpr auto is_packed_bool = [](auto&&... args) {
        return ::hipgraph::backend::is_packed_bool<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto has_packed_bool_element = [](auto&&... args) {
        return ::hipgraph::backend::has_packed_bool_element<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    constexpr auto packed_bools_per_word = [](auto&&... args) {
        return ::hipgraph::backend::packed_bools_per_word(std::forward<decltype(args)>(args)...);
    };
    constexpr auto packed_bool_size = [](auto&&... args) {
        return ::hipgraph::backend::packed_bool_size(std::forward<decltype(args)>(args)...);
    };
    constexpr auto packed_bool_mask = [](auto&&... args) {
        return ::hipgraph::backend::packed_bool_mask(std::forward<decltype(args)>(args)...);
    };
    constexpr auto packed_bool_full_mask = [](auto&&... args) {
        return ::hipgraph::backend::packed_bool_full_mask(std::forward<decltype(args)>(args)...);
    };
    constexpr auto packed_bool_partial_mask = [](auto&&... args) {
        return ::hipgraph::backend::packed_bool_partial_mask(std::forward<decltype(args)>(args)...);
    };
    constexpr auto packed_bool_empty_mask = [](auto&&... args) {
        return ::hipgraph::backend::packed_bool_empty_mask(std::forward<decltype(args)>(args)...);
    };
    constexpr auto packed_bool_offset = [](auto&&... args) {
        return ::hipgraph::backend::packed_bool_offset(std::forward<decltype(args)>(args)...);
    };

} // namespace hipgraph

#endif // HIPGRAPH_HDR___UTILITIES_PACKED_BOOL_UTILS_HPP_
