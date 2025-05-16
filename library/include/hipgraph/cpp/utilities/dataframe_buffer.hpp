#if !defined(HIPGRAPH_HDR___UTILITIES_DATAFRAME_BUFFER_HPP_)
#define HIPGRAPH_HDR___UTILITIES_DATAFRAME_BUFFER_HPP_ 1
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
#include <cugraph/./utilities/dataframe_buffer.hpp>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "utilities/dataframe_buffer.hpp"
namespace hipgraph
{
    namespace backend = ::cuda;
}
#endif
#else
#include <rocgraph/cpp/./utilities/dataframe_buffer.hpp>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "utilities/dataframe_buffer.hpp"
namespace hipgraph
{
    namespace backend = ::rocgraph;
}
#endif
#endif

#include "thrust_tuple_utils.hpp"

#include <raft/core/handle.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <type_traits>

namespace hipgraph
{
    // Classes
    template <typename... Ts>
    using dataframe_element = ::hipgraph::backend::dataframe_element<Ts...>;

    // Functions
    template <typename... OrigArgs>
    constexpr auto allocate_dataframe_buffer = [](auto&&... args) {
        return ::hipgraph::backend::allocate_dataframe_buffer<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto reserve_dataframe_buffer = [](auto&&... args) {
        return ::hipgraph::backend::reserve_dataframe_buffer<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto resize_dataframe_buffer = [](auto&&... args) {
        return ::hipgraph::backend::resize_dataframe_buffer<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto shrink_to_fit_dataframe_buffer = [](auto&&... args) {
        return ::hipgraph::backend::shrink_to_fit_dataframe_buffer<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto size_dataframe_buffer = [](auto&&... args) {
        return ::hipgraph::backend::size_dataframe_buffer<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto get_dataframe_buffer_begin = [](auto&&... args) {
        return ::hipgraph::backend::get_dataframe_buffer_begin<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto get_dataframe_buffer_cbegin = [](auto&&... args) {
        return ::hipgraph::backend::get_dataframe_buffer_cbegin<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto get_dataframe_buffer_end = [](auto&&... args) {
        return ::hipgraph::backend::get_dataframe_buffer_end<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto get_dataframe_buffer_cend = [](auto&&... args) {
        return ::hipgraph::backend::get_dataframe_buffer_cend<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
} // namespace hipgraph

#endif // HIPGRAPH_HDR___UTILITIES_DATAFRAME_BUFFER_HPP_
