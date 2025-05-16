#if !defined(HIPGRAPH_HDR___UTILITIES_DEVICE_COMM_HPP_)
#define HIPGRAPH_HDR___UTILITIES_DEVICE_COMM_HPP_ 1
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
#include <cugraph/./utilities/device_comm.hpp>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "utilities/device_comm.hpp"
namespace hipgraph
{
    namespace backend = ::cuda;
}
#endif
#else
#include <rocgraph/cpp/./utilities/device_comm.hpp>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "utilities/device_comm.hpp"
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

#include <thrust/detail/type_traits/iterator/is_discard_iterator.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/detail/any_assign.h>
#include <thrust/iterator/detail/normal_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/memory.h>
#include <thrust/tuple.h>

#include <type_traits>

namespace hipgraph
{
    // Functions
    template <typename... OrigArgs>
    constexpr auto device_isend = [](auto&&... args) {
        return ::hipgraph::backend::device_isend<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto device_irecv = [](auto&&... args) {
        return ::hipgraph::backend::device_irecv<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto device_sendrecv = [](auto&&... args) {
        return ::hipgraph::backend::device_sendrecv<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto device_multicast_sendrecv = [](auto&&... args) {
        return ::hipgraph::backend::device_multicast_sendrecv<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto device_bcast = [](auto&&... args) {
        return ::hipgraph::backend::device_bcast<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto device_allreduce = [](auto&&... args) {
        return ::hipgraph::backend::device_allreduce<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto device_reduce = [](auto&&... args) {
        return ::hipgraph::backend::device_reduce<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto device_allgatherv = [](auto&&... args) {
        return ::hipgraph::backend::device_allgatherv<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto device_gatherv = [](auto&&... args) {
        return ::hipgraph::backend::device_gatherv<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    constexpr auto device_group_start = [](auto&&... args) {
        return ::hipgraph::backend::device_group_start(std::forward<decltype(args)>(args)...);
    };
    constexpr auto device_group_end = [](auto&&... args) {
        return ::hipgraph::backend::device_group_end(std::forward<decltype(args)>(args)...);
    };
} // namespace hipgraph

#endif // HIPGRAPH_HDR___UTILITIES_DEVICE_COMM_HPP_
