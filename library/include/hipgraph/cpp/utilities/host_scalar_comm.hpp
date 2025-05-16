#if !defined(HIPGRAPH_HDR___UTILITIES_HOST_SCALAR_COMM_HPP_)
#define HIPGRAPH_HDR___UTILITIES_HOST_SCALAR_COMM_HPP_ 1
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
#include <cugraph/./utilities/host_scalar_comm.hpp>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "utilities/host_scalar_comm.hpp"
namespace hipgraph
{
    namespace backend = ::cuda;
}
#endif
#else
#include <rocgraph/cpp/./utilities/host_scalar_comm.hpp>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "utilities/host_scalar_comm.hpp"
namespace hipgraph
{
    namespace backend = ::rocgraph;
}
#endif
#endif

#include "error.hpp"
#include "thrust_tuple_utils.hpp"

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/tuple.h>

#include <numeric>
#include <type_traits>

namespace hipgraph
{
    // Functions
    template <typename... OrigArgs>
    constexpr auto host_scalar_allreduce = [](auto&&... args) {
        return ::hipgraph::backend::host_scalar_allreduce<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto host_scalar_reduce = [](auto&&... args) {
        return ::hipgraph::backend::host_scalar_reduce<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto host_scalar_bcast = [](auto&&... args) {
        return ::hipgraph::backend::host_scalar_bcast<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto host_scalar_allgather = [](auto&&... args) {
        return ::hipgraph::backend::host_scalar_allgather<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto host_scalar_scatter = [](auto&&... args) {
        return ::hipgraph::backend::host_scalar_scatter<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto host_scalar_gather = [](auto&&... args) {
        return ::hipgraph::backend::host_scalar_gather<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
} // namespace hipgraph

#endif // HIPGRAPH_HDR___UTILITIES_HOST_SCALAR_COMM_HPP_
