#if !defined(HIPGRAPH_HDR___UTILITIES_ATOMIC_OPS_CUH_)
#define HIPGRAPH_HDR___UTILITIES_ATOMIC_OPS_CUH_ 1
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
#include <cugraph/utilities/atomic_ops.cuh>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "utilities/atomic_ops.cuh"
namespace hipgraph
{
    namespace backend = ::cuda;
}
#endif
#else
#include <rocgraph/cpp/utilities/atomic_ops_device.hpp>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "utilities/atomic_ops_device.hpp"
namespace hipgraph
{
    namespace backend = ::rocgraph;
}
#endif
#endif

#include "thrust_tuple_utils.hpp"

#include <raft/util/device_atomics.cuh>

#include <thrust/detail/type_traits/iterator/is_discard_iterator.h>
#include <thrust/iterator/detail/any_assign.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/memory.h>
#include <thrust/tuple.h>

namespace hipgraph
{
    // Functions
    template <typename... OrigArgs>
    constexpr auto atomic_and = [](auto&&... args) {
        return ::hipgraph::backend::atomic_and<OrigArgs...>(std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto atomic_or = [](auto&&... args) {
        return ::hipgraph::backend::atomic_or<OrigArgs...>(std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto atomic_add = [](auto&&... args) {
        return ::hipgraph::backend::atomic_add<OrigArgs...>(std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto elementwise_atomic_cas = [](auto&&... args) {
        return ::hipgraph::backend::elementwise_atomic_cas<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto elementwise_atomic_min = [](auto&&... args) {
        return ::hipgraph::backend::elementwise_atomic_min<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto elementwise_atomic_max = [](auto&&... args) {
        return ::hipgraph::backend::elementwise_atomic_max<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
} // namespace hipgraph

#endif // HIPGRAPH_HDR___UTILITIES_ATOMIC_OPS_CUH_
