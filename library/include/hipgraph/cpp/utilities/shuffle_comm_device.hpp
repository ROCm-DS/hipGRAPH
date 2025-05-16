#if !defined(HIPGRAPH_HDR___UTILITIES_SHUFFLE_COMM_CUH_)
#define HIPGRAPH_HDR___UTILITIES_SHUFFLE_COMM_CUH_ 1
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
#include <cugraph/utilities/shuffle_comm.cuh>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "utilities/shuffle_comm.cuh"
namespace hipgraph
{
    namespace backend = ::cuda;
}
#endif
#else
#include <rocgraph/cpp/utilities/shuffle_comm_device.hpp>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "utilities/shuffle_comm_device.hpp"
namespace hipgraph
{
    namespace backend = ::rocgraph;
}
#endif
#endif

#include "dataframe_buffer.hpp"
#include "device_comm.hpp"

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/partition.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/tuple.h>

#include <algorithm>
#include <hip/atomic>
#include <numeric>
#include <vector>

namespace hipgraph
{
    // Functions
    template <typename... OrigArgs>
    constexpr auto groupby_and_count = [](auto&&... args) {
        return ::hipgraph::backend::groupby_and_count<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto shuffle_values = [](auto&&... args) {
        return ::hipgraph::backend::shuffle_values<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto groupby_gpu_id_and_shuffle_values = [](auto&&... args) {
        return ::hipgraph::backend::groupby_gpu_id_and_shuffle_values<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto groupby_gpu_id_and_shuffle_kv_pairs = [](auto&&... args) {
        return ::hipgraph::backend::groupby_gpu_id_and_shuffle_kv_pairs<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
} // namespace hipgraph

#endif // HIPGRAPH_HDR___UTILITIES_SHUFFLE_COMM_CUH_
