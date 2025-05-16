#if !defined(HIPGRAPH_HDR___EDGE_PARTITION_DEVICE_VIEW_CUH_)
#define HIPGRAPH_HDR___EDGE_PARTITION_DEVICE_VIEW_CUH_ 1
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
#include <cugraph/edge_partition_device_view.cuh>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "edge_partition_device_view.cuh"
namespace hipgraph
{
    namespace backend = ::cuda;
}
#endif
#else
#include <rocgraph/cpp/edge_partition_device_view_device.hpp>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "edge_partition_device_view_device.hpp"
namespace hipgraph
{
    namespace backend = ::rocgraph;
}
#endif
#endif

#include "edge_partition_view.hpp"
#include "utilities/error.hpp"
#include "utilities/mask_utils_device.hpp"
#include "utilities/misc_utils_device.hpp"

#include <raft/core/device_span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/optional.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

#include <cassert>
#include <optional>
#include <type_traits>

namespace hipgraph
{
    // Classes
    template <typename vertex_t, typename edge_t, bool multi_gpu>
    using edge_partition_device_view_t
        = ::hipgraph::backend::edge_partition_device_view_t<vertex_t, edge_t, multi_gpu>;

    template <typename vertex_t, typename edge_t, bool multi_gpu>
    using edge_partition_device_view_t
        = ::hipgraph::backend::edge_partition_device_view_t<vertex_t, edge_t, multi_gpu>;

} // namespace hipgraph

#endif // HIPGRAPH_HDR___EDGE_PARTITION_DEVICE_VIEW_CUH_
