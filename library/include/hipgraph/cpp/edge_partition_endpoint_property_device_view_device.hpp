#if !defined(HIPGRAPH_HDR___EDGE_PARTITION_ENDPOINT_PROPERTY_DEVICE_VIEW_CUH_)
#define HIPGRAPH_HDR___EDGE_PARTITION_ENDPOINT_PROPERTY_DEVICE_VIEW_CUH_ 1
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
#include <cugraph/edge_partition_endpoint_property_device_view.cuh>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "edge_partition_endpoint_property_device_view.cuh"
namespace hipgraph
{
    namespace backend = ::cuda;
}
#endif
#else
#include <rocgraph/cpp/edge_partition_endpoint_property_device_view_device.hpp>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "edge_partition_endpoint_property_device_view_device.hpp"
namespace hipgraph
{
    namespace backend = ::rocgraph;
}
#endif
#endif

#include "edge_src_dst_property.hpp"
#include "utilities/atomic_ops_device.hpp"
#include "utilities/device_functors_device.hpp"
#include "utilities/packed_bool_utils.hpp"

#include <raft/core/device_span.hpp>

#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/optional.h>

namespace hipgraph
{
} // namespace hipgraph

#endif // HIPGRAPH_HDR___EDGE_PARTITION_ENDPOINT_PROPERTY_DEVICE_VIEW_CUH_
