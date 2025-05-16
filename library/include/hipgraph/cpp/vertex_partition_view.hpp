#if !defined(HIPGRAPH_HDR___VERTEX_PARTITION_VIEW_HPP_)
#define HIPGRAPH_HDR___VERTEX_PARTITION_VIEW_HPP_ 1
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
#include <cugraph/./vertex_partition_view.hpp>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "vertex_partition_view.hpp"
namespace hipgraph
{
    namespace backend = ::cuda;
}
#endif
#else
#include <rocgraph/cpp/./vertex_partition_view.hpp>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "vertex_partition_view.hpp"
namespace hipgraph
{
    namespace backend = ::rocgraph;
}
#endif
#endif

#include <type_traits>

namespace hipgraph
{
    // Classes
    template <typename vertex_t, bool multi_gpu>
    using vertex_partition_view_t
        = ::hipgraph::backend::vertex_partition_view_t<vertex_t, multi_gpu>;

    template <typename vertex_t, bool multi_gpu>
    using vertex_partition_view_t
        = ::hipgraph::backend::vertex_partition_view_t<vertex_t, multi_gpu>;

} // namespace hipgraph

#endif // HIPGRAPH_HDR___VERTEX_PARTITION_VIEW_HPP_
