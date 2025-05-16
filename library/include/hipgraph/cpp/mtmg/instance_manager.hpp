#if !defined(HIPGRAPH_HDR___MTMG_INSTANCE_MANAGER_HPP_)
#define HIPGRAPH_HDR___MTMG_INSTANCE_MANAGER_HPP_ 1
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
#include <cugraph/./mtmg/instance_manager.hpp>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "mtmg/instance_manager.hpp"
namespace hipgraph
{
    namespace backend = ::cuda;
}
#endif
#else
#include <rocgraph/cpp/./mtmg/instance_manager.hpp>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "mtmg/instance_manager.hpp"
namespace hipgraph
{
    namespace backend = ::rocgraph;
}
#endif
#endif

#include "mtmg/handle.hpp"

#include <raft/comms/std_comms.hpp>

#include <vector>

namespace hipgraph
{
    // Namespaces
    namespace mtmg = ::hipgraph::backend::mtmg;
} // namespace hipgraph

#endif // HIPGRAPH_HDR___MTMG_INSTANCE_MANAGER_HPP_
