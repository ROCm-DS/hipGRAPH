#if !defined(HIPGRAPH_HDR___API_HELPERS_HPP_)
#define HIPGRAPH_HDR___API_HELPERS_HPP_ 1
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

// Andrei Schaffer, aschaffer@nvidia.com
//
// This is a collection of aggregates used by (parts of) the API defined in algorithms.hpp;
// These aggregates get propagated to the C-only API (which is why they're non-template aggregates)

#include <utility>

#if defined(USE_CUDA)
#include <cugraph/./api_helpers.hpp>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "api_helpers.hpp"
namespace hipgraph
{
    namespace backend = ::cuda;
}
#endif
#else
#include <rocgraph/cpp/./api_helpers.hpp>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "api_helpers.hpp"
namespace hipgraph
{
    namespace backend = ::rocgraph;
}
#endif
#endif

namespace hipgraph
{
    // Enums
    using sampling_strategy_t = ::hipgraph::backend::sampling_strategy_t;
    // Classes
    using sampling_params_t = ::hipgraph::backend::sampling_params_t;
} // namespace hipgraph

#endif // HIPGRAPH_HDR___API_HELPERS_HPP_
