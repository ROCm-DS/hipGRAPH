#if !defined(HIPGRAPH_HDR___UTILITIES_ERROR_HPP_)
#define HIPGRAPH_HDR___UTILITIES_ERROR_HPP_ 1
/*
 * SPDX-FileCopyrightText: Modifications Copyright (C) 2024 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 */
/*
 * Copyright (C) 2019-2024, NVIDIA CORPORATION.
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
#include <cugraph/./utilities/error.hpp>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "utilities/error.hpp"
namespace hipgraph
{
    namespace backend = ::cuda;
}
#endif
#else
#include <rocgraph/cpp/./utilities/error.hpp>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "utilities/error.hpp"
namespace hipgraph
{
    namespace backend = ::rocgraph;
}
#endif
#endif

#include <raft/core/error.hpp>

namespace hipgraph
{
    // Classes
    using logic_error = ::hipgraph::backend::logic_error;
} // namespace hipgraph

#define HIPGRAPH_EXPECTS(cond, fmt, ...)                                    \
    do                                                                      \
    {                                                                       \
        if(!(cond))                                                         \
        {                                                                   \
            std::string msg{};                                              \
            SET_ERROR_MSG(msg, "hipGRAPH failure at ", fmt, ##__VA_ARGS__); \
            throw hipgraph::logic_error(msg);                               \
        }                                                                   \
    } while(0)

#define HIPGRAPH_FAIL(fmt, ...)                                         \
    do                                                                  \
    {                                                                   \
        std::string msg{};                                              \
        SET_ERROR_MSG(msg, "hipGRAPH failure at ", fmt, ##__VA_ARGS__); \
        throw hipgraph::logic_error(msg);                               \
    } while(0)
#endif // HIPGRAPH_HDR___UTILITIES_ERROR_HPP_
