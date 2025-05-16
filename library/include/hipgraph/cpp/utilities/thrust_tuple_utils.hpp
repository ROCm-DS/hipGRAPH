#if !defined(HIPGRAPH_HDR___UTILITIES_THRUST_TUPLE_UTILS_HPP_)
#define HIPGRAPH_HDR___UTILITIES_THRUST_TUPLE_UTILS_HPP_ 1
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
#include <cugraph/./utilities/thrust_tuple_utils.hpp>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "utilities/thrust_tuple_utils.hpp"
namespace hipgraph
{
    namespace backend = ::cuda;
}
#endif
#else
#include <rocgraph/cpp/./utilities/thrust_tuple_utils.hpp>
#if !defined(HIPGRAPH_BACKEND_DECLARED_)
#define HIPGRAPH_BACKEND_DECLARED_ "utilities/thrust_tuple_utils.hpp"
namespace hipgraph
{
    namespace backend = ::rocgraph;
}
#endif
#endif

#include <rmm/device_uvector.hpp>

#include <thrust/tuple.h>

#include <array>
#include <type_traits>

namespace hipgraph
{
    // Functions
    template <typename... OrigArgs>
    constexpr auto sum_thrust_tuple_element_sizes = [](auto&&... args) {
        return ::hipgraph::backend::sum_thrust_tuple_element_sizes<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto thrust_tuple_to_std_tuple = [](auto&&... args) {
        return ::hipgraph::backend::thrust_tuple_to_std_tuple<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto std_tuple_to_thrust_tuple = [](auto&&... args) {
        return ::hipgraph::backend::std_tuple_to_thrust_tuple<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto to_thrust_tuple = [](auto&&... args) {
        return ::hipgraph::backend::to_thrust_tuple<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto to_thrust_iterator_tuple = [](auto&&... args) {
        return ::hipgraph::backend::to_thrust_iterator_tuple<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto thrust_tuple_cat = [](auto&&... args) {
        return ::hipgraph::backend::thrust_tuple_cat<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto thrust_tuple_of_arithmetic_numeric_limits_lowest = [](auto&&... args) {
        return ::hipgraph::backend::thrust_tuple_of_arithmetic_numeric_limits_lowest<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto thrust_tuple_of_arithmetic_numeric_limits_max = [](auto&&... args) {
        return ::hipgraph::backend::thrust_tuple_of_arithmetic_numeric_limits_max<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    template <typename... OrigArgs>
    constexpr auto get_first_of_pack = [](auto&&... args) {
        return ::hipgraph::backend::get_first_of_pack<OrigArgs...>(
            std::forward<decltype(args)>(args)...);
    };
    // Classes
    template <typename... Ts>
    using is_thrust_tuple = ::hipgraph::backend::is_thrust_tuple<Ts...>;

    template <typename... Ts>
    using is_thrust_tuple_of_arithmetic = ::hipgraph::backend::is_thrust_tuple_of_arithmetic<Ts...>;

    template <typename... Ts>
    using is_std_tuple = ::hipgraph::backend::is_std_tuple<Ts...>;

    //template <typename T, template <typename> typename Vector> using is_arithmetic_vector = ::hipgraph::backend::is_arithmetic_vector<T, Vector>;

    //template <template <typename> typename Vector, typename T> using is_arithmetic_vector = ::hipgraph::backend::is_arithmetic_vector<Vector, T>;
    template <typename... Ts>
    using is_arithmetic_vector = ::hipgraph::backend::is_arithmetic_vector<Ts...>;

    template <typename... Ts>
    using is_std_tuple_of_arithmetic_vectors
        = ::hipgraph::backend::is_std_tuple_of_arithmetic_vectors<Ts...>;

    template <typename... Ts>
    using is_arithmetic_or_thrust_tuple_of_arithmetic
        = ::hipgraph::backend::is_arithmetic_or_thrust_tuple_of_arithmetic<Ts...>;

    template <typename... Ts>
    using thrust_tuple_size_or_one = ::hipgraph::backend::thrust_tuple_size_or_one<Ts...>;

    template <typename TupleType>
    using compute_thrust_tuple_element_sizes
        = ::hipgraph::backend::compute_thrust_tuple_element_sizes<TupleType>;

    template <typename TupleType, size_t I>
    using thrust_tuple_get = ::hipgraph::backend::thrust_tuple_get<TupleType, I>;

} // namespace hipgraph

#endif // HIPGRAPH_HDR___UTILITIES_THRUST_TUPLE_UTILS_HPP_
