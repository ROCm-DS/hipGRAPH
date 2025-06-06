// Copyright (c) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <raft/core/detail/mdspan_util.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/coalesced_reduction.cuh>

namespace raft::detail
{

    /**
 * @brief Count the number of bits that are set to 1 in a vector.
 *
 * @tparam value_t the value type of the vector.
 * @tparam index_t the index type of vector and scalar.
 *
 * @param[in] res raft handle for managing expensive resources
 * @param[in] values Number of row in the matrix.
 * @param[in] max_len Maximum number of bits to count.
 * @param[out] counter Number of bits that are set to 1.
 */
    template <typename value_t, typename index_t>
    void popc(const raft::resources&               res,
              device_vector_view<value_t, index_t> values,
              index_t                              max_len,
              raft::device_scalar_view<index_t>    counter)
    {
        auto values_size   = values.size();
        auto values_matrix = raft::make_device_matrix_view<value_t, index_t, col_major>(
            values.data_handle(), values_size, 1);
        auto counter_vector
            = raft::make_device_vector_view<index_t, index_t>(counter.data_handle(), 1);

        static constexpr index_t len_per_item = sizeof(value_t) * 8;

        value_t tail_len = (max_len % len_per_item);
        value_t tail_mask
            = tail_len ? (value_t)((value_t{1} << tail_len) - value_t{1}) : ~value_t{0};
        raft::linalg::coalesced_reduction(
            res,
            values_matrix,
            counter_vector,
            index_t{0},
            false,
            [tail_mask, values_size] __device__(value_t value, index_t index) {
                index_t result = 0;
                if constexpr(len_per_item == 64)
                {
                    if(index == values_size - 1)
                        result = index_t(raft::detail::popc(value & tail_mask));
                    else
                        result = index_t(raft::detail::popc(value));
                }
                else
                { // Needed because popc is not overloaded for 16 and 8 bit elements
                    if(index == values_size - 1)
                        result = index_t(raft::detail::popc(uint32_t{value} & tail_mask));
                    else
                        result = index_t(raft::detail::popc(uint32_t{value}));
                }

                return result;
            });
    }

} // end namespace raft::detail
