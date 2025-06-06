// Copyright (c) 2019-2022, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <raft/util/cudart_utils.hpp>

namespace raft::stats
{

    /**
 * @ingroup stats_histogram
 * @{
 */

    /**
 * @brief Types of support histogram implementations
 */
    enum HistType
    {
        /** shared mem atomics but with bins to be 1b int's */
        HistTypeSmemBits1 = 1,
        /** shared mem atomics but with bins to be 2b int's */
        HistTypeSmemBits2 = 2,
        /** shared mem atomics but with bins to be 4b int's */
        HistTypeSmemBits4 = 4,
        /** shared mem atomics but with bins to ba 1B int's */
        HistTypeSmemBits8 = 8,
        /** shared mem atomics but with bins to be 2B int's */
        HistTypeSmemBits16 = 16,
        /** use only global atomics */
        HistTypeGmem,
        /** uses shared mem atomics to reduce global traffic */
        HistTypeSmem,
        /**
   * uses shared mem atomics with match_any intrinsic to further reduce shared
   * memory traffic. This can only be enabled on Volta and later architectures.
   * If one tries to enable this for older arch's, it will fall back to
   * `HistTypeSmem`.
   * @note This is to be used only when the input dataset leads to a lot of
   *       repetitions in a given warp, else, this algo can be much slower than
   *       `HistTypeSmem`!
   */
        HistTypeSmemMatchAny,
        /** builds a hashmap of active bins in shared mem */
        HistTypeSmemHash,
        /** decide at runtime the best algo for the given inputs */
        HistTypeAuto
    };

    /** @} */

    /**
 * @ingroup stats_information_criterion
 * @{
 */

    /**
 * @brief Supported types of information criteria
 */
    enum IC_Type
    {
        AIC,
        AICc,
        BIC
    };

    /** @} */

}; // end namespace raft::stats
