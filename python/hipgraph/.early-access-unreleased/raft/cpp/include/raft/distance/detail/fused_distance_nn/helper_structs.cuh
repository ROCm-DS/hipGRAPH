// Copyright (c) 2021-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

/*
 * Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#pragma once

#include <raft/core/kvp.hpp> // raft::KeyValuePair
#include <raft/core/operators.hpp> // raft::identity_op
#include <raft/distance/detail/distance_ops/l2_exp.cuh> // ops::l2_exp_distance_op
#include <raft/distance/detail/fused_distance_nn/simt_kernel.cuh>
#include <raft/distance/detail/pairwise_distance_base.cuh> // PairwiseDistances
#include <raft/linalg/contractions.cuh> // Policy
#include <raft/util/arch.cuh> // raft::util::arch::SM_*
#include <raft/util/cuda_utils.cuh> // raft::ceildiv, raft::shfl
#include <raft/util/device_atomics.cuh>

#ifdef __HIP_PLATFORM_AMD__
// TODO(HIP/AMD): Need to add support for fused distance NN
#else
#include <raft/distance/detail/fused_distance_nn/cutlass_base.cuh>
#endif

#include <cstddef> // size_t
#include <limits> // std::numeric_limits

namespace raft
{
    namespace distance
    {

        namespace detail
        {

            template <typename LabelT, typename DataT>
            struct KVPMinReduceImpl
            {
                typedef raft::KeyValuePair<LabelT, DataT> KVP;
                DI KVP operator()(LabelT rit, const KVP& a, const KVP& b)
                {
                    return b.value < a.value ? b : a;
                }
                DI KVP operator()(const KVP& a, const KVP& b)
                {
                    return b.value < a.value ? b : a;
                }

            }; // KVPMinReduce

            template <typename LabelT, typename DataT>
            struct MinAndDistanceReduceOpImpl
            {
                typedef typename raft::KeyValuePair<LabelT, DataT> KVP;

                DI void operator()(LabelT rid, KVP* out, const KVP& other) const
                {
                    if(other.value < out->value)
                    {
                        out->key   = other.key;
                        out->value = other.value;
                    }
                }
                DI void operator()(LabelT rid, volatile KVP* out, const KVP& other) const
                {
                    if(other.value < out->value)
                    {
                        out->key   = other.key;
                        out->value = other.value;
                    }
                }

                DI void operator()(LabelT rid, DataT* out, const KVP& other) const
                {
                    if(other.value < *out)
                    {
                        *out = other.value;
                    }
                }

                DI void operator()(LabelT rid, volatile DataT* out, const KVP& other) const
                {
                    if(other.value < *out)
                    {
                        *out = other.value;
                    }
                }

                DI void operator()(LabelT rid, DataT* out, const DataT& other) const
                {
                    if(other < *out)
                    {
                        *out = other;
                    }
                }

                DI void operator()(LabelT rid, volatile DataT* out, const DataT& other) const
                {
                    if(other < *out)
                    {
                        *out = other;
                    }
                }

                DI void init(DataT* out, DataT maxVal) const
                {
                    *out = maxVal;
                }
                DI void init(KVP* out, DataT maxVal) const
                {
                    out->value = maxVal;
                    out->key   = 0xfffffff0;
                }

                DI void init_key(DataT& out, LabelT idx) const
                {
                    return;
                }
                DI void init_key(KVP& out, LabelT idx) const
                {
                    out.key = idx;
                }

                DI DataT get_value(KVP& out) const
                {
                    return out.value;
                }
                DI DataT get_value(DataT& out) const
                {
                    return out;
                }
            };

            template <typename LabelT, typename DataT>
            struct MinReduceOpImpl
            {
                typedef typename raft::KeyValuePair<LabelT, DataT> KVP;
                DI void operator()(LabelT rid, DataT* out, const KVP& other)
                {
                    if(other.value < *out)
                    {
                        *out = other.value;
                    }
                }

                DI void init(DataT* out, DataT maxVal)
                {
                    *out = maxVal;
                }
            };

            template <typename DataT, typename OutT, typename IdxT, typename ReduceOpT>
            RAFT_KERNEL initKernel(OutT* min, IdxT m, DataT maxVal, ReduceOpT redOp)
            {
                auto tid = IdxT(blockIdx.x) * blockDim.x + threadIdx.x;
                if(tid < m)
                {
                    redOp.init(min + tid, maxVal);
                }
            }

            template <typename DataT, typename OutT, typename IdxT, typename ReduceOpT>
            void initialize(OutT* min, IdxT m, DataT maxVal, ReduceOpT redOp, cudaStream_t stream)
            {
                auto blks = raft::ceildiv(m, 256);
                initKernel<DataT, OutT, IdxT><<<blks, 256, 0, stream>>>(min, m, maxVal, redOp);
            }

            // cg::reduce functor for FusedDistanceNN used in its cutlass version
            // to output the min distance value & key(loc id).
            // This is used in fused_distance_nn/predicated_tile_iterator_reduced_vec.h
            // store_with_byte_offset() passed to cg::reduce() & select_reduce.
            template <typename AccType, typename Index, typename OutType>
            struct kvp_cg_min_reduce_op
            {
                typedef typename raft::KeyValuePair<Index, AccType> KVP;

                __host__ __device__ kvp_cg_min_reduce_op() noexcept {};

                using AccTypeT = AccType;
                using IndexT   = Index;
                // functor signature.
                __host__ __device__ KVP operator()(KVP a, KVP b) const
                {
                    return a.value < b.value ? a : b;
                }

                __host__ __device__ AccType operator()(AccType a, AccType b) const
                {
                    return min(a, b);
                }

                __host__ __device__ bool isAmin(AccType a, AccType b) const
                {
                    return a < b ? true : false;
                }
            };

        } // namespace detail
    } // namespace distance
} // namespace raft
