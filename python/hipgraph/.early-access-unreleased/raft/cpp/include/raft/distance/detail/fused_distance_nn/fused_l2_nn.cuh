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
#include <raft/distance/detail/fused_distance_nn/helper_structs.cuh>
#include <raft/distance/detail/fused_distance_nn/simt_kernel.cuh>
#include <raft/distance/detail/pairwise_distance_base.cuh> // PairwiseDistances
#include <raft/linalg/contractions.cuh> // Policy
#include <raft/util/arch.cuh> // raft::util::arch::SM_*
#include <raft/util/cuda_utils.cuh> // raft::ceildiv, raft::shfl

#ifdef __HIP_PLATFORM_AMD__
// TODO(HIP/AMD): Add distance based dependency
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

            template <typename DataT,
                      typename OutT,
                      typename IdxT,
                      typename Policy,
                      typename ReduceOpT,
                      typename KVPReduceOpT>
            void fusedL2NNImpl(OutT*        min,
                               const DataT* x,
                               const DataT* y,
                               const DataT* xn,
                               const DataT* yn,
                               IdxT         m,
                               IdxT         n,
                               IdxT         k,
                               int*         workspace,
                               ReduceOpT    redOp,
                               KVPReduceOpT pairRedOp,
                               bool         sqrt,
                               bool         initOutBuffer,
                               cudaStream_t stream)
            {
                // The kernel policy is determined by fusedL2NN.
                typedef Policy P;

                dim3                              blk(P::Nthreads);
                auto                              nblks  = raft::ceildiv<int>(m, P::Nthreads);
                constexpr auto                    maxVal = std::numeric_limits<DataT>::max();
                typedef KeyValuePair<IdxT, DataT> KVPair;

                RAFT_CUDA_TRY(cudaMemsetAsync(workspace, 0, sizeof(int) * m, stream));
                if(initOutBuffer)
                {
                    initKernel<DataT, OutT, IdxT, ReduceOpT>
                        <<<nblks, P::Nthreads, 0, stream>>>(min, m, maxVal, redOp);
                    RAFT_CUDA_TRY(cudaGetLastError());
                }

                namespace arch = raft::util::arch;
                using AccT     = DataT;
                ops::l2_exp_distance_op<DataT, AccT, IdxT> distance_op{sqrt};

                raft::identity_op fin_op{};

                auto kernel = fusedDistanceNNkernel<DataT,
                                                    OutT,
                                                    IdxT,
                                                    P,
                                                    ReduceOpT,
                                                    KVPReduceOpT,
                                                    decltype(distance_op),
                                                    decltype(fin_op)>;

#ifdef __HIP_PLATFORM_AMD__
                // NOTE(HIP/AMD): Invoking non cutlass based kernel
                constexpr size_t shmemSize = P::SmemSize + ((P::Mblk + P::Nblk) * sizeof(DataT));
                dim3             grid      = launchConfigGenerator<P>(m, n, shmemSize, kernel);

                kernel<<<grid, blk, shmemSize, stream>>>(min,
                                                         x,
                                                         y,
                                                         xn,
                                                         yn,
                                                         m,
                                                         n,
                                                         k,
                                                         maxVal,
                                                         workspace,
                                                         redOp,
                                                         pairRedOp,
                                                         distance_op,
                                                         fin_op);
                RAFT_CUDA_TRY(cudaGetLastError());
#else
                // Get pointer to fp32 SIMT kernel to determine the best compute architecture
                // out of all for which the kernel was compiled for that matches closely
                // to the current device. Other methods to determine the architecture (that do not
                // require a pointer) can be error prone. See:
                // https://github.com/NVIDIA/cub/issues/545
                void* kernel_ptr    = reinterpret_cast<void*>(kernel);
                auto  runtime_arch  = arch::kernel_virtual_arch(kernel_ptr);
                auto  cutlass_range = arch::SM_range(arch::SM_80(), arch::SM_future());

                if(cutlass_range.contains(runtime_arch))
                {
                    // If device is SM_80 or later, use CUTLASS-based kernel.
                    using L2Op = raft::distance::detail::ops::l2_exp_cutlass_op<DataT, DataT>;
                    using kvp_cg_min_reduce_op_ = kvp_cg_min_reduce_op<DataT, IdxT, OutT>;
                    kvp_cg_min_reduce_op_ cg_reduce_op;
                    L2Op                  L2_dist_op(sqrt);

                    IdxT lda, ldb, ldd;
                    lda = k, ldb = k, ldd = n;

                    cutlassFusedDistanceNN<DataT,
                                           DataT,
                                           OutT,
                                           IdxT,
                                           P::Veclen,
                                           kvp_cg_min_reduce_op_,
                                           L2Op,
                                           ReduceOpT,
                                           KVPReduceOpT>(x,
                                                         y,
                                                         xn,
                                                         yn,
                                                         m,
                                                         n,
                                                         k,
                                                         lda,
                                                         ldb,
                                                         ldd,
                                                         min,
                                                         workspace,
                                                         cg_reduce_op,
                                                         L2_dist_op,
                                                         redOp,
                                                         pairRedOp,
                                                         stream);
                }
                else
                {
                    // If device less than SM_80, use fp32 SIMT kernel.
                    constexpr size_t shmemSize
                        = P::SmemSize + ((P::Mblk + P::Nblk) * sizeof(DataT));
                    dim3 grid = launchConfigGenerator<P>(m, n, shmemSize, kernel);

                    kernel<<<grid, blk, shmemSize, stream>>>(min,
                                                             x,
                                                             y,
                                                             xn,
                                                             yn,
                                                             m,
                                                             n,
                                                             k,
                                                             maxVal,
                                                             workspace,
                                                             redOp,
                                                             pairRedOp,
                                                             distance_op,
                                                             fin_op);
                    RAFT_CUDA_TRY(cudaGetLastError());
                }
#endif
            }

        } // namespace detail
    } // namespace distance
} // namespace raft
