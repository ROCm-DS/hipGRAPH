// Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <raft/core/nvtx.hpp>
#include <raft/core/operators.hpp>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_uvector.hpp>

#ifdef __HIP_PLATFORM_AMD__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#else
#include <cub/cub.cuh>
#endif

namespace raft
{
    namespace linalg
    {
        namespace detail
        {

            template <int warpSize, int tpb, int rpw, bool noLoop = false>
            struct ReductionThinPolicy
            {
                static_assert(tpb % warpSize == 0);

                static constexpr int LogicalWarpSize    = warpSize;
                static constexpr int ThreadsPerBlock    = tpb;
                static constexpr int RowsPerLogicalWarp = rpw;
                static constexpr int NumLogicalWarps    = ThreadsPerBlock / LogicalWarpSize;
                static constexpr int RowsPerBlock       = NumLogicalWarps * RowsPerLogicalWarp;

                // Whether D (run-time arg) will be smaller than warpSize (compile-time parameter)
                static constexpr bool NoSequentialReduce = noLoop;
            };

            template <typename Policy,
                      typename InType,
                      typename OutType,
                      typename IdxType,
                      typename MainLambda,
                      typename ReduceLambda,
                      typename FinalLambda>
            RAFT_KERNEL __launch_bounds__(Policy::ThreadsPerBlock)
                coalescedReductionThinKernel(OutType*      dots,
                                             const InType* data,
                                             IdxType       D,
                                             IdxType       N,
                                             OutType       init,
                                             MainLambda    main_op,
                                             ReduceLambda  reduce_op,
                                             FinalLambda   final_op,
                                             bool          inplace = false)
            {
                /* The strategy to achieve near-SOL memory bandwidth differs based on D:
   *  - For small D, we need to process multiple rows per logical warp in order to have
   *    multiple loads per thread and increase bytes in flight and amortize latencies.
   *  - For large D, we start with a sequential reduction. The compiler partially unrolls
   *    that loop (e.g. first a loop of stride 16, then 8, 4, and 1).
   */
                IdxType i0
                    = threadIdx.y + (Policy::RowsPerBlock * static_cast<IdxType>(blockIdx.x));
                if(i0 >= N)
                    return;

                OutType acc[Policy::RowsPerLogicalWarp];
#pragma unroll
                for(int k = 0; k < Policy::RowsPerLogicalWarp; k++)
                {
                    acc[k] = init;
                }

                if constexpr(Policy::NoSequentialReduce)
                {
                    IdxType j = threadIdx.x;
                    if(j < D)
                    {
#pragma unroll
                        for(IdxType k = 0; k < Policy::RowsPerLogicalWarp; k++)
                        {
                            // Only the first row is known to be within bounds. Clamp to avoid out-of-mem read.
                            const IdxType i = raft::min(i0 + k * Policy::NumLogicalWarps, N - 1);
                            acc[k]          = reduce_op(acc[k], main_op(data[j + (D * i)], j));
                        }
                    }
                }
                else
                {
                    for(IdxType j = threadIdx.x; j < D; j += Policy::LogicalWarpSize)
                    {
#pragma unroll
                        for(IdxType k = 0; k < Policy::RowsPerLogicalWarp; k++)
                        {
                            const IdxType i = raft::min(i0 + k * Policy::NumLogicalWarps, N - 1);
                            acc[k]          = reduce_op(acc[k], main_op(data[j + (D * i)], j));
                        }
                    }
                }

                /* This vector reduction has two benefits compared to naive separate reductions:
   * - It avoids the LSU bottleneck when the number of columns is around 32 (e.g. for 32, 5 shuffles
   *   are required and there is no initial sequential reduction to amortize that cost).
   * - It distributes the outputs to multiple threads, enabling a coalesced store when the number of
   *   rows per logical warp and logical warp size are equal.
   */
                raft::logicalWarpReduceVector<Policy::LogicalWarpSize, Policy::RowsPerLogicalWarp>(
                    acc, threadIdx.x, reduce_op);

                constexpr int reducOutVecWidth
                    = std::max(1, Policy::RowsPerLogicalWarp / Policy::LogicalWarpSize);
                constexpr int reducOutGroupSize
                    = std::max(1, Policy::LogicalWarpSize / Policy::RowsPerLogicalWarp);
                constexpr int reducNumGroups = Policy::LogicalWarpSize / reducOutGroupSize;

                if(threadIdx.x % reducOutGroupSize == 0)
                {
                    const int groupId = threadIdx.x / reducOutGroupSize;
                    if(inplace)
                    {
#pragma unroll
                        for(int k = 0; k < reducOutVecWidth; k++)
                        {
                            const int     reductionId = k * reducNumGroups + groupId;
                            const IdxType i           = i0 + reductionId * Policy::NumLogicalWarps;
                            if(i < N)
                            {
                                dots[i] = final_op(reduce_op(dots[i], acc[k]));
                            }
                        }
                    }
                    else
                    {
#pragma unroll
                        for(int k = 0; k < reducOutVecWidth; k++)
                        {
                            const int     reductionId = k * reducNumGroups + groupId;
                            const IdxType i           = i0 + reductionId * Policy::NumLogicalWarps;
                            if(i < N)
                            {
                                dots[i] = final_op(acc[k]);
                            }
                        }
                    }
                }
            }

            template <typename Policy,
                      typename InType,
                      typename OutType      = InType,
                      typename IdxType      = int,
                      typename MainLambda   = raft::identity_op,
                      typename ReduceLambda = raft::add_op,
                      typename FinalLambda  = raft::identity_op>
            void coalescedReductionThin(OutType*      dots,
                                        const InType* data,
                                        IdxType       D,
                                        IdxType       N,
                                        OutType       init,
                                        cudaStream_t  stream,
                                        bool          inplace   = false,
                                        MainLambda    main_op   = raft::identity_op(),
                                        ReduceLambda  reduce_op = raft::add_op(),
                                        FinalLambda   final_op  = raft::identity_op())
            {
                common::nvtx::range<common::nvtx::domain::raft> fun_scope(
                    "coalescedReductionThin<%d,%d,%d,%d>",
                    Policy::LogicalWarpSize,
                    Policy::ThreadsPerBlock,
                    Policy::RowsPerLogicalWarp,
                    static_cast<int>(Policy::NoSequentialReduce));
                dim3 threads(Policy::LogicalWarpSize, Policy::NumLogicalWarps, 1);
                dim3 blocks(ceildiv<IdxType>(N, Policy::RowsPerBlock), 1, 1);
                coalescedReductionThinKernel<Policy><<<blocks, threads, 0, stream>>>(
                    dots, data, D, N, init, main_op, reduce_op, final_op, inplace);
                RAFT_CUDA_TRY(cudaPeekAtLastError());
            }

            template <typename InType,
                      typename OutType      = InType,
                      typename IdxType      = int,
                      typename MainLambda   = raft::identity_op,
                      typename ReduceLambda = raft::add_op,
                      typename FinalLambda  = raft::identity_op>
            void coalescedReductionThinDispatcher(OutType*      dots,
                                                  const InType* data,
                                                  IdxType       D,
                                                  IdxType       N,
                                                  OutType       init,
                                                  cudaStream_t  stream,
                                                  bool          inplace   = false,
                                                  MainLambda    main_op   = raft::identity_op(),
                                                  ReduceLambda  reduce_op = raft::add_op(),
                                                  FinalLambda   final_op  = raft::identity_op())
            {
                if(D <= IdxType(2))
                {
                    coalescedReductionThin<ReductionThinPolicy<2, 128, 8, true>>(
                        dots, data, D, N, init, stream, inplace, main_op, reduce_op, final_op);
                }
                else if(D <= IdxType(4))
                {
                    coalescedReductionThin<ReductionThinPolicy<4, 128, 8, true>>(
                        dots, data, D, N, init, stream, inplace, main_op, reduce_op, final_op);
                }
                else if(D <= IdxType(8))
                {
                    coalescedReductionThin<ReductionThinPolicy<8, 128, 8, true>>(
                        dots, data, D, N, init, stream, inplace, main_op, reduce_op, final_op);
                }
                else if(D <= IdxType(16))
                {
                    coalescedReductionThin<ReductionThinPolicy<16, 128, 8, true>>(
                        dots, data, D, N, init, stream, inplace, main_op, reduce_op, final_op);
                }
                else if(D <= IdxType(32))
                {
                    coalescedReductionThin<ReductionThinPolicy<32, 128, 8, true>>(
                        dots, data, D, N, init, stream, inplace, main_op, reduce_op, final_op);
                }
                else if(D < IdxType(128))
                {
                    coalescedReductionThin<ReductionThinPolicy<32, 128, 4, false>>(
                        dots, data, D, N, init, stream, inplace, main_op, reduce_op, final_op);
                }
                else
                {
                    // For D=128 (included) and above, the 4x-unrolled loading loop is used
                    // and multiple rows per warp are counter-productive in terms of cache-friendliness
                    // and register use.
                    coalescedReductionThin<ReductionThinPolicy<32, 128, 1, false>>(
                        dots, data, D, N, init, stream, inplace, main_op, reduce_op, final_op);
                }
            }

            template <int TPB,
                      typename InType,
                      typename OutType,
                      typename IdxType,
                      typename MainLambda,
                      typename ReduceLambda,
                      typename FinalLambda>
            RAFT_KERNEL __launch_bounds__(TPB)
                coalescedReductionMediumKernel(OutType*      dots,
                                               const InType* data,
                                               IdxType       D,
                                               IdxType       N,
                                               OutType       init,
                                               MainLambda    main_op,
                                               ReduceLambda  reduce_op,
                                               FinalLambda   final_op,
                                               bool          inplace = false)
            {
                typedef cub::BlockReduce<OutType, TPB, cub::BLOCK_REDUCE_RAKING> BlockReduce;
                __shared__ typename BlockReduce::TempStorage                     temp_storage;
                OutType                                                          thread_data = init;
                IdxType rowStart = blockIdx.x * D;
                for(IdxType i = threadIdx.x; i < D; i += TPB)
                {
                    IdxType idx = rowStart + i;
                    thread_data = reduce_op(thread_data, main_op(data[idx], i));
                }
                OutType acc = BlockReduce(temp_storage).Reduce(thread_data, reduce_op);
                if(threadIdx.x == 0)
                {
                    if(inplace)
                    {
                        dots[blockIdx.x] = final_op(reduce_op(dots[blockIdx.x], acc));
                    }
                    else
                    {
                        dots[blockIdx.x] = final_op(acc);
                    }
                }
            }

            template <int TPB,
                      typename InType,
                      typename OutType      = InType,
                      typename IdxType      = int,
                      typename MainLambda   = raft::identity_op,
                      typename ReduceLambda = raft::add_op,
                      typename FinalLambda  = raft::identity_op>
            void coalescedReductionMedium(OutType*      dots,
                                          const InType* data,
                                          IdxType       D,
                                          IdxType       N,
                                          OutType       init,
                                          cudaStream_t  stream,
                                          bool          inplace   = false,
                                          MainLambda    main_op   = raft::identity_op(),
                                          ReduceLambda  reduce_op = raft::add_op(),
                                          FinalLambda   final_op  = raft::identity_op())
            {
                common::nvtx::range<common::nvtx::domain::raft> fun_scope(
                    "coalescedReductionMedium<%d>", TPB);
                coalescedReductionMediumKernel<TPB><<<N, TPB, 0, stream>>>(
                    dots, data, D, N, init, main_op, reduce_op, final_op, inplace);
                RAFT_CUDA_TRY(cudaPeekAtLastError());
            }

            template <typename InType,
                      typename OutType      = InType,
                      typename IdxType      = int,
                      typename MainLambda   = raft::identity_op,
                      typename ReduceLambda = raft::add_op,
                      typename FinalLambda  = raft::identity_op>
            void coalescedReductionMediumDispatcher(OutType*      dots,
                                                    const InType* data,
                                                    IdxType       D,
                                                    IdxType       N,
                                                    OutType       init,
                                                    cudaStream_t  stream,
                                                    bool          inplace   = false,
                                                    MainLambda    main_op   = raft::identity_op(),
                                                    ReduceLambda  reduce_op = raft::add_op(),
                                                    FinalLambda   final_op  = raft::identity_op())
            {
                // Note: for now, this kernel is only used when D > 256. If this changes in the future, use
                // smaller block sizes when relevant.
                coalescedReductionMedium<256>(
                    dots, data, D, N, init, stream, inplace, main_op, reduce_op, final_op);
            }

            template <int tpb, int bpr>
            struct ReductionThickPolicy
            {
                static constexpr int ThreadsPerBlock = tpb;
                static constexpr int BlocksPerRow    = bpr;
                static constexpr int BlockStride     = tpb * bpr;
            };

            template <typename Policy,
                      typename InType,
                      typename OutType,
                      typename IdxType,
                      typename MainLambda,
                      typename ReduceLambda>
            RAFT_KERNEL __launch_bounds__(Policy::ThreadsPerBlock)
                coalescedReductionThickKernel(OutType*      buffer,
                                              const InType* data,
                                              IdxType       D,
                                              IdxType       N,
                                              OutType       init,
                                              MainLambda    main_op,
                                              ReduceLambda  reduce_op)
            {
                typedef cub::BlockReduce<OutType, Policy::ThreadsPerBlock, cub::BLOCK_REDUCE_RAKING>
                                                             BlockReduce;
                __shared__ typename BlockReduce::TempStorage temp_storage;
                OutType                                      thread_data = init;
                IdxType                                      rowStart    = blockIdx.x * D;
                for(IdxType i = blockIdx.y * Policy::ThreadsPerBlock + threadIdx.x; i < D;
                    i += Policy::BlockStride)
                {
                    IdxType idx = rowStart + i;
                    thread_data = reduce_op(thread_data, main_op(data[idx], i));
                }
                OutType acc = BlockReduce(temp_storage).Reduce(thread_data, reduce_op);
                if(threadIdx.x == 0)
                {
                    buffer[Policy::BlocksPerRow * blockIdx.x + blockIdx.y] = acc;
                }
            }

            template <typename ThickPolicy,
                      typename ThinPolicy,
                      typename InType,
                      typename OutType      = InType,
                      typename IdxType      = int,
                      typename MainLambda   = raft::identity_op,
                      typename ReduceLambda = raft::add_op,
                      typename FinalLambda  = raft::identity_op>
            void coalescedReductionThick(OutType*      dots,
                                         const InType* data,
                                         IdxType       D,
                                         IdxType       N,
                                         OutType       init,
                                         cudaStream_t  stream,
                                         bool          inplace   = false,
                                         MainLambda    main_op   = raft::identity_op(),
                                         ReduceLambda  reduce_op = raft::add_op(),
                                         FinalLambda   final_op  = raft::identity_op())
            {
                common::nvtx::range<common::nvtx::domain::raft> fun_scope(
                    "coalescedReductionThick<%d,%d>",
                    ThickPolicy::ThreadsPerBlock,
                    ThickPolicy::BlocksPerRow);

                dim3 threads(ThickPolicy::ThreadsPerBlock, 1, 1);
                dim3 blocks(N, ThickPolicy::BlocksPerRow, 1);

                rmm::device_uvector<OutType> buffer(N * ThickPolicy::BlocksPerRow, stream);

                /* We apply a two-step reduction:
   *  1. coalescedReductionThickKernel reduces the [N x D] input data to [N x BlocksPerRow]. It
   *     applies the main_op but not the final op.
   *  2. coalescedReductionThinKernel reduces [N x BlocksPerRow] to [N x 1]. It doesn't apply any
   *     main_op but applies final_op. If in-place, the existing and new values are reduced.
   */

                coalescedReductionThickKernel<ThickPolicy><<<blocks, threads, 0, stream>>>(
                    buffer.data(), data, D, N, init, main_op, reduce_op);
                RAFT_CUDA_TRY(cudaPeekAtLastError());

                coalescedReductionThin<ThinPolicy>(dots,
                                                   buffer.data(),
                                                   static_cast<IdxType>(ThickPolicy::BlocksPerRow),
                                                   N,
                                                   init,
                                                   stream,
                                                   inplace,
                                                   raft::identity_op(),
                                                   reduce_op,
                                                   final_op);
            }

            template <typename InType,
                      typename OutType      = InType,
                      typename IdxType      = int,
                      typename MainLambda   = raft::identity_op,
                      typename ReduceLambda = raft::add_op,
                      typename FinalLambda  = raft::identity_op>
            void coalescedReductionThickDispatcher(OutType*      dots,
                                                   const InType* data,
                                                   IdxType       D,
                                                   IdxType       N,
                                                   OutType       init,
                                                   cudaStream_t  stream,
                                                   bool          inplace   = false,
                                                   MainLambda    main_op   = raft::identity_op(),
                                                   ReduceLambda  reduce_op = raft::add_op(),
                                                   FinalLambda   final_op  = raft::identity_op())
            {
                // Note: multiple elements per thread to take advantage of the sequential reduction and loop
                // unrolling
                if(D < IdxType(32768))
                {
                    coalescedReductionThick<ReductionThickPolicy<256, 32>,
                                            ReductionThinPolicy<32, 128, 1>>(
                        dots, data, D, N, init, stream, inplace, main_op, reduce_op, final_op);
                }
                else
                {
                    coalescedReductionThick<ReductionThickPolicy<256, 64>,
                                            ReductionThinPolicy<32, 128, 1>>(
                        dots, data, D, N, init, stream, inplace, main_op, reduce_op, final_op);
                }
            }

            // Primitive to perform reductions along the coalesced dimension of the matrix, i.e. reduce along
            // rows for row major or reduce along columns for column major layout. Can do an inplace reduction
            // adding to original values of dots if requested.
            template <typename InType,
                      typename OutType      = InType,
                      typename IdxType      = int,
                      typename MainLambda   = raft::identity_op,
                      typename ReduceLambda = raft::add_op,
                      typename FinalLambda  = raft::identity_op>
            void coalescedReduction(OutType*      dots,
                                    const InType* data,
                                    IdxType       D,
                                    IdxType       N,
                                    OutType       init,
                                    cudaStream_t  stream,
                                    bool          inplace   = false,
                                    MainLambda    main_op   = raft::identity_op(),
                                    ReduceLambda  reduce_op = raft::add_op(),
                                    FinalLambda   final_op  = raft::identity_op())
            {
                /* The primitive selects one of three implementations based on heuristics:
   *  - Thin: very efficient when D is small and/or N is large
   *  - Thick: used when N is very small and D very large
   *  - Medium: used when N is too small to fill the GPU with the thin kernel
   */
                const IdxType numSMs = raft::getMultiProcessorCount();
                if(D <= IdxType(256) || N >= IdxType(4) * numSMs)
                {
                    coalescedReductionThinDispatcher(
                        dots, data, D, N, init, stream, inplace, main_op, reduce_op, final_op);
                }
                else if(N < numSMs && D >= IdxType(16384))
                {
                    coalescedReductionThickDispatcher(
                        dots, data, D, N, init, stream, inplace, main_op, reduce_op, final_op);
                }
                else
                {
                    coalescedReductionMediumDispatcher(
                        dots, data, D, N, init, stream, inplace, main_op, reduce_op, final_op);
                }
            }

        } // namespace detail
    } // namespace linalg
} // namespace raft
