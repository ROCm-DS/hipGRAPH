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

#include <raft/core/operators.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/util/cuda_utils.cuh>

#ifdef __HIP_PLATFORM_AMD__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#else
#include <cub/cub.cuh>
#endif

#include <type_traits>

namespace raft
{
    namespace linalg
    {
        namespace detail
        {

            // Kernel to perform reductions along the strided dimension
            // of the matrix, i.e. reduce along columns for row major or reduce along rows
            // for column major layout
            template <typename Type, typename MainLambda>
            RAFT_KERNEL stridedSummationKernel(
                Type* dots, const Type* data, int D, int N, Type init, MainLambda main_op)
            {
                // Thread reduction
                Type thread_data = Type(init);
                int  colStart    = blockIdx.x * blockDim.x + threadIdx.x;
                if(colStart < D)
                {
                    int rowStart = blockIdx.y * blockDim.y + threadIdx.y;
                    int stride   = blockDim.y * gridDim.y;
                    for(int j = rowStart; j < N; j += stride)
                    {
                        int idx = colStart + j * D;
                        thread_data += main_op(data[idx], j);
                    }
                }

                // Block reduction
                extern __shared__ char tmp[]; // One element per thread in block
                Type*                  temp  = (Type*)tmp; // Cast to desired type
                int                    myidx = threadIdx.x + blockDim.x * threadIdx.y;
                temp[myidx]                  = thread_data;
                __syncthreads();
                for(int j = blockDim.y / 2; j > 0; j /= 2)
                {
                    if(threadIdx.y < j)
                        temp[myidx] += temp[myidx + j * blockDim.x];
                    __syncthreads();
                }

                // Grid reduction
                if((colStart < D) && (threadIdx.y == 0))
                    raft::myAtomicAdd(dots + colStart, temp[myidx]);
            }

            // Kernel to perform reductions along the strided dimension
            // of the matrix, i.e. reduce along columns for row major or reduce along rows
            // for column major layout
            template <typename InType,
                      typename OutType,
                      typename IdxType,
                      typename MainLambda,
                      typename ReduceLambda>
            RAFT_KERNEL stridedReductionKernel(OutType*      dots,
                                               const InType* data,
                                               int           D,
                                               int           N,
                                               OutType       init,
                                               MainLambda    main_op,
                                               ReduceLambda  reduce_op)
            {
                // Thread reduction
                OutType thread_data = init;
                IdxType colStart    = blockIdx.x * blockDim.x + threadIdx.x;
                if(colStart < D)
                {
                    IdxType rowStart = blockIdx.y * blockDim.y + threadIdx.y;
                    IdxType stride   = blockDim.y * gridDim.y;
                    for(IdxType j = rowStart; j < N; j += stride)
                    {
                        IdxType idx = colStart + j * D;
                        thread_data = reduce_op(thread_data, main_op(data[idx], j));
                    }
                }

                // Block reduction
                extern __shared__ char tmp[]; // One element per thread in block
                auto*                  temp = (OutType*)tmp; // Cast to desired type
                IdxType myidx = threadIdx.x + ((IdxType)blockDim.x * (IdxType)threadIdx.y);
                temp[myidx]   = thread_data;
                __syncthreads();
                for(int j = blockDim.y / 2; j > 0; j /= 2)
                {
                    if(threadIdx.y < j)
                        temp[myidx] = reduce_op(temp[myidx], temp[myidx + j * blockDim.x]);
                    __syncthreads();
                }

                // Grid reduction
                if((colStart < D) && (threadIdx.y == 0))
                    raft::myAtomicReduce(dots + colStart, temp[myidx], reduce_op);
            }

            template <typename InType,
                      typename OutType      = InType,
                      typename IdxType      = int,
                      typename MainLambda   = raft::identity_op,
                      typename ReduceLambda = raft::add_op,
                      typename FinalLambda  = raft::identity_op>
            void stridedReduction(OutType*      dots,
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
                ///@todo: this extra should go away once we have eliminated the need
                /// for atomics in stridedKernel (redesign for this is already underway)
                if(!inplace)
                    raft::linalg::unaryOp(dots, dots, D, raft::const_op(init), stream);

                // Arbitrary numbers for now, probably need to tune
                const dim3 thrds(32, 16);
                IdxType    elemsPerThread = raft::ceildiv(N, (IdxType)thrds.y);
                elemsPerThread            = (elemsPerThread > 8) ? 8 : elemsPerThread;
                const dim3   nblks(raft::ceildiv(D, (IdxType)thrds.x),
                                 raft::ceildiv(N, (IdxType)thrds.y * elemsPerThread));
                const size_t shmemSize = sizeof(OutType) * thrds.x * thrds.y;

                ///@todo: this complication should go away once we have eliminated the need
                /// for atomics in stridedKernel (redesign for this is already underway)
                if constexpr(std::is_same<ReduceLambda, raft::add_op>::value
                             && std::is_same<InType, OutType>::value)
                    stridedSummationKernel<InType>
                        <<<nblks, thrds, shmemSize, stream>>>(dots, data, D, N, init, main_op);
                else
                    stridedReductionKernel<InType, OutType, IdxType>
                        <<<nblks, thrds, shmemSize, stream>>>(
                            dots, data, D, N, init, main_op, reduce_op);

                ///@todo: this complication should go away once we have eliminated the need
                /// for atomics in stridedKernel (redesign for this is already underway)
                // Perform final op on output data
                if(!std::is_same<FinalLambda, raft::identity_op>::value)
                    raft::linalg::unaryOp(dots, dots, D, final_op, stream);
            }

        }; // end namespace detail
    }; // end namespace linalg
}; // end namespace raft
