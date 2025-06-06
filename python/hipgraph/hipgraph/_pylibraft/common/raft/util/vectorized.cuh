// Copyright (c) 2018-2024, NVIDIA CORPORATION.
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

#include <raft/util/cuda_utils.cuh>

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_fp16.h>
#else
#include <cuda_fp16.h>
#endif

namespace raft
{

    template <typename math_, int VecLen>
    struct IOType
    {
    };
    template <>
    struct IOType<bool, 1>
    {
        static_assert(sizeof(bool) == sizeof(int8_t), "IOType bool size assumption failed");
        typedef int8_t Type;
    };
    template <>
    struct IOType<bool, 2>
    {
        typedef int16_t Type;
    };
    template <>
    struct IOType<bool, 4>
    {
        typedef int32_t Type;
    };
    template <>
    struct IOType<bool, 8>
    {
        typedef int2 Type;
    };
    template <>
    struct IOType<bool, 16>
    {
        typedef int4 Type;
    };
    template <>
    struct IOType<int8_t, 1>
    {
        typedef int8_t Type;
    };
    template <>
    struct IOType<int8_t, 2>
    {
        typedef int16_t Type;
    };
    template <>
    struct IOType<int8_t, 4>
    {
        typedef int32_t Type;
    };
    template <>
    struct IOType<int8_t, 8>
    {
        typedef int2 Type;
    };
    template <>
    struct IOType<int8_t, 16>
    {
        typedef int4 Type;
    };
    template <>
    struct IOType<uint8_t, 1>
    {
        typedef uint8_t Type;
    };
    template <>
    struct IOType<uint8_t, 2>
    {
        typedef uint16_t Type;
    };
    template <>
    struct IOType<uint8_t, 4>
    {
        typedef uint32_t Type;
    };
    template <>
    struct IOType<uint8_t, 8>
    {
        typedef uint2 Type;
    };
    template <>
    struct IOType<uint8_t, 16>
    {
        typedef uint4 Type;
    };
    template <>
    struct IOType<int16_t, 1>
    {
        typedef int16_t Type;
    };
    template <>
    struct IOType<int16_t, 2>
    {
        typedef int32_t Type;
    };
    template <>
    struct IOType<int16_t, 4>
    {
        typedef int2 Type;
    };
    template <>
    struct IOType<int16_t, 8>
    {
        typedef int4 Type;
    };
    template <>
    struct IOType<uint16_t, 1>
    {
        typedef uint16_t Type;
    };
    template <>
    struct IOType<uint16_t, 2>
    {
        typedef uint32_t Type;
    };
    template <>
    struct IOType<uint16_t, 4>
    {
        typedef uint2 Type;
    };
    template <>
    struct IOType<uint16_t, 8>
    {
        typedef uint4 Type;
    };
    template <>
    struct IOType<__half, 1>
    {
        typedef __half Type;
    };
    template <>
    struct IOType<__half, 2>
    {
        typedef __half2 Type;
    };
    template <>
    struct IOType<__half, 4>
    {
        typedef uint2 Type;
    };
    template <>
    struct IOType<__half, 8>
    {
        typedef uint4 Type;
    };
    template <>
    struct IOType<__half2, 1>
    {
        typedef __half2 Type;
    };
    template <>
    struct IOType<__half2, 2>
    {
        typedef uint2 Type;
    };
    template <>
    struct IOType<__half2, 4>
    {
        typedef uint4 Type;
    };
    template <>
    struct IOType<int32_t, 1>
    {
        typedef int32_t Type;
    };
    template <>
    struct IOType<int32_t, 2>
    {
        typedef uint2 Type;
    };
    template <>
    struct IOType<int32_t, 4>
    {
        typedef uint4 Type;
    };
    template <>
    struct IOType<uint32_t, 1>
    {
        typedef uint32_t Type;
    };
    template <>
    struct IOType<uint32_t, 2>
    {
        typedef uint2 Type;
    };
    template <>
    struct IOType<uint32_t, 4>
    {
        typedef uint4 Type;
    };
    template <>
    struct IOType<float, 1>
    {
        typedef float Type;
    };
    template <>
    struct IOType<float, 2>
    {
        typedef float2 Type;
    };
    template <>
    struct IOType<float, 4>
    {
        typedef float4 Type;
    };
    template <>
    struct IOType<int64_t, 1>
    {
        typedef int64_t Type;
    };
    template <>
    struct IOType<int64_t, 2>
    {
        typedef uint4 Type;
    };
    template <>
    struct IOType<uint64_t, 1>
    {
        typedef uint64_t Type;
    };
    template <>
    struct IOType<uint64_t, 2>
    {
        typedef uint4 Type;
    };
    template <>
    struct IOType<unsigned long long, 1>
    {
        typedef unsigned long long Type;
    };
    template <>
    struct IOType<unsigned long long, 2>
    {
        typedef uint4 Type;
    };
    template <>
    struct IOType<double, 1>
    {
        typedef double Type;
    };
    template <>
    struct IOType<double, 2>
    {
        typedef double2 Type;
    };

    /**
 * @struct TxN_t
 *
 * @brief Internal data structure that is used to define a facade for vectorized
 * loads/stores across the most common POD types. The goal of his file is to
 * provide with CUDA programmers, an easy way to have compiler issue vectorized
 * load or store instructions to memory (either global or shared). Vectorized
 * accesses to memory are important as they'll utilize its resources
 * efficiently,
 * when compared to their non-vectorized counterparts. Obviously, for whatever
 * reasons if one is unable to issue such vectorized operations, one can always
 * fallback to using POD types.
 *
 * Concept of vectorized accesses : Threads process multiple elements
 * to speed up processing. These are loaded in a single read thanks
 * to type promotion. It is then reinterpreted as a vector elements
 * to perform the kernel's work.
 *
 * Caution : vectorized accesses requires input addresses to be memory aligned
 * according not to the input type but to the promoted type used for reading.
 *
 * Example demonstrating the use of load operations, performing math on such
 * loaded data and finally storing it back.
 * @code{.cu}
 * TxN_t<uint8_t,8> mydata1, mydata2;
 * int idx = (threadIdx.x + (blockIdx.x * blockDim.x)) * mydata1.Ratio;
 * mydata1.load(ptr1, idx);
 * mydata2.load(ptr2, idx);
 * #pragma unroll
 * for(int i=0;i<mydata1.Ratio;++i) {
 *     mydata1.val.data[i] += mydata2.val.data[i];
 * }
 * mydata1.store(ptr1, idx);
 * @endcode
 *
 * By doing as above, the interesting thing is that the code effectively remains
 * almost the same, in case one wants to upgrade to TxN_t<uint16_t,16> type.
 * Only change required is to replace variable declaration appropriately.
 *
 * Obviously, it's caller's responsibility to take care of pointer alignment!
 *
 * @tparam math_ the data-type in which the compute/math needs to happen
 * @tparam veclen_ the number of 'math_' types to be loaded/stored per
 * instruction
 */
    template <typename math_, int veclen_>
    struct TxN_t
    {
        /** underlying math data type */
        typedef math_ math_t;
        /** internal storage data type */
        typedef typename IOType<math_t, veclen_>::Type io_t;

        /** defines the number of 'math_t' types stored by this struct */
        static const int Ratio = veclen_;

        struct alignas(io_t)
        {
            /** the vectorized data that is used for subsequent operations */
            math_t data[Ratio];
        } val;

        __device__ auto* vectorized_data()
        {
            return reinterpret_cast<io_t*>(val.data);
        }

        ///@todo: add default constructor

        /**
   * @brief Fill the contents of this structure with a constant value
   * @param _val the constant to be filled
   */
        DI void fill(math_t _val)
        {
#pragma unroll
            for(int i = 0; i < Ratio; ++i)
            {
                val.data[i] = _val;
            }
        }

        ///@todo: how to handle out-of-bounds!!?

        /**
   * @defgroup LoadsStores Global/Shared vectored loads or stores
   *
   * @brief Perform vectored loads/stores on this structure
   * @tparam idx_t index data type
   * @param ptr base pointer from where to load (or store) the data. It must
   *  be aligned to 'sizeof(io_t)'!
   * @param idx the offset from the base pointer which will be loaded
   *  (or stored) by the current thread. This must be aligned to 'Ratio'!
   *
   * @note: In case of loads, after a successful execution, the val.data will
   *  be populated with the desired data loaded from the pointer location. In
   * case of stores, the data in the val.data will be stored to that location.
   * @{
   */
        template <typename idx_t = int>
        DI void load(const math_t* ptr, idx_t idx)
        {
            const io_t* bptr   = reinterpret_cast<const io_t*>(&ptr[idx]);
            *vectorized_data() = __ldg(bptr);
        }

        template <typename idx_t = int>
        DI void load(math_t* ptr, idx_t idx)
        {
            io_t* bptr         = reinterpret_cast<io_t*>(&ptr[idx]);
            *vectorized_data() = *bptr;
        }

        template <typename idx_t = int>
        DI void store(math_t* ptr, idx_t idx)
        {
            io_t* bptr = reinterpret_cast<io_t*>(&ptr[idx]);
            *bptr      = *vectorized_data();
        }
        /** @} */
    };

    /** this is just to keep the compiler happy! */
    template <typename math_>
    struct TxN_t<math_, 0>
    {
        typedef math_    math_t;
        static const int Ratio = 1;

        struct
        {
            math_t data[1];
        } val;

        DI void fill(math_t _val) {}
        template <typename idx_t = int>
        DI void load(const math_t* ptr, idx_t idx)
        {
        }
        template <typename idx_t = int>
        DI void load(math_t* ptr, idx_t idx)
        {
        }
        template <typename idx_t = int>
        DI void store(math_t* ptr, idx_t idx)
        {
        }
    };

} // namespace raft
