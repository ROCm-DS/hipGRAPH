// Copyright (c) 2019-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <raft/util/cuda_utils.cuh>

#include <vector>

// Taken from:
//  https://github.com/teju85/programming/blob/master/euler/include/seive.h

namespace raft
{
    namespace common
    {

        /**
 * @brief Implementation of 'Seive of Eratosthenes'
 */
        class Seive
        {
        public:
            /**
   * @param _num number of integers for which seive is needed
   */
            Seive(unsigned _num)
            {
                N = _num;
                generateSeive();
            }

            /**
   * @brief Check whether a number is prime or not
   * @param num number to be checked
   * @return true if the 'num' is prime, else false
   */
            bool isPrime(unsigned num) const
            {
                unsigned mask, pos;
                if(num <= 1)
                {
                    return false;
                }
                if(num == 2)
                {
                    return true;
                }
                if(!(num & 1))
                {
                    return false;
                }
                getMaskPos(num, mask, pos);
                return (seive[pos] & mask);
            }

        private:
            void generateSeive()
            {
                auto sqN  = fastIntSqrt(N);
                auto size = raft::ceildiv<unsigned>(N, sizeof(unsigned) * 8);
                seive.resize(size);
                // assume all to be primes initially
                for(auto& itr : seive)
                {
                    itr = 0xffffffffu;
                }
                unsigned cid  = 0;
                unsigned cnum = getNum(cid);
                while(cnum <= sqN)
                {
                    do
                    {
                        ++cid;
                        cnum = getNum(cid);
                        if(isPrime(cnum))
                        {
                            break;
                        }
                    } while(cnum <= sqN);
                    auto cnum2 = cnum << 1;
                    // 'unmark' all the 'odd' multiples of the current prime
                    for(unsigned i = 3, num = i * cnum; num <= N; i += 2, num += cnum2)
                    {
                        unmark(num);
                    }
                }
            }

            unsigned getId(unsigned num) const
            {
                return (num >> 1);
            }

            unsigned getNum(unsigned id) const
            {
                if(id == 0)
                {
                    return 2;
                }
                return ((id << 1) + 1);
            }

            void getMaskPos(unsigned num, unsigned& mask, unsigned& pos) const
            {
                pos  = getId(num);
                mask = 1 << (pos & 0x1f);
                pos >>= 5;
            }

            void unmark(unsigned num)
            {
                unsigned mask, pos;
                getMaskPos(num, mask, pos);
                seive[pos] &= ~mask;
            }

            // REF: http://www.azillionmonkeys.com/qed/ulerysqroot.pdf
            unsigned fastIntSqrt(unsigned val)
            {
                unsigned g     = 0;
                auto     bshft = 15u, b = 1u << bshft;
                do
                {
                    unsigned temp = ((g << 1) + b) << bshft--;
                    if(val >= temp)
                    {
                        g += b;
                        val -= temp;
                    }
                } while(b >>= 1);
                return g;
            }

            /** find all primes till this number */
            unsigned N;
            /** the seive */
            std::vector<unsigned> seive;
        };
    }; // namespace common
}; // namespace raft
