// Copyright (2019) Sandia Corporation
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#ifndef MDSPAN_BENCHMARKS_FILL_HPP
#define MDSPAN_BENCHMARKS_FILL_HPP

#include <experimental/mdspan>

#include <benchmark/benchmark.h>

#include <memory>
#include <random>

namespace stdex = std::experimental;

#if !(defined(__cpp_lib_make_unique) && __cpp_lib_make_unique >= 201304) && !MDSPAN_HAS_CXX_14
// Not actually conforming, but it works for the purposes of this file
namespace std
{
    template <class T>
    struct __unique_ptr_new_impl
    {
        template <class... Args>
        static T* __impl(Args&&... args)
        {
            return new T((Args&&)args...);
        }
    };

    template <class T>
    struct __unique_ptr_new_impl<T[]>
    {
        static T* __impl(size_t size)
        {
            return new T[size];
        }
    };

    template <class T, class... Args>
    std::unique_ptr<T> make_unique(Args&&... args)
    {
        return std::unique_ptr<T>(__unique_ptr_new_impl<T>::__impl((Args&&)args...));
    }
} // end namespace std
#endif

namespace mdspan_benchmark
{

    namespace _impl
    {

        template <class, class T>
        T&& _repeated_with(T&& v) noexcept
        {
            return std::forward<T>(v);
        }

        template <class T, class SizeT, class... Rest, class RNG, class Dist>
        void _do_fill_random(
            std::experimental::mdspan<T, std::experimental::extents<SizeT>, Rest...> s,
            RNG&                                                                     gen,
            Dist&                                                                    dist)
        {
            s() = dist(gen);
        }

        template <class T,
                  class SizeT,
                  size_t E,
                  size_t... Es,
                  class... Rest,
                  class RNG,
                  class Dist>
        void _do_fill_random(
            std::experimental::mdspan<T, std::experimental::extents<SizeT, E, Es...>, Rest...> s,
            RNG&                                                                               gen,
            Dist&                                                                              dist)
        {
            for(SizeT i = 0; i < s.extent(0); ++i)
            {
                _do_fill_random(
                    std::experimental::submdspan(
                        s, i, _repeated_with<decltype(Es)>(std::experimental::full_extent)...),
                    gen,
                    dist);
            }
        }

    } // end namespace _impl

    template <class T, class E, class... Rest>
    void fill_random(std::experimental::mdspan<T, E, Rest...> s, long long seed = 1234)
    {
        std::mt19937 gen(seed);
        auto         val_dist = std::uniform_int_distribution<>(0, 127);
        _impl::_do_fill_random(s, gen, val_dist);
    }

} // namespace mdspan_benchmark

//==============================================================================
// <editor-fold desc="A helpful template for instantiating all 3D combinations"> {{{1

#define MDSPAN_BENCHMARK_ALL_3D(bench_template, prefix, md_template, X, Y, Z)                    \
    BENCHMARK_CAPTURE(                                                                           \
        bench_template, prefix##fixed_##X##_##Y##_##Z, md_template<int, X, Y, Z>{nullptr});      \
    BENCHMARK_CAPTURE(bench_template,                                                            \
                      prefix##dyn_##X##_##Y##_d##Z,                                              \
                      md_template<int, X, Y, stdex::dynamic_extent>{},                           \
                      Z);                                                                        \
    BENCHMARK_CAPTURE(bench_template,                                                            \
                      prefix##dyn_##X##_d##Y##_##Z,                                              \
                      md_template<int, X, stdex::dynamic_extent, Z>{},                           \
                      Y);                                                                        \
    BENCHMARK_CAPTURE(bench_template,                                                            \
                      prefix##dyn_d##X##_##Y##_##Z,                                              \
                      md_template<int, stdex::dynamic_extent, Y, Z>{},                           \
                      X);                                                                        \
    BENCHMARK_CAPTURE(bench_template,                                                            \
                      prefix##dyn_##X##_d##Y##_d##Z,                                             \
                      md_template<int, X, stdex::dynamic_extent, stdex::dynamic_extent>{},       \
                      Y,                                                                         \
                      Z);                                                                        \
    BENCHMARK_CAPTURE(bench_template,                                                            \
                      prefix##dyn_d##X##_##Y##_d##Z,                                             \
                      md_template<int, stdex::dynamic_extent, Y, stdex::dynamic_extent>{},       \
                      X,                                                                         \
                      Z);                                                                        \
    BENCHMARK_CAPTURE(bench_template,                                                            \
                      prefix##dyn_d##X##_d##Y##_##Z,                                             \
                      md_template<int, stdex::dynamic_extent, stdex::dynamic_extent, Z>{},       \
                      X,                                                                         \
                      Y);                                                                        \
    BENCHMARK_CAPTURE(                                                                           \
        bench_template,                                                                          \
        prefix##dyn_d##X##_d##Y##_d##Z,                                                          \
        md_template<int, stdex::dynamic_extent, stdex::dynamic_extent, stdex::dynamic_extent>{}, \
        X,                                                                                       \
        Y,                                                                                       \
        Z)

#define MDSPAN_BENCHMARK_ALL_3D_MANUAL(bench_template, prefix, md_template, X, Y, Z)             \
    BENCHMARK_CAPTURE(                                                                           \
        bench_template, prefix##fixed_##X##_##Y##_##Z, md_template<int, X, Y, Z>{nullptr})       \
        ->UseManualTime();                                                                       \
    BENCHMARK_CAPTURE(bench_template,                                                            \
                      prefix##dyn_##X##_##Y##_d##Z,                                              \
                      md_template<int, X, Y, stdex::dynamic_extent>{},                           \
                      Z)                                                                         \
        ->UseManualTime();                                                                       \
    BENCHMARK_CAPTURE(bench_template,                                                            \
                      prefix##dyn_##X##_d##Y##_##Z,                                              \
                      md_template<int, X, stdex::dynamic_extent, Z>{},                           \
                      Y)                                                                         \
        ->UseManualTime();                                                                       \
    BENCHMARK_CAPTURE(bench_template,                                                            \
                      prefix##dyn_d##X##_##Y##_##Z,                                              \
                      md_template<int, stdex::dynamic_extent, Y, Z>{},                           \
                      X)                                                                         \
        ->UseManualTime();                                                                       \
    BENCHMARK_CAPTURE(bench_template,                                                            \
                      prefix##dyn_##X##_d##Y##_d##Z,                                             \
                      md_template<int, X, stdex::dynamic_extent, stdex::dynamic_extent>{},       \
                      Y,                                                                         \
                      Z)                                                                         \
        ->UseManualTime();                                                                       \
    BENCHMARK_CAPTURE(bench_template,                                                            \
                      prefix##dyn_d##X##_##Y##_d##Z,                                             \
                      md_template<int, stdex::dynamic_extent, Y, stdex::dynamic_extent>{},       \
                      X,                                                                         \
                      Z)                                                                         \
        ->UseManualTime();                                                                       \
    BENCHMARK_CAPTURE(bench_template,                                                            \
                      prefix##dyn_d##X##_d##Y##_##Z,                                             \
                      md_template<int, stdex::dynamic_extent, stdex::dynamic_extent, Z>{},       \
                      X,                                                                         \
                      Y)                                                                         \
        ->UseManualTime();                                                                       \
    BENCHMARK_CAPTURE(                                                                           \
        bench_template,                                                                          \
        prefix##dyn_d##X##_d##Y##_d##Z,                                                          \
        md_template<int, stdex::dynamic_extent, stdex::dynamic_extent, stdex::dynamic_extent>{}, \
        X,                                                                                       \
        Y,                                                                                       \
        Z)                                                                                       \
        ->UseManualTime()

#define MDSPAN_BENCHMARK_ALL_3D_REAL_TIME(bench_template, prefix, md_template, X, Y, Z)          \
    BENCHMARK_CAPTURE(                                                                           \
        bench_template, prefix##fixed_##X##_##Y##_##Z, md_template<int, X, Y, Z>{nullptr})       \
        ->UseRealTime();                                                                         \
    BENCHMARK_CAPTURE(bench_template,                                                            \
                      prefix##dyn_##X##_##Y##_d##Z,                                              \
                      md_template<int, X, Y, stdex::dynamic_extent>{},                           \
                      Z)                                                                         \
        ->UseRealTime();                                                                         \
    BENCHMARK_CAPTURE(bench_template,                                                            \
                      prefix##dyn_##X##_d##Y##_##Z,                                              \
                      md_template<int, X, stdex::dynamic_extent, Z>{},                           \
                      Y)                                                                         \
        ->UseRealTime();                                                                         \
    BENCHMARK_CAPTURE(bench_template,                                                            \
                      prefix##dyn_d##X##_##Y##_##Z,                                              \
                      md_template<int, stdex::dynamic_extent, Y, Z>{},                           \
                      X)                                                                         \
        ->UseRealTime();                                                                         \
    BENCHMARK_CAPTURE(bench_template,                                                            \
                      prefix##dyn_##X##_d##Y##_d##Z,                                             \
                      md_template<int, X, stdex::dynamic_extent, stdex::dynamic_extent>{},       \
                      Y,                                                                         \
                      Z)                                                                         \
        ->UseRealTime();                                                                         \
    BENCHMARK_CAPTURE(bench_template,                                                            \
                      prefix##dyn_d##X##_##Y##_d##Z,                                             \
                      md_template<int, stdex::dynamic_extent, Y, stdex::dynamic_extent>{},       \
                      X,                                                                         \
                      Z)                                                                         \
        ->UseRealTime();                                                                         \
    BENCHMARK_CAPTURE(bench_template,                                                            \
                      prefix##dyn_d##X##_d##Y##_##Z,                                             \
                      md_template<int, stdex::dynamic_extent, stdex::dynamic_extent, Z>{},       \
                      X,                                                                         \
                      Y)                                                                         \
        ->UseRealTime();                                                                         \
    BENCHMARK_CAPTURE(                                                                           \
        bench_template,                                                                          \
        prefix##dyn_d##X##_d##Y##_d##Z,                                                          \
        md_template<int, stdex::dynamic_extent, stdex::dynamic_extent, stdex::dynamic_extent>{}, \
        X,                                                                                       \
        Y,                                                                                       \
        Z)                                                                                       \
        ->UseRealTime()

// </editor-fold> end A helpful template for instantiating all 3D combinations }}}1
//==============================================================================

#endif // MDSPAN_BENCHMARKS_FILL_HPP
