// Copyright (2019) Sandia Corporation
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#ifndef MDSPAN_INCLUDE_EXPERIMENTAL_BITS_TRAIT_BACKPORTS_HPP_
#define MDSPAN_INCLUDE_EXPERIMENTAL_BITS_TRAIT_BACKPORTS_HPP_

#include "config.hpp"
#include "macros.hpp"

#include <type_traits>
#include <utility> // integer_sequence

//==============================================================================
// <editor-fold desc="Variable template trait backports (e.g., is_void_v)"> {{{1

#ifdef _MDSPAN_NEEDS_TRAIT_VARIABLE_TEMPLATE_BACKPORTS

#if _MDSPAN_USE_VARIABLE_TEMPLATES
namespace std
{

#define _MDSPAN_BACKPORT_TRAIT(TRAIT) \
    template <class... Args>          \
    _MDSPAN_INLINE_VARIABLE constexpr auto TRAIT##_v = TRAIT<Args...>::value;

    _MDSPAN_BACKPORT_TRAIT(is_assignable)
    _MDSPAN_BACKPORT_TRAIT(is_constructible)
    _MDSPAN_BACKPORT_TRAIT(is_convertible)
    _MDSPAN_BACKPORT_TRAIT(is_default_constructible)
    _MDSPAN_BACKPORT_TRAIT(is_trivially_destructible)
    _MDSPAN_BACKPORT_TRAIT(is_same)
    _MDSPAN_BACKPORT_TRAIT(is_empty)
    _MDSPAN_BACKPORT_TRAIT(is_void)

#undef _MDSPAN_BACKPORT_TRAIT

} // end namespace std

#endif // _MDSPAN_USE_VARIABLE_TEMPLATES

#endif // _MDSPAN_NEEDS_TRAIT_VARIABLE_TEMPLATE_BACKPORTS

// </editor-fold> end Variable template trait backports (e.g., is_void_v) }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="integer sequence (ugh...)"> {{{1

#if !defined(_MDSPAN_USE_INTEGER_SEQUENCE) || !_MDSPAN_USE_INTEGER_SEQUENCE

namespace std
{

    template <class T, T... Vals>
    struct integer_sequence
    {
        static constexpr std::size_t size() noexcept
        {
            return sizeof...(Vals);
        }
        using value_type = T;
    };

    template <std::size_t... Vals>
    using index_sequence = std::integer_sequence<std::size_t, Vals...>;

    namespace __detail
    {

        template <class T, T N, T I, class Result>
        struct __make_int_seq_impl;

        template <class T, T N, T... Vals>
        struct __make_int_seq_impl<T, N, N, integer_sequence<T, Vals...>>
        {
            using type = integer_sequence<T, Vals...>;
        };

        template <class T, T N, T I, T... Vals>
        struct __make_int_seq_impl<T, N, I, integer_sequence<T, Vals...>>
            : __make_int_seq_impl<T, N, I + 1, integer_sequence<T, Vals..., I>>
        {
        };

    } // end namespace __detail

    template <class T, T N>
    using make_integer_sequence =
        typename __detail::__make_int_seq_impl<T, N, 0, integer_sequence<T>>::type;

    template <std::size_t N>
    using make_index_sequence =
        typename __detail::__make_int_seq_impl<size_t, N, 0, integer_sequence<size_t>>::type;

    template <class... T>
    using index_sequence_for = make_index_sequence<sizeof...(T)>;

} // end namespace std

#endif

// </editor-fold> end integer sequence (ugh...) }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="standard trait aliases"> {{{1

#if !defined(_MDSPAN_USE_STANDARD_TRAIT_ALIASES) || !_MDSPAN_USE_STANDARD_TRAIT_ALIASES

namespace std
{

#define _MDSPAN_BACKPORT_TRAIT_ALIAS(TRAIT) \
    template <class... Args>                \
    using TRAIT##_t = typename TRAIT<Args...>::type;

    _MDSPAN_BACKPORT_TRAIT_ALIAS(remove_cv)
    _MDSPAN_BACKPORT_TRAIT_ALIAS(remove_reference)

    template <bool _B, class _T = void>
    using enable_if_t = typename enable_if<_B, _T>::type;

#undef _MDSPAN_BACKPORT_TRAIT_ALIAS

} // end namespace std

#endif

// </editor-fold> end standard trait aliases }}}1
//==============================================================================

#endif //MDSPAN_INCLUDE_EXPERIMENTAL_BITS_TRAIT_BACKPORTS_HPP_
