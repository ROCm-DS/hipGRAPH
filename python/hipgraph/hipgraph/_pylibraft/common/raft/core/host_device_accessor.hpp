// Copyright (c) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once
#include <raft/core/memory_type.hpp>

#include <type_traits>

namespace raft
{

    /**
 * @brief A mixin to distinguish host and device memory. This is the primary
 * accessor used throughout RAFT's APIs to denote whether an underlying pointer
 * is accessible from device, host, or both.
 */
    template <typename AccessorPolicy, memory_type MemType>
    struct host_device_accessor : public AccessorPolicy
    {
        using accessor_type                  = AccessorPolicy;
        auto static constexpr const mem_type = MemType;
        using is_host_type                   = std::
            conditional_t<raft::is_host_accessible(mem_type), std::true_type, std::false_type>;
        using is_device_type = std::
            conditional_t<raft::is_device_accessible(mem_type), std::true_type, std::false_type>;
        using is_managed_type = std::conditional_t<raft::is_host_device_accessible(mem_type),
                                                   std::true_type,
                                                   std::false_type>;
        static constexpr bool is_host_accessible    = raft::is_host_accessible(mem_type);
        static constexpr bool is_device_accessible  = raft::is_device_accessible(mem_type);
        static constexpr bool is_managed_accessible = raft::is_host_device_accessible(mem_type);
        // make sure the explicit ctor can fall through
        using AccessorPolicy::AccessorPolicy;
        using offset_policy = host_device_accessor;
        host_device_accessor(AccessorPolicy const& that)
            : AccessorPolicy{that}
        {
        } // NOLINT

        // Prevent implicit conversion from incompatible host_device_accessor types
        template <memory_type OtherMemType>
        host_device_accessor(host_device_accessor<AccessorPolicy, OtherMemType> const& that)
            = delete;

        template <memory_type OtherMemType, typename = std::enable_if_t<mem_type == OtherMemType>>
        host_device_accessor(host_device_accessor<AccessorPolicy, OtherMemType> const& that)
            : AccessorPolicy{that}
        {
        }
    };

} // namespace raft
