// Copyright (c) 2022-2023, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <raft/core/comms.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>

namespace raft::resource
{
    class sub_comms_resource : public resource
    {
    public:
        sub_comms_resource()
            : communicators_()
        {
        }
        void* get_resource() override
        {
            return &communicators_;
        }

        ~sub_comms_resource() override {}

    private:
        std::unordered_map<std::string, std::shared_ptr<comms::comms_t>> communicators_;
    };

    /**
 * Factory that knows how to construct a
 * specific raft::resource to populate
 * the res_t.
 */
    class sub_comms_resource_factory : public resource_factory
    {
    public:
        resource_type get_resource_type() override
        {
            return resource_type::SUB_COMMUNICATOR;
        }
        resource* make_resource() override
        {
            return new sub_comms_resource();
        }
    };

    /**
 * @defgroup resource_sub_comms Subcommunicator resource functions
 * @{
 */

    inline const comms::comms_t& get_subcomm(const resources& res, std::string key)
    {
        if(!res.has_resource_factory(resource_type::SUB_COMMUNICATOR))
        {
            res.add_resource_factory(std::make_shared<sub_comms_resource_factory>());
        }

        auto sub_comms
            = res.get_resource<std::unordered_map<std::string, std::shared_ptr<comms::comms_t>>>(
                resource_type::SUB_COMMUNICATOR);
        auto sub_comm = sub_comms->at(key);
        RAFT_EXPECTS(nullptr != sub_comm.get(), "ERROR: Subcommunicator was not initialized");

        return *sub_comm;
    }

    inline void
        set_subcomm(resources const& res, std::string key, std::shared_ptr<comms::comms_t> subcomm)
    {
        if(!res.has_resource_factory(resource_type::SUB_COMMUNICATOR))
        {
            res.add_resource_factory(std::make_shared<sub_comms_resource_factory>());
        }
        auto sub_comms
            = res.get_resource<std::unordered_map<std::string, std::shared_ptr<comms::comms_t>>>(
                resource_type::SUB_COMMUNICATOR);
        sub_comms->insert(std::make_pair(key, subcomm));
    }

    /**
 * @}
 */

} // namespace raft::resource
