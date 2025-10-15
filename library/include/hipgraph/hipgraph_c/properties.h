// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

//
// Speculative description of handling generic vertex and edge properties.
//
// If we have vertex properties and edge properties that we want to apply to an existing graph
// (after it was created) we could use these methods to construct C++ objects to represent these
// properties.
//
// These assume the use of external vertex ids and external edge ids as the mechanism for
// correlating a property to a particular vertex or edge.
//

#include "resource_handle.h"


#include "hipgraph/hipgraph-common.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct hipgraph_vertex_property hipgraph_vertex_property_t;

typedef struct hipgraph_edge_property hipgraph_edge_property_t;

typedef struct hipgraph_vertex_property_view hipgraph_vertex_property_view_t;

typedef struct hipgraph_edge_property_view hipgraph_edge_property_view_t;



#ifdef __cplusplus
}
#endif
