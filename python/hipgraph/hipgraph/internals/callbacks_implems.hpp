// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: Apache-2.0
// SPDX-License-Identifier: MIT

/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <Python.h>
#include "internals.hpp"

#include <iostream>

namespace hipgraph
{
    namespace internals
    {

        class DefaultGraphBasedDimRedCallback : public GraphBasedDimRedCallback
        {
        public:
            PyObject* get_numba_matrix(void* positions)
            {
                PyObject* pycl = (PyObject*)this->pyCallbackClass;

                if(isFloat)
                {
                    return PyObject_CallMethod(pycl,
                                               "get_numba_matrix",
                                               "(l(ll)s)",
                                               positions,
                                               n,
                                               n_components,
                                               "float32");
                }
                else
                {
                    return PyObject_CallMethod(pycl,
                                               "get_numba_matrix",
                                               "(l(ll)s)",
                                               positions,
                                               n,
                                               n_components,
                                               "float64");
                }
            }

            void on_preprocess_end(void* positions) override
            {
                PyObject* numba_matrix = get_numba_matrix(positions);
                PyObject* res          = PyObject_CallMethod(
                    this->pyCallbackClass, "on_preprocess_end", "(O)", numba_matrix);
                Py_DECREF(numba_matrix);
                Py_DECREF(res);
            }

            void on_epoch_end(void* positions) override
            {
                PyObject* numba_matrix = get_numba_matrix(positions);
                PyObject* res          = PyObject_CallMethod(
                    this->pyCallbackClass, "on_epoch_end", "(O)", numba_matrix);
                Py_DECREF(numba_matrix);
                Py_DECREF(res);
            }

            void on_train_end(void* positions) override
            {
                PyObject* numba_matrix = get_numba_matrix(positions);
                PyObject* res          = PyObject_CallMethod(
                    this->pyCallbackClass, "on_train_end", "(O)", numba_matrix);
                Py_DECREF(numba_matrix);
                Py_DECREF(res);
            }

        public:
            PyObject* pyCallbackClass;
        };

    } // namespace internals
} // namespace hipgraph
