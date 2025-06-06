// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: Copyright (c) 2025, Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: Apache-2.0
// SPDX-License-Identifier: MIT

/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <type_traits>

namespace hipgraph
{
    namespace internals
    {

        class Callback
        {
        public:
            virtual ~Callback() {}
        };

        class GraphBasedDimRedCallback : public Callback
        {
        public:
            template <typename T>
            void setup(int n, int n_components)
            {
                this->n            = n;
                this->n_components = n_components;
                this->isFloat      = std::is_same<T, float>::value;
            }
            virtual void on_preprocess_end(void* positions) = 0;
            virtual void on_epoch_end(void* positions)      = 0;
            virtual void on_train_end(void* positions)      = 0;

        protected:
            int  n;
            int  n_components;
            bool isFloat;
        };

    } // namespace internals
} // namespace hipgraph
