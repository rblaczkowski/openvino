//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include "ngraph/ops.hpp"
#include "op/avg_pool.hpp"
#include "op/convolution.hpp"
#include "op/group_conv.hpp"

namespace ngraph
{
    namespace opset0
    {
#ifdef NGRAPH_OP
#include "opset0_tbl.hpp"
#else
#define NGRAPH_OP(a, b) using b::a;
#include "opset0_tbl.hpp"
#undef NGRAPH_OP
#endif
    }
}
