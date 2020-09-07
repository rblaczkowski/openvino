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

#include "ngraph/op/sum.hpp"
#include "itt.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/sum.hpp"
#include "ngraph/shape_util.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v0::Sum::type_info;

op::v0::Sum::Sum(const Output<Node>& arg, const AxisSet& reduction_axes)
    : ArithmeticReduction(arg, reduction_axes)
{
    constructor_validate_and_infer_types();
}

op::v0::Sum::Sum(const Output<Node>& arg, const Output<Node>& reduction_axes)
    : ArithmeticReduction(arg, reduction_axes)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::Sum::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<op::v0::Sum>(new_args.at(0), new_args.at(1));
}

shared_ptr<Node> op::v0::Sum::get_default_value() const
{
    return ngraph::make_constant_from_string("0", get_element_type(), get_shape());
}

namespace
{
    template <element::Type_t ET>
    bool evaluate(const HostTensorPtr& arg,
                  const HostTensorPtr& out,
                  const AxisSet& axes,
                  bool keep_dims)
    {
        out->set_shape(reduce(arg->get_shape(), axes, false));
        runtime::reference::sum(
            arg->get_data_ptr<ET>(), out->get_data_ptr<ET>(), arg->get_shape(), axes, keep_dims);
        return true;
    }

    bool evaluate_sum(const HostTensorPtr& arg,
                      const HostTensorPtr& out,
                      const AxisSet& axes,
                      bool keep_dims)
    {
        bool rc = true;
        switch (arg->get_element_type())
        {
            TYPE_CASE(i32)(arg, out, axes, keep_dims);
            break;
            TYPE_CASE(i64)(arg, out, axes, keep_dims);
            break;
            TYPE_CASE(u32)(arg, out, axes, keep_dims);
            break;
            TYPE_CASE(u64)(arg, out, axes, keep_dims);
            break;
            TYPE_CASE(f16)(arg, out, axes, keep_dims);
            break;
            TYPE_CASE(f32)(arg, out, axes, keep_dims);
            break;
        default: rc = false; break;
        }
        return rc;
    }
}

bool op::v0::Sum::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const
{
    OV_ITT_SCOPED_TASK(itt::domains::nGraphOp, "op::v0::Sum::evaluate");
    return evaluate_sum(inputs[0], outputs[0], get_reduction_axes(), false);
}
