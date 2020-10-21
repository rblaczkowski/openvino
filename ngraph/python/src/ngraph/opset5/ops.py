# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

"""Factory functions for all ngraph ops."""
from typing import Callable, Iterable, List, Optional, Set, Union

import numpy as np
from functools import partial

from ngraph.impl import Node, Shape
from ngraph.impl.op import Constant, Parameter
from ngraph.opset_utils import _get_node_factory
from ngraph.utils.decorators import binary_op, nameable_op, unary_op
from ngraph.utils.input_validation import (
    assert_list_of_ints,
    check_valid_attributes,
    is_non_negative_value,
    is_positive_value,
)
from ngraph.utils.node_factory import NodeFactory
from ngraph.utils.tensor_iterator_types import (
    GraphBody,
    TensorIteratorSliceInputDesc,
    TensorIteratorMergedInputDesc,
    TensorIteratorInvariantInputDesc,
    TensorIteratorBodyOutputDesc,
    TensorIteratorConcatOutputDesc,
)
from ngraph.utils.types import (
    NodeInput,
    NumericData,
    NumericType,
    ScalarData,
    TensorShape,
    as_node,
    as_nodes,
    get_dtype,
    get_element_type,
    get_element_type_str,
    make_constant_node,
)

_get_node_factory_opset5 = partial(_get_node_factory, "opset5")

# -------------------------------------------- ops ------------------------------------------------


@nameable_op
def gather_nd(
    data: NodeInput,
    indices: NodeInput,
    batch_dims: Optional[int] = 0,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs GatherND.

    :param data:       N-D tensor with data for gathering
    :param indices:    K-D tensor of tuples with indices by which data is gathered
    :param batch_dims: Scalar value of batch dimensions
    :return: The new node which performs GatherND
    """
    inputs = as_nodes(data, indices)

    attributes = {
        "batch_dims": batch_dims
    }

    return _get_node_factory_opset5().create("GatherND", inputs, attributes)


@nameable_op
def log_softmax(data: NodeInput, axis: int, name: Optional[str] = None) -> Node:
    """Apply LogSoftmax operation on each element of input tensor.

    :param data: The tensor providing input data.
    :param axis: An axis along which LogSoftmax should be calculated
    :return: The new node with LogSoftmax operation applied on each element.
    """
    return _get_node_factory_opset5().create("LogSoftmax", [as_node(data)], {"axis": axis})


@nameable_op
def round(data: NodeInput, mode: str = "half_to_even", name: Optional[str] = None) -> Node:
    """Apply Round operation on each element of input tensor.

    :param data: The tensor providing input data.
    :param mode: Rule to round halfway cases. If set to 'half_to_even' then halfs round to the nearest even
        integer or rounding in such a way that the result heads away from zero if `mode` attribute is
        'half_away_from_zero`.
    :param name: An optional name of the output node.
    :return: The new node with Round operation applied on each element.
    """
    return _get_node_factory_opset5().create("Round", as_nodes(data), {"mode": mode.upper()})
