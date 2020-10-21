// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/function.hpp>

#include <memory>
#include <string>
#include <vector>

#include <legacy/cnn_network_impl.hpp>

namespace InferenceEngine {
namespace details {

INFERENCE_ENGINE_API_CPP(std::shared_ptr<CNNNetworkImpl>)
convertFunctionToICNNNetwork(const std::shared_ptr<const ::ngraph::Function>& graph,
                             const ICNNNetwork &network, bool keep_constant_inputs = false);

INFERENCE_ENGINE_API_CPP(void)
convertFunctionToICNNNetwork(const std::shared_ptr<const ::ngraph::Function>& graph,
                             const ICNNNetwork &ngraphNetwork, 
                             CNNNetworkImpl* cnnNetworkImpl,
                             bool keep_constant_inputs = false);


}  // namespace details
}  // namespace InferenceEngine
