// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A file containing ngraph implementation of public ICNNNetwork interface
 * @file cnn_network_ngraph_impl.hpp
 */

#pragma once

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <ngraph/attribute_visitor.hpp>
#include <ngraph/function.hpp>
#include <ngraph/node.hpp>

#include <ie_icnn_network.hpp>
#include "description_buffer.hpp"
#include "ie_api.h"
#include "ie_blob.h"
#include "ie_common.h"
#include "ie_data.h"
#include "ie_input_info.hpp"

#include <legacy/cnn_network_impl.hpp>

namespace InferenceEngine {
namespace ShapeInfer {
class Reshaper;

using ReshaperPtr = std::shared_ptr<Reshaper>;
}  // namespace ShapeInfer

namespace details {

/**
 * @brief Ngraph-based implementation of the ICNNNetwork interface.
 */
class INFERENCE_ENGINE_API_CLASS(CNNNetworkNGraphImpl): public ICNNNetwork {
public:
    CNNNetworkNGraphImpl(const std::shared_ptr<::ngraph::Function>& nGraph);
    CNNNetworkNGraphImpl(const ICNNNetwork& nGraph);
    ~CNNNetworkNGraphImpl() override = default;

    void getOutputsInfo(std::map<std::string, DataPtr>& out) const noexcept override;

    void getInputsInfo(InputsDataMap& inputs) const noexcept override;

    InputInfo::Ptr getInput(const std::string& inputName) const noexcept override;
    const std::string& getName() const noexcept override;

    size_t layerCount() const noexcept override;

    void setInputInfo(InputInfo::Ptr data);

    // public version
    StatusCode setBatchSize(size_t size, ResponseDesc* responseDesc) noexcept override;

    // for internal usage (e.g. setBatch via reshape in tests)
    StatusCode setBatchSizeReshape(size_t size, ResponseDesc* responseDesc) noexcept;

    size_t getBatchSize() const noexcept override;

    StatusCode addOutput(const std::string& layerName, size_t outputIndex, ResponseDesc* resp) noexcept override;

    void addOutput(const ::ngraph::Output<::ngraph::Node> & dataName);

    void Release() noexcept override {
        delete this;
    }

    std::shared_ptr<const ::ngraph::Function> getFunction() const noexcept override {
        return !cnnNetwork ? _ngraph_function : nullptr;
    }
    std::shared_ptr<::ngraph::Function> getFunction() noexcept override {
        return !cnnNetwork ? _ngraph_function : nullptr;
    }

    virtual void validate(int = 10);

    StatusCode reshape(const std::map<std::string, std::vector<size_t>>& inputShapes,
                       ResponseDesc* resp) noexcept override;

    StatusCode serialize(const std::string& xmlPath, const std::string& binPath, ResponseDesc* resp) const
        noexcept override;

protected:
    virtual std::shared_ptr<::ngraph::Function> cloneFunction(bool constFolding = false) const;
    std::shared_ptr<::ngraph::Function> _ngraph_function;

private:
    std::map<std::string, DataPtr> _data;
    InferenceEngine::InputsDataMap _inputData;
    std::map<std::string, DataPtr> _outputData;
    std::shared_ptr<CNNNetworkImpl> cnnNetwork;

    /**
     * @brief Create DataPtr for nGraph operation
     *
     * @param output output port from nGraph op
     * @param outName name for DataPtr
     * @param ptr reference to new DataPtr
     */
    void createDataForResult(const ::ngraph::Output<::ngraph::Node>& output, const std::string& outName, DataPtr& ptr);

    /**
     * @brief Converts ngraph::Function to old CNNNetworkImpl representation
     */
    void convertToCNNNetworkImpl();

    friend INFERENCE_ENGINE_API_CPP(void)
    convertFunctionToICNNNetwork(const std::shared_ptr<const ::ngraph::Function>& graph,
                                 const ICNNNetwork& nGraphImpl,
                                 CNNNetworkImpl* cnnNetworkImpl,
                                 bool keep_constant_inputs);

    /**
     * @brief Reshape on the same shape
     */
    void reshape();
};

class TINGraphBody : public CNNNetworkNGraphImpl {
public:
    explicit TINGraphBody(const std::shared_ptr<::ngraph::Function>& func): CNNNetworkNGraphImpl(func) {}

protected:
    std::shared_ptr<::ngraph::Function> cloneFunction(bool constFolding) const override {
        return _ngraph_function;
    }
};

}  // namespace details
}  // namespace InferenceEngine
