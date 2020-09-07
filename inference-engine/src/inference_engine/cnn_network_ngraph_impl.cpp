// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cnn_network_ngraph_impl.hpp"

#include <cpp/ie_cnn_network.h>
#include <ie_common.h>
#include <math.h>

#include <cassert>
#include <map>
#include <memory>
#include <vector>
#include <unordered_set>
#include <ngraph/ngraph.hpp>
#include <ngraph/graph_util.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/manager.hpp>
#include <set>
#include <string>

#include <transformations/utils/utils.hpp>
#include <transformations/convert_opset1_to_legacy/convert_one_hot_to_one_hot_ie.hpp>

#include "ngraph_ops/eltwise.hpp"
#include "exec_graph_info.hpp"
#include <legacy/ie_ngraph_utils.hpp>
#include "ie_itt.hpp"
#include "network_serializer.hpp"
#include "generic_ie.hpp"
#include <legacy/shape_infer/built-in/ie_built_in_holder.hpp>

using namespace std;
using namespace InferenceEngine;
using details::CNNNetworkNGraphImpl;
using InferenceEngine::details::CNNNetworkNGraphImpl;
using ngraph::Function;

static std::shared_ptr<ngraph::Function> copyFunction(const std::shared_ptr<const ngraph::Function>& func,
                                                      bool constFolding) {
    OV_ITT_SCOPED_TASK(itt::domains::IE, "copyFunction");

    ::ngraph::op::GenericIE::DisableReshape noReshape(func);
    auto specialized_function = ngraph::clone_function(*func);

    if (constFolding) {
        ngraph::pass::ConstantFolding().run_on_function(specialized_function);
    }
    return specialized_function;
}

CNNNetwork::CNNNetwork(const std::shared_ptr<ngraph::Function>& graph) {
    OV_ITT_SCOPED_TASK(itt::domains::IE, "CNNNetwork::CNNNetwork");

    if (graph == nullptr) {
        THROW_IE_EXCEPTION << "CNNNetwork was not initialized: 'graph' object is empty";
    }

    // Create CNNNetworkNGraphImpl
    network = std::make_shared<CNNNetworkNGraphImpl>(graph);
    actual = network.get();
    if (actual == nullptr) {
        THROW_IE_EXCEPTION << "CNNNetwork was not initialized.";
    }
}

void CNNNetworkNGraphImpl::createDataForResult(const ::ngraph::Output<::ngraph::Node>& output, const std::string& outName,
                                               DataPtr& ptr) {
    const auto isCompatible = [](size_t size, const Layout& l) -> bool {
        switch (size) {
        case 0:
            return l == Layout::SCALAR;
        case 1:
            return l == Layout::C;
        case 2:
            return l == Layout::CN || l == Layout::HW || l == Layout::NC;
        case 3:
            return l == Layout::CHW;
        case 4:
            return l == Layout::NCHW || l == Layout::NHWC;
        case 5:
            return l == Layout::NCDHW || l == Layout::NDHWC;
        default:
            return false;
        }
    };
    // query shape from ngraph::Parameter output shape and check there are no zeros in it
    SizeVector dims;
    if (output.get_partial_shape().is_static()) {
        dims = output.get_shape();
    }
    for (const auto& dim : dims) {
        if (!dim)
            THROW_IE_EXCEPTION << outName << " has zero dimension which is not allowed";
    }

    if (ptr) {
        const auto origLayout = ptr->getTensorDesc().getLayout();
        const auto layout = isCompatible(dims.size(), origLayout) ? origLayout : TensorDesc::getLayoutByDims(dims);
        ptr->reshape(dims, layout);
    } else {
        const auto layout = TensorDesc::getLayoutByDims(dims);
        const auto precision = details::convertPrecision(output.get_element_type());
        ptr.reset(new Data(outName, {precision, dims, layout}));
    }
}

CNNNetworkNGraphImpl::CNNNetworkNGraphImpl(const std::shared_ptr<Function>& nGraph)
    : _ngraph_function(nGraph) {
    // Restore usual attributes for ICNNNetwork
    auto keep_input_info = [](CNNNetworkNGraphImpl& network, const DataPtr& inData) {
        InputInfo::Ptr info(new InputInfo());
        info->setInputData(inData);
        Precision prc = info->getPrecision();

        // Convert precision into native format (keep element size)
        prc = prc == Precision::Q78
                  ? Precision::I16
                  : prc == Precision::FP16 ? Precision::FP32 : static_cast<Precision::ePrecision>(prc);

        info->setPrecision(prc);
        network.setInputInfo(info);
    };

    // Add shape infer method for old operations which are not included to opset1, opset2 and opset3
    ::ngraph::op::GenericIE::addExtension(_ngraph_function, std::make_shared<ShapeInfer::BuiltInShapeInferHolder>());

    reshape();
    for (const auto& layer : _ngraph_function->get_parameters()) {
        std::string outName = layer->get_friendly_name();
        IE_ASSERT(layer->get_output_size() == 1);  // Parameter as only singly output port

        DataPtr& ptr = _data[outName];
        IE_ASSERT(ptr);  // Data must be allocated after the reshape method

        keep_input_info(*this, ptr);
    }
    for (auto& output : _outputData) {
        // Convert precision into native format. Be consistent with possible conversion to CNNNetwork later.
        if (output.second->getPrecision() == Precision::I64) {
            output.second->setPrecision(Precision::I32);
        } else if (output.second->getPrecision() != Precision::FP32 &&
            output.second->getPrecision() != Precision::I32) {
            output.second->setPrecision(Precision::FP32);
        }
    }
}

CNNNetworkNGraphImpl::CNNNetworkNGraphImpl(const ICNNNetwork& network) {
    if (network.getFunction() == nullptr) {
        THROW_IE_EXCEPTION << "Cannot create CNNNetwork with nGraph from legacy network format!";
    }

    _ngraph_function = copyFunction(network.getFunction(), false);
    InputsDataMap inputs;
    OutputsDataMap outputs;
    network.getInputsInfo(inputs);
    network.getOutputsInfo(outputs);

    for (const auto& outputInfo : outputs) {
        const auto& name = outputInfo.second->getName();
        DataPtr output = std::make_shared<Data>(name, outputInfo.second->getTensorDesc());
        _outputData[name] = output;
        _data[name] = output;
    }
    for (const auto& inputInfo : inputs) {
        InputInfo::Ptr info = std::make_shared<InputInfo>();
        const auto& name = inputInfo.second->getInputData()->getName();
        DataPtr input = std::make_shared<Data>(name, inputInfo.second->getInputData()->getTensorDesc());
        _data[name] = input;
        info->setInputData(input);
        info->getPreProcess() = inputInfo.second->getPreProcess();
        info->setPrecision(inputInfo.second->getPrecision());
        info->setLayout(inputInfo.second->getLayout());
        _inputData[name] = info;
    }
}

void CNNNetworkNGraphImpl::setInputInfo(InputInfo::Ptr data) {
    if (cnnNetwork) cnnNetwork->setInputInfo(data);
    _inputData[data->name()] = data;
}

const std::string& CNNNetworkNGraphImpl::getName() const noexcept {
    if (cnnNetwork) {
        return cnnNetwork->getName();
    }
    return _ngraph_function->get_friendly_name();
}

InputInfo::Ptr CNNNetworkNGraphImpl::getInput(const std::string& inputName) const noexcept {
    if (cnnNetwork) return cnnNetwork->getInput(inputName);
    auto it = _inputData.find(inputName);
    if (it == _inputData.end()) {
        return nullptr;
    }
    return it->second;
}

void CNNNetworkNGraphImpl::getOutputsInfo(OutputsDataMap& out) const noexcept {
    if (cnnNetwork) {
        cnnNetwork->getOutputsInfo(out);
        return;
    }
    out = _outputData;
}

void CNNNetworkNGraphImpl::getInputsInfo(InputsDataMap& inputs) const noexcept {
    if (cnnNetwork) {
        cnnNetwork->getInputsInfo(inputs);
        return;
    }
    inputs = _inputData;
}

size_t CNNNetworkNGraphImpl::layerCount() const noexcept {
    if (cnnNetwork) return cnnNetwork->layerCount();
    return _ngraph_function->get_ops().size();
}

void CNNNetworkNGraphImpl::validate(int version) {
    if (cnnNetwork)
        cnnNetwork->validate();
    else
        _ngraph_function->validate_nodes_and_infer_types();
}

StatusCode CNNNetworkNGraphImpl::addOutput(const std::string& layerName, size_t outputIndex,
                                           ResponseDesc* resp) noexcept {
    OV_ITT_SCOPED_TASK(itt::domains::IE, "CNNNetworkNGraphImpl::addOutput");

    if (cnnNetwork) {
        return cnnNetwork->addOutput(layerName, outputIndex, resp);
    }

    try {
        for (const auto & layer : _ngraph_function->get_ops()) {
            if (layer->get_friendly_name() == layerName) {
                auto& results = const_cast<::ngraph::ResultVector&>(_ngraph_function->get_results());
                auto result = make_shared<::ngraph::op::Result>(layer->output(outputIndex));
                results.push_back(result);

                std::string outputName = layerName;
                if (layer->outputs().size() != 1) {
                    outputName += "." + std::to_string(outputIndex);
                }
                if (_outputData.count(outputName) == 0) {
                    reshape();
                }
                return OK;
            }
        }
    } catch (...) {
        return GENERAL_ERROR;
    }
    return DescriptionBuffer(NOT_FOUND, resp) << "Cannot add output! Layer " << layerName << " wasn't found!";
}

void CNNNetworkNGraphImpl::addOutput(const ::ngraph::Output<::ngraph::Node> & output) {
    auto dataName = ngraph::op::util::create_ie_output_name(output);
    DataPtr data;
    if (_data.count(dataName))
        data = _data[dataName];
    createDataForResult(output, dataName, data);
    _data[dataName] = data;
    _outputData[dataName] = data;
}

size_t CNNNetworkNGraphImpl::getBatchSize() const noexcept {
    // TODO Provide adequate implementation.
    // The original code from CNNNetworkImpl just gets the first input and returns the first dimension.
    // This is not correct in general. We can follow the same semantics, but order of inputs should be
    // guaranteed to be the same.
    if (cnnNetwork) {
        return cnnNetwork->getBatchSize();
    }
    auto params = _ngraph_function->get_parameters();
    for (const auto& param : params) {
        if (param->get_partial_shape().is_dynamic())
            continue;
        auto shape = param->get_shape();
        // WA: for speech recognition and scalar layouts (copy-past from CNNNetwork)
        if (!shape.empty() && shape.size() != 3 && shape.size() != 1)
            return shape[0];
    }
    return 1;
}

std::shared_ptr<ngraph::Function> CNNNetworkNGraphImpl::cloneFunction(bool constFolding) const {
    return copyFunction(_ngraph_function, constFolding);
}

void CNNNetworkNGraphImpl::reshape() {
    ResponseDesc desc;

    // Disable reshape for generic nodes
    ::ngraph::op::GenericIE::DisableReshape noReshape(_ngraph_function);
    StatusCode ret = reshape({}, &desc);
    if (ret != OK)
        THROW_IE_EXCEPTION << desc.msg;
}

StatusCode
CNNNetworkNGraphImpl::reshape(const std::map<std::string, std::vector<size_t>>& inputShapes,
                        ResponseDesc* responseDesc) noexcept {
    OV_ITT_SCOPED_TASK(itt::domains::IE, "CNNNetworkNGraphImpl::reshape");

    if (cnnNetwork)
        return cnnNetwork->reshape(inputShapes, responseDesc);
    try {
        auto params = _ngraph_function->get_parameters();

        for (size_t i = 0; i < params.size(); i++) {
            const auto& param = params[i];
            if (inputShapes.find(param->get_friendly_name()) == inputShapes.end())
                continue;
            ::ngraph::PartialShape shape(inputShapes.at(param->get_friendly_name()));
            auto newParam = std::make_shared<::ngraph::op::Parameter>(param->get_element_type(), shape);
            newParam->set_friendly_name(param->get_friendly_name());
            _ngraph_function->replace_parameter(i, newParam);
        }
        _ngraph_function->validate_nodes_and_infer_types();

        {
            auto specialized_ngraph_function = cloneFunction(true);
            // Call this transformation because OneHot IE and nGraph have different output precisions
            {
                OV_ITT_SCOPED_TASK(itt::domains::IE, "CNNNetworkNGraphImpl::ConvertOneHot");
                ::ngraph::pass::Manager manager;
                manager.register_pass<::ngraph::pass::ConvertOneHotToOneHotIEMatcher>()->detect_output_type(
                        specialized_ngraph_function);
                manager.run_passes(specialized_ngraph_function);
            }
            specialized_ngraph_function->validate_nodes_and_infer_types();

#if 0
            for (const auto &op : specialized_ngraph_function->get_ordered_ops()) {
                cout << "[ " <<  op->description() << " ] " << op->get_friendly_name() << endl;
                cout << "    Inputs: ";
                for (const auto &in : op->inputs()) {
                    cout << "[" << in.get_element_type().get_type_name() << "]";
                    if (in.get_partial_shape().is_dynamic()) {
                        cout << "dyn_shape";
                    } else {
                        cout << "{";
                        bool first = true;
                        for (auto i : in.get_shape()) {
                            if (!first) cout << ",";
                            cout << i;
                            first = false;
                        }
                        cout << "} ";
                    }
                }
                cout << endl << "    Outputs: ";
                for (const auto &in : op->outputs()) {
                    cout << "[" << in.get_element_type().get_type_name() << "]";
                    if (in.get_partial_shape().is_dynamic()) {
                        cout << "dyn_shape";
                    } else {
                        cout << "{";
                        bool first = true;
                        for (auto i : in.get_shape()) {
                            if (!first) cout << ",";
                            cout << i;
                            first = false;
                        }
                        cout << "} ";
                    }
                }
                cout << endl;
            }
#endif
            std::unordered_set<std::string> opName;
            for (const auto &result : specialized_ngraph_function->get_results()) {
                addOutput(result->input_value(0));
            }

            for (const auto &parameter : specialized_ngraph_function->get_parameters()) {
                const auto &outName = parameter->get_friendly_name();
                if (opName.find(outName) != opName.end()) {
                    THROW_IE_EXCEPTION << "All operations in nGraph function should have unique friendly names!";
                }
                opName.insert(outName);
                createDataForResult(parameter, outName, _data[outName]);
            }
        }
    } catch (std::exception& ex) {
        return DescriptionBuffer(GENERAL_ERROR, responseDesc) << ex.what();
    }

    return OK;
}

StatusCode CNNNetworkNGraphImpl::serialize(const std::string& xmlPath, const std::string& binPath,
                                           ResponseDesc* resp) const noexcept {
    auto network = cnnNetwork;
    if (!network) {
        // TODO: once Serialization::SerializeV10 supports true IR v10
        // remove this conversion and WA for execution graph
        try {
            bool isExecutionGraph = true;
            for (const auto & op : _ngraph_function->get_ops()) {
                auto & rtInfo = op->get_rt_info();
                if (rtInfo.find(ExecGraphInfoSerialization::PERF_COUNTER) == rtInfo.end()) {
                    isExecutionGraph = false;
                    break;
                }
            }
            if (isExecutionGraph) {
                Serialization::SerializeV10(xmlPath, binPath, (InferenceEngine::ICNNNetwork&)*this);
                return OK;
            }

#ifdef ENABLE_V7_SERIALIZE
            network = std::make_shared<details::CNNNetworkImpl>(*this);
#endif
        } catch (const InferenceEngineException& e) {
            return DescriptionBuffer(GENERAL_ERROR, resp) << e.what();
        } catch (const std::exception& e) {
            return DescriptionBuffer(UNEXPECTED, resp) << e.what();
        } catch (...) {
            return DescriptionBuffer(UNEXPECTED, resp);
        }
    }

#ifdef ENABLE_V7_SERIALIZE
    return network->serialize(xmlPath, binPath, resp);
#else
    return DescriptionBuffer(NOT_IMPLEMENTED, resp) << "The serialize for IR v10 is not implemented";
#endif
}

StatusCode CNNNetworkNGraphImpl::setBatchSize(size_t size, ResponseDesc* responseDesc) noexcept {
    try {
        if (size == getBatchSize())
            return OK;
        if (!cnnNetwork)
            convertToCNNNetworkImpl();
        return cnnNetwork->setBatchSize(size, responseDesc);
    } catch (std::exception& ex) {
        return DescriptionBuffer(GENERAL_ERROR, responseDesc) << ex.what();
    }
}

StatusCode CNNNetworkNGraphImpl::setBatchSizeReshape(size_t size, ResponseDesc* responseDesc) noexcept {
    if (cnnNetwork)
        return cnnNetwork->setBatchSizeReshape(size, responseDesc);
    try {
        auto original_parameters = _ngraph_function->get_parameters();

        std::map<std::string, std::vector<size_t>> origShapes;
        std::map<std::string, std::vector<size_t>> inShapes;
        for (const auto &parameter : original_parameters) {
            if (parameter->get_partial_shape().is_dynamic())
                THROW_IE_EXCEPTION << "Cannot setBatch! Network contains inputs with dynamic shapes!";
            std::vector<size_t> shape = parameter->get_shape();
            origShapes[parameter->get_friendly_name()] = shape;
            shape[0] = size;
            inShapes[parameter->get_friendly_name()] = shape;
        }
        auto sts = reshape(inShapes, responseDesc);
        if (sts == OK) return OK;
        for (size_t i = 0; i < original_parameters.size(); i++) {
            const auto& param = original_parameters[i];
            if (origShapes.find(param->get_friendly_name()) == origShapes.end())
                continue;
            ::ngraph::PartialShape shape(origShapes.at(param->get_friendly_name()));
            auto newParam = std::make_shared<::ngraph::op::Parameter>(param->get_element_type(), shape);
            newParam->set_friendly_name(param->get_friendly_name());
            _ngraph_function->replace_parameter(i, newParam);
        }
        convertToCNNNetworkImpl();
        return cnnNetwork->setBatchSize(size, responseDesc);
    } catch (std::exception& ex) {
        return DescriptionBuffer(GENERAL_ERROR, responseDesc) << ex.what();
    }
}

void CNNNetworkNGraphImpl::convertToCNNNetworkImpl() {
    OV_ITT_SCOPED_TASK(itt::domains::IE, "CNNNetworkNGraphImpl::convertToCNNNetworkImpl");
    if (!cnnNetwork)
        cnnNetwork = std::make_shared<details::CNNNetworkImpl>(*this);
}
