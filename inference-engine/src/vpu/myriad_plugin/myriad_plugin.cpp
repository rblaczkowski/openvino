// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>
#include <tuple>
#include <utility>

#include <ie_metric_helpers.hpp>
#include <cpp/ie_cnn_network.h>
#include <cpp_interfaces/impl/ie_executable_network_internal.hpp>
#include <legacy/ie_util_internal.hpp>

#include <vpu/vpu_plugin_config.hpp>
#include <vpu/parsed_config.hpp>
#include <vpu/utils/profiling.hpp>
#include <vpu/utils/error.hpp>
#include <transformations/tensor_iterator_transformations/apply_transformations_to_ti_body.hpp>
#include <transformations/common_optimizations/common_optimizations.hpp>
#include <vpu/ngraph/transformations/convert_nms_4_to_nms_dynamic.hpp>

#include "vpu/ngraph/transformations/dynamic_to_static_shape.hpp"
#include "vpu/ngraph/transformations/eliminate_shapeof_after_dsr.hpp"

#include "generic_ie.hpp"

#include "myriad_plugin.h"

using namespace InferenceEngine;
using namespace InferenceEngine::PluginConfigParams;
using namespace InferenceEngine::VPUConfigParams;
using namespace vpu::MyriadPlugin;

ExecutableNetworkInternal::Ptr Engine::LoadExeNetworkImpl(
        const ICNNNetwork& network,
        const std::map<std::string, std::string>& config) {
    VPU_PROFILE(LoadExeNetworkImpl);

    auto parsedConfigCopy = _parsedConfig;
    parsedConfigCopy.update(config);

    auto clonedNetwork = cloneNetwork(network);
    if (auto function = clonedNetwork->getFunction()) {
        ngraph::op::GenericIE::DisableReshape noReshape(function);
        ngraph::pass::Manager manager;
        manager.register_pass<vpu::UpgradeNMS4ToNMSDynamic>();
        manager.register_pass<ngraph::pass::CommonOptimizations>();
        manager.register_pass<vpu::DynamicToStaticShape>();
        manager.register_pass<vpu::EliminateShapeOfAfterDSR>();
        manager.run_passes(function);

        ngraph::pass::Manager ti_manager;
        ti_manager.register_pass<ngraph::pass::ApplyTransformationsToTIBody>(manager);
        ti_manager.run_passes(function);
    }

    return std::make_shared<ExecutableNetwork>(*clonedNetwork,
        _mvnc, _devicePool,
        parsedConfigCopy, GetCore());
}

void Engine::SetConfig(const std::map<std::string, std::string> &config) {
    _parsedConfig.update(config);

    for (const auto& entry : config) {
        _config[entry.first] = entry.second;
    }
}

Parameter Engine::GetConfig(const std::string& name, const std::map<std::string, Parameter>& options) const {
    auto supported_keys = _metrics->SupportedConfigKeys();
    if (std::find(supported_keys.begin(),
        supported_keys.end(), name) == supported_keys.end()) {
        THROW_IE_EXCEPTION << "Unsupported config key : " << name;
    }

    Parameter result;
    auto option = _config.find(name);
    if (option != _config.end())
        result = option->second;

    return result;
}

void Engine::QueryNetwork(
        const ICNNNetwork& network,
        const std::map<std::string, std::string>& config,
        QueryNetworkResult& res) const {
    VPU_PROFILE(QueryNetwork);

    auto parsedConfigCopy = _parsedConfig;
    parsedConfigCopy.update(config);

    const auto deviceName = parsedConfigCopy.deviceName();
    if (!deviceName.empty()) {
        const auto deviceIDs = GetMetric(METRIC_KEY(AVAILABLE_DEVICES), {}).as<std::vector<std::string>>();
        VPU_THROW_UNLESS(!(std::find(deviceIDs.begin(), deviceIDs.end(), deviceName) == deviceIDs.end()), "Myriad device: {} not found.", deviceName);
    }

    if (network.getFunction()) {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str << " ngraph::Function is not supported natively";
    }

    const auto log = std::make_shared<Logger>(
        "GraphCompiler",
        parsedConfigCopy.logLevel(),
        defaultOutput(parsedConfigCopy.compilerLogFilePath()));

    const auto layerNames = getSupportedLayers(
        network,
        static_cast<Platform>(parsedConfigCopy.platform()),
        parsedConfigCopy.compileConfig(),
        log,
        GetCore());

    for (const auto& layerName : layerNames) {
        res.supportedLayersMap.insert({ layerName, GetName() });
    }
}

Engine::Engine(std::shared_ptr<IMvnc> mvnc) :
        _mvnc(std::move(mvnc)),
        _metrics(std::make_shared<MyriadMetrics>()) {
    VPU_THROW_UNLESS(_mvnc, "mvnc is null");

    _pluginName = "MYRIAD";

IE_SUPPRESS_DEPRECATED_START
    _config = {
        { MYRIAD_ENABLE_HW_ACCELERATION, CONFIG_VALUE(YES) },
        { MYRIAD_ENABLE_RECEIVING_TENSOR_TIME, CONFIG_VALUE(NO) },
        { MYRIAD_CUSTOM_LAYERS, "" },
        { MYRIAD_ENABLE_FORCE_RESET, CONFIG_VALUE(NO) },

        // Deprecated
        { KEY_VPU_HW_STAGES_OPTIMIZATION, CONFIG_VALUE(YES) },
        { KEY_VPU_PRINT_RECEIVE_TENSOR_TIME, CONFIG_VALUE(NO) },
        { KEY_VPU_CUSTOM_LAYERS, "" },
        { KEY_VPU_MYRIAD_FORCE_RESET, CONFIG_VALUE(NO) },
        { KEY_VPU_MYRIAD_PLATFORM, "" },

        { KEY_LOG_LEVEL, CONFIG_VALUE(LOG_NONE) },
        { KEY_EXCLUSIVE_ASYNC_REQUESTS, CONFIG_VALUE(NO) },
        { KEY_PERF_COUNT, CONFIG_VALUE(NO) },
        { KEY_CONFIG_FILE, "" },
        { KEY_DEVICE_ID, "" },
    };
IE_SUPPRESS_DEPRECATED_END
}

InferenceEngine::ExecutableNetwork Engine::ImportNetwork(
        std::istream& model,
        const std::map<std::string, std::string>& config) {
    VPU_PROFILE(ImportNetwork);

    auto parsedConfigCopy = _parsedConfig;
    parsedConfigCopy.update(config, ConfigMode::RunTime);

    const auto executableNetwork =
            std::make_shared<ExecutableNetwork>(
                model, _mvnc, _devicePool, parsedConfigCopy, GetCore());

    return make_executable_network(executableNetwork);
}

IExecutableNetwork::Ptr Engine::ImportNetwork(
        const std::string& modelFileName,
        const std::map<std::string, std::string>& config) {
    VPU_PROFILE(ImportNetwork);

    std::ifstream blobFile(modelFileName, std::ios::binary);

    if (!blobFile.is_open()) {
        THROW_IE_EXCEPTION << ie::details::as_status << NETWORK_NOT_READ;
    }

    return ImportNetwork(blobFile, config);
}

InferenceEngine::Parameter Engine::GetMetric(const std::string& name,
                                     const std::map<std::string, InferenceEngine::Parameter> & options) const {
    const auto mvnc = _mvnc;
    const auto metrics = _metrics;
    const auto devicePool = _devicePool;
    const auto getSpecifiedDeviceName = [&mvnc, &metrics, &devicePool, &options]() {
        if (options.count(KEY_DEVICE_ID)) {
            return options.at(KEY_DEVICE_ID).as<std::string>();
        }

        const auto availableDevices = metrics->AvailableDevicesNames(mvnc, devicePool);
        VPU_THROW_UNLESS(!availableDevices.empty(), "No devices available.");
        VPU_THROW_UNLESS(availableDevices.size() == 1, "KEY_DEVICE_ID is undefined.");

        return availableDevices.front();
    };
    const auto getDeviceByName = [&devicePool](const std::string& deviceName) {
        const auto deviceIt = std::find_if(
                devicePool.begin(), devicePool.end(), [&deviceName](DevicePtr device) {
                    return device->_name == deviceName;
                });
        if (deviceIt == devicePool.end()) {
            return DevicePtr();
        }
        return *deviceIt;
    };

    if (name == METRIC_KEY(AVAILABLE_DEVICES)) {
        IE_SET_METRIC_RETURN(AVAILABLE_DEVICES, _metrics->AvailableDevicesNames(_mvnc, _devicePool));
    } else if (name == METRIC_KEY(FULL_DEVICE_NAME)) {
        IE_SET_METRIC_RETURN(FULL_DEVICE_NAME, _metrics->FullName(getSpecifiedDeviceName()));
    } else if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        const auto& supportedMetrics = _metrics->SupportedMetrics();
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, std::vector<std::string>{supportedMetrics.cbegin(), supportedMetrics.cend()});
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        const auto& supportedConfigKeys = _metrics->SupportedConfigKeys();
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, std::vector<std::string>{supportedConfigKeys.cbegin(), supportedConfigKeys.cend()});
    } else if (name == METRIC_KEY(OPTIMIZATION_CAPABILITIES)) {
        const auto& optimizationCapabilities = _metrics->OptimizationCapabilities();
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, std::vector<std::string>{optimizationCapabilities.cbegin(), optimizationCapabilities.cend()});
    } else if (name == METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS)) {
        IE_SET_METRIC_RETURN(RANGE_FOR_ASYNC_INFER_REQUESTS, _metrics->RangeForAsyncInferRequests(_config));
    } else if (name == METRIC_KEY(DEVICE_THERMAL)) {
        const auto& device = getDeviceByName(getSpecifiedDeviceName());
        if (device != nullptr) {
            IE_SET_METRIC_RETURN(DEVICE_THERMAL, _metrics->DevicesThermal(device));
        } else {
            return Parameter();
        }
    }
    THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
}
