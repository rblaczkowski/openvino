// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "description_buffer.hpp"
#include "ie_icore.hpp"
#include "cpp_interfaces/impl/ie_plugin_internal.hpp"
#include <memory>
#include <string>
#include <map>
#include <unordered_map>
#include <vector>
#include <utility>
#include <legacy/ie_util_internal.hpp>

namespace HeteroPlugin {

class Engine : public InferenceEngine::InferencePluginInternal {
public:
    using Configs = std::map<std::string, std::string>;
    using DeviceMetaInformationMap = std::unordered_map<std::string, Configs>;

    Engine();

    InferenceEngine::ExecutableNetworkInternal::Ptr
    LoadExeNetworkImpl(const InferenceEngine::ICNNNetwork &network, const Configs &config) override;

    void SetConfig(const Configs &config) override;

    void QueryNetwork(const InferenceEngine::ICNNNetwork &network,
                      const Configs& config, InferenceEngine::QueryNetworkResult &res) const override;

    InferenceEngine::Parameter GetMetric(const std::string& name, const std::map<std::string,
                                         InferenceEngine::Parameter> & options) const override;

    InferenceEngine::Parameter GetConfig(const std::string& name, const std::map<std::string,
                                         InferenceEngine::Parameter> & options) const override;

    ExecutableNetwork ImportNetworkImpl(std::istream& heteroModel, const Configs& config) override;


    void SetAffinity(InferenceEngine::ICNNNetwork& network, const Configs &config);

    DeviceMetaInformationMap GetDevicePlugins(const std::string& targetFallback,
        const Configs & localConfig) const;

private:
    Configs GetSupportedConfig(const Configs& config, const std::string & deviceName) const;
};

struct HeteroLayerColorer {
    explicit HeteroLayerColorer(const std::vector<std::string>& devices);

    void operator() (const CNNLayerPtr layer,
                    ordered_properties &printed_properties,
                    ordered_properties &node_properties);

    std::unordered_map<std::string, std::string> deviceColorMap;
};

}  // namespace HeteroPlugin
