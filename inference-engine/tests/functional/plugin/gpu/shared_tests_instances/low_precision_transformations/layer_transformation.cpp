// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/low_precision_transformations/layer_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>
#include "generic_ie.hpp"

#include <legacy/net_pass.h>
#include <legacy/graph_transformer.h>
#include <legacy/convert_function_to_cnn_network.hpp>
#include <transformations/common_optimizations/common_optimizations.hpp>
#include <transformations/convert_opset1_to_legacy/convert_opset1_to_legacy.hpp>
#include <transformations/convert_opset2_to_opset1/convert_opset2_to_opset1.hpp>
#include <transformations/convert_opset3_to_opset2/convert_opset3_to_opset2.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/op/gelu.hpp>
#include <ngraph/pass/manager.hpp>
#include "ngraph_functions/pass/convert_prc.hpp"

#include "common_test_utils/common_utils.hpp"
#include <legacy/ie_util_internal.hpp>

#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/low_precision_transformations/layer_transformation.hpp"

#include "low_precision_transformations/transformer.hpp"
#include "low_precision_transformations/convolution.hpp"
#include "low_precision_transformations/scaleshift_to_convolution.hpp"
#include "low_precision_transformations/fully_connected.hpp"
#include "low_precision_transformations/gemm.hpp"

using namespace InferenceEngine::details;
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"

namespace LayerTestsUtils {

InferenceEngine::details::LowPrecisionTransformations LayerTransformation::getLowPrecisionTransformations(
    const InferenceEngine::details::LayerTransformation::Params& params) const {
    return LowPrecisionTransformer::getAllTransformations(params)
        .add<FullyConnectedTransformation>(
            InferenceEngine::details::LayerTransformation::Params(params).setSupportAsymmetricQuantization(false), "FullyConnected")
        .add<GemmTransformation>(
            InferenceEngine::details::LayerTransformation::Params(params).setSupportAsymmetricQuantization(false), "GEMM");
}

InferenceEngine::CNNNetwork LayerTransformation::transform(InferenceEngine::details::LayerTransformation::Params& params) {
    auto ngraphNetwork = InferenceEngine::CNNNetwork(function);
    std::shared_ptr<InferenceEngine::ICNNNetwork> clonedNetwork = InferenceEngine::cloneNetwork(ngraphNetwork);

    if (clonedNetwork->getFunction()) {
        const auto transformations_callback = [](const std::shared_ptr<const ::ngraph::Node> &node) -> bool {
            return std::dynamic_pointer_cast<const ::ngraph::opset2::Gelu>(node) ||
                std::dynamic_pointer_cast<const ::ngraph::opset2::BatchToSpace>(node);
        };
        auto nGraphFunc = clonedNetwork->getFunction();
        // Disable shape inference (WA for generic operations)
        ::ngraph::op::GenericIE::DisableReshape noReshape(nGraphFunc);

        // Note: instead of running all Conversion Transformations you can make up your own transformation pipeline
        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::CommonOptimizations>();
        manager.register_pass<ngraph::pass::ConvertOpSet3ToOpSet2>();
        manager.register_pass<ngraph::pass::ConvertOpSet2ToOpSet1>();
        manager.register_pass<ngraph::pass::ConvertOpSet1ToLegacy>();

        manager.set_callback(transformations_callback);
        manager.run_passes(nGraphFunc);
        clonedNetwork = InferenceEngine::details::convertFunctionToICNNNetwork(nGraphFunc, *clonedNetwork);
    }

    auto implNetwork = std::dynamic_pointer_cast<InferenceEngine::details::CNNNetworkImpl>(clonedNetwork);
    if (implNetwork) {
        // valid for CNNNetworkImpl only, while there's no API in ICNNNetwork to change network
        InferenceEngine::ConstTransformer transformator(implNetwork.get());
        transformator.fullTrim();
    }

    InferenceEngine::NetPass::ConvertPrecision(*implNetwork, InferenceEngine::Precision::FP16, InferenceEngine::Precision::FP32);

    auto transformer = getLowPrecisionTransformer(params);
    transformer.transform(*implNetwork);

    return InferenceEngine::CNNNetwork(implNetwork);
}

InferenceEngine::Precision LayerTransformation::getDeviceInternalPrecision(const InferenceEngine::Precision precision) {
    if (precision == InferenceEngine::Precision::FP16) {
        return InferenceEngine::Precision::FP32;
    }

    return precision;
}

InferenceEngine::CNNNetwork LayerTransformation::transform(const InferenceEngine::details::LowPrecisionTransformations& transformations) {
    // convert to old representation
    InferenceEngine::CNNNetwork ngraphNetwork(function);
    auto cnnNetworkImp = std::make_shared<InferenceEngine::details::CNNNetworkImpl>(ngraphNetwork);

    InferenceEngine::NetPass::ConvertPrecision(*cnnNetworkImp, InferenceEngine::Precision::FP16, InferenceEngine::Precision::FP32);

    InferenceEngine::details::LowPrecisionTransformer transformer(transformations);
    transformer.transform(*cnnNetworkImp);

    return InferenceEngine::CNNNetwork(cnnNetworkImp);
}

InferenceEngine::details::LayerTransformation::Params LayerTransformationParamsFactory::createParams() {
    return InferenceEngine::details::LayerTransformation::Params(
        true,
        true,
        true,
        InferenceEngine::details::LayerTransformation::QuantizedTensorAlignment::UpdateLevel,
        InferenceEngine::details::LayerTransformation::QuantizedTensorAlignment::None,
        true,
        true,
        true);
}
}  // namespace LayerTestsUtils
