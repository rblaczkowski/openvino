// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>

#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

typedef std::tuple<
    std::vector<std::vector<size_t>>,   // Input shapes
    InferenceEngine::Precision,         // Network Precision
    std::string,                        // Target Device
    std::map<std::string, std::string>  // Config
> concatMultiParams;

namespace LayerTestsDefinitions {

class ConcatMultiInput : public testing::WithParamInterface<concatMultiParams>,
    virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<concatMultiParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
