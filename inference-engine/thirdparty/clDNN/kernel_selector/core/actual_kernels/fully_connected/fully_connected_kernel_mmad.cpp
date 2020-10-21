﻿// Copyright (c) 2016-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "fully_connected_kernel_mmad.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

ParamsKey FullyConnectedKernelMMAD::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);

    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);

    k.EnableInputWeightsType(WeightsType::INT8);

    k.EnableDifferentInputWeightsTypes();
    k.EnableDifferentTypes();

    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv32);
    k.EnableOutputLayout(DataLayout::bf);

    k.EnableBiasPerOutput();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableQuantization(QuantizationType::SYMMETRIC);
    return k;
}

bool FullyConnectedKernelMMAD::Validate(const Params& params, const optional_params& options) const {
    if (!Parent::Validate(params, options))
        return false;

    auto fc_params = static_cast<const fully_connected_params&>(params);
    auto input = fc_params.inputs[0];
    if (input.GetLayout() == DataLayout::bfyx &&
        (input.X().LogicalDimPadded() != 1 || input.Y().LogicalDimPadded() != 1 || input.Z().LogicalDimPadded() != 1)) {
        return false;
    }

    return true;
}

FullyConnectedKernelMMAD::FullyConnectedTuningData FullyConnectedKernelMMAD::SetTuningParams(const fully_connected_params& params) const {
    FullyConnectedTuningData tuning_data;

    const auto& input = params.inputs[0];

    size_t feature_blocks_count = input.GetLayout() == DataLayout::bfyx && input.Feature().v % 32 != 0 ?
                                  input.Feature().v / 32 : CeilDiv(input.Feature().v, 32);

    if (feature_blocks_count)
        while (feature_blocks_count % (tuning_data.slm_div_factor * 2) == 0 &&
               (tuning_data.slm_div_factor * 2 <= params.engineInfo.maxWorkGroupSize / tuning_data.sub_group_size))
            tuning_data.slm_div_factor *= 2;

    tuning_data.work_group_size = tuning_data.slm_div_factor * tuning_data.sub_group_size;

    return tuning_data;
}

FullyConnectedKernelMMAD::DispatchData FullyConnectedKernelMMAD::SetDefault(const fully_connected_params& params,
                                                                            int) const {
    FullyConnectedTuningData tuning_data = SetTuningParams(params);
    auto dispatchData = Parent::SetDefault(params);
    const auto& output = params.output;

    dispatchData.gws = { Align(output.Feature().v, tuning_data.sub_group_size) * tuning_data.slm_div_factor, output.Batch().v, 1 };
    dispatchData.lws = { tuning_data.work_group_size, 1, 1 };

    return dispatchData;
}

JitConstants FullyConnectedKernelMMAD::GetJitConstants(const fully_connected_params& params,
                                                       const DispatchData& dispatchData) const {
    FullyConnectedTuningData tuning_data = SetTuningParams(params);

    auto jit = Parent::GetJitConstants(params, dispatchData);

    auto& input = params.inputs[0];
    auto& weights = params.weights;

    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", tuning_data.sub_group_size));
    if (input.GetDims().size() == 5) {
        jit.AddConstant(MakeJitConstant("FILTER_GET_OFFSET(f)", "GET_FILTER_OS_IS_YX_ISA8_OSV8_ISV4_INDEX(FILTER, f, 0, 0, 0)"));
    } else {
        jit.AddConstant(MakeJitConstant("FILTER_GET_OFFSET(f)", "GET_FILTER_OS_IS_ZYX_ISA8_OSV8_ISV4_INDEX(FILTER, f, 0, 0, 0, 0)"));
    }

    Datatype input_packed_type = Datatype::INT32;
    Datatype filter_packed_type = Datatype::INT32;

    if (input.GetDType() == Datatype::UINT8) {
        input_packed_type = Datatype::UINT32;
    } else if (input.GetDType() == Datatype::INT8) {
        input_packed_type = Datatype::INT32;
    }

    if (weights.GetDType() == WeightsType::UINT8) {
        filter_packed_type = Datatype::UINT32;
    } else if (weights.GetDType() == WeightsType::INT8) {
        filter_packed_type = Datatype::INT32;
    }

    jit.Merge(MakeTypeJitConstants(input_packed_type, "INPUT_PACKED"));
    jit.Merge(MakeTypeJitConstants(filter_packed_type, "FILTER_PACKED"));

    auto filter_spatial_size = weights.X().v * weights.Y().v * weights.Z().v;
    int filter_spatial_pitch = 4 * 8 * 8;

    jit.AddConstant(MakeJitConstant("FILTER_SPATIAL_SIZE", filter_spatial_size));
    jit.AddConstant(MakeJitConstant("MMAD_FILTER_SPATIAL_PITCH", filter_spatial_pitch));
    jit.AddConstant(MakeJitConstant("MMAD_FILTER_FBLOCK_PITCH", filter_spatial_size * filter_spatial_pitch));

    size_t input_x_pitch = input.X().pitch;
    size_t input_y_pitch = input.Y().pitch;
    size_t input_z_pitch = input.Z().pitch;

    if (input.GetLayout() == DataLayout::bfyx) {
        jit.AddConstant(MakeJitConstant("MMAD_INPUT_FBLOCK_PITCH", 32));
    } else if (input.GetLayout() == DataLayout::b_fs_yx_fsv32 || input.GetLayout() == DataLayout::b_fs_zyx_fsv32) {
        input_x_pitch = 32;
        input_y_pitch *= 32;
        input_z_pitch *= 32;
        jit.AddConstant(MakeJitConstant("MMAD_INPUT_FBLOCK_PITCH", input.Feature().pitch * 32));
    }

    jit.AddConstant(MakeJitConstant("SLM_DIV_FACTOR", tuning_data.slm_div_factor));

    size_t feature_blocks_count;
    size_t temp_unroll_factor = 9, unroll_factor, full_unroll_factor;

    if (input.GetLayout() == DataLayout::bfyx && input.Feature().v % 32 != 0) {
        feature_blocks_count = input.Feature().v / 32;
        jit.AddConstant(MakeJitConstant("HAS_FEATURE_LEFTOVERS", true));
    } else {
        feature_blocks_count = CeilDiv(input.Feature().v, 32);
    }

    full_unroll_factor = feature_blocks_count / tuning_data.slm_div_factor;

    if (full_unroll_factor > 9) {
        while (full_unroll_factor % temp_unroll_factor)
            temp_unroll_factor--;
        unroll_factor = temp_unroll_factor;
    } else {
        unroll_factor = full_unroll_factor;
    }

    jit.AddConstant(MakeJitConstant("FEATURE_BLOCKS_COUNT", feature_blocks_count));
    jit.AddConstant(MakeJitConstant("UNROLL_FACTOR", unroll_factor));
    jit.AddConstant(MakeJitConstant("FULL_UNROLL_FACTOR", full_unroll_factor));
    jit.AddConstant(MakeJitConstant("WORK_GROUP_SIZE", tuning_data.work_group_size));

    jit.AddConstant(MakeJitConstant("MMAD_INPUT_SPATIAL_PITCH", input_x_pitch));
    jit.AddConstant(MakeJitConstant("MMAD_INPUT_X_PITCH", input_x_pitch));
    jit.AddConstant(MakeJitConstant("MMAD_INPUT_Y_PITCH", input_y_pitch));
    jit.AddConstant(MakeJitConstant("MMAD_INPUT_Z_PITCH", input_z_pitch));

    bool split_spatial = input.X().pad.Total() != 0 || input.Y().pad.Total() != 0 || input.Z().pad.Total() != 0;
    bool spatial_major = DataTensor::Channelndex(input.GetLayout(), Tensor::DataChannelName::X) <
                         DataTensor::Channelndex(input.GetLayout(), Tensor::DataChannelName::FEATURE);

    jit.AddConstant(MakeJitConstant("SPLIT_SPATIAL", split_spatial));
    jit.AddConstant(MakeJitConstant("SPATIAL_MAJOR", spatial_major));

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        FusedOpsConfiguration conf = { "", {"batch", "feature", "0", "0"}, "dequantized", input_dt, 1 };
        jit.Merge(MakeFusedOpsJitConstants(params, { conf }));
    }

    return jit;
}

KernelsData FullyConnectedKernelMMAD::GetKernelsData(const Params& params, const optional_params& options) const {
    auto fc_params = static_cast<const fully_connected_params&>(params);
    auto& input = fc_params.inputs[0];

    auto w_layout = WeightsLayout::os_is_yx_isa8_osv8_isv4;
    if (input.GetDims().size() == 5) {
        w_layout = WeightsLayout::os_is_zyx_isa8_osv8_isv4;
    }

    KernelsData res = {};
    for (size_t i = 0; i < autoTuneOptions.size(); i++) {
        KernelsData kd = GetTunedKernelsDataByIndex(params,
                                                    options,
                                                    input.GetLayout(),
                                                    w_layout,
                                                    FORCE_PRIORITY_7,
                                                    static_cast<int>(i));
        if (!kd.empty()) {
            res.emplace_back(kd[0]);
        }
    }
    return res;
}
}  // namespace kernel_selector
