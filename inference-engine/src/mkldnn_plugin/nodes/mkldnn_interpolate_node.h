// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>

using namespace InferenceEngine;

namespace MKLDNNPlugin {
enum InterpolateLayoutType {
    planar,
    block,
    by_channel
};

enum InterpolateMode {
    nearest,
    linear,
    linear_onnx,
    cubic
};

enum InterpolateCoordTransMode {
    half_pixel,
    pytorch_half_pixel,
    asymmetric,
    tf_half_pixel_for_nn,
    align_corners
};

enum class InterpolateNearestMode {
    round_prefer_floor,
    round_prefer_ceil,
    floor,
    ceil,
    simple
};

struct jit_interpolate_config_params {
    InterpolateLayoutType layout;
    InterpolateMode mode;
    mkldnn::memory::data_type src_dt;
    mkldnn::memory::data_type dst_dt;
    int src_data_size;
    int dst_data_size;
    int indices_size;
    int IH, IW, OH, OW;
};

struct jit_interpolate_call_args {
    const void *src;
    const void *srcTR;
    const void *srcBL;
    const void *srcBR;
    const float *weight;
    const float *weightR;
    const float *weightT;
    const float *weightB;
    const int *index;
    void *dst;
    size_t work_amount;
    size_t oc_off;
};

struct jit_uni_interpolate_kernel {
    void (*ker_)(const jit_interpolate_call_args *);

    void operator()(const jit_interpolate_call_args *args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_interpolate_kernel(jit_interpolate_config_params jcp, const mkldnn_primitive_attr &attr) : ker_(nullptr), jcp_(jcp), attr_(attr) {}
    virtual ~jit_uni_interpolate_kernel() {}

    jit_interpolate_config_params jcp_;
    const mkldnn_primitive_attr &attr_;
};


class MKLDNNInterpolateNode : public MKLDNNNode {
public:
    MKLDNNInterpolateNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNInterpolateNode() override = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    bool created() const override;
    void execute(mkldnn::stream strm) override;
    bool canBeInPlace() const override {
        return false;
    }
    bool canFuse(const MKLDNNNodePtr& node) const;

private:
    // nearest neighbor
    void NNPlanar(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int ID, int IH, int IW,
                                          float fx, float fy, float fz, int OD, int OH, int OW);
    void NNCGathered(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int ID, int IH, int IW,
                                          float fx, float fy, float fz, int OD, int OH, int OW);
    void NNRef(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int ID, int IH, int IW,
                                          float fx, float fy, float fz, int OD, int OH, int OW);

    // onnx linear
    void linearOnnxPlanar(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int IH, int IW,
                                          float fx, float fy, int OH, int OW);
    void linearOnnxCGathered(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int IH, int IW,
                                          float fx, float fy, int OH, int OW);
    void linearOnnxRef(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int IH, int IW,
                                              float fx, float fy, int OH, int OW);

    // linear
    void linearInterpolation(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int ID, int IH, int IW,
                                          float fx, float fy, float fz, int OD, int OH, int OW, int kernel_width, bool antialias);

    // cubic
    std::vector<float> getCubicCoeffs(float mantissa, float a);
    void cubic(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int IH, int IW,
                                    float fx, float fy, int OH, int OW, float a);

    void buildTblNN(SizeVector& srcDimPad5d, SizeVector& dstDim5d, std::vector<float>& dataScales, InterpolateLayoutType layout);
    void buildTblLinearOnnx(SizeVector& srcDimPad5d, SizeVector& dstDim5d, std::vector<float>& dataScales, InterpolateLayoutType layout);
    void buidTblLinear(SizeVector& srcDimPad5d, SizeVector& dstDim5d, std::vector<float>& dataScales, int kernel_width, bool antialias);
    void buidTblCubic(SizeVector& srcDimPad5d, SizeVector& dstDim5d, std::vector<float>& dataScales, float cubicCoeff);

    void setPostOps(mkldnn::primitive_attr &attr, bool initWeights = false);

    inline float coordTransToInput(int outCoord, float scale, int inShape, int outShape);
    inline int nearestRound(float origin, bool isDownsample);
    float getValue(const uint8_t *base, size_t offset, InferenceEngine::Precision prec);
    void setValue(uint8_t *base, size_t offset, float value, InferenceEngine::Precision prec);

    SizeVector getPaddedInputShape();
    std::vector<float> getScales();

    const size_t DATA_ID = 0;
    const size_t TARGET_SHAPE_ID = 1;
    const size_t SCALES_ID = 2;
    const size_t AXES_ID = 3;
    const int LINEAR_KERNEL = 2;
    const int CUBIC_GRID_LEN = 4;

    InterpolateMode mode;
    InterpolateCoordTransMode coordTransMode = InterpolateCoordTransMode::half_pixel;
    bool antialias = false;
    std::vector<int> padBegin;
    std::vector<int> padEnd;
    bool hasPad = false;
    InterpolateNearestMode nearestMode = InterpolateNearestMode::round_prefer_floor;
    float cubeCoeff = -0.75;

    bool isAxesSpecified = false;
    // axes and scales from buffer, partical size.
    std::vector<int> axes;
    std::vector<float> scales;
    // target shape is dst dim, full size.
    SizeVector dstDim;
    std::string shapeInferMode;
    SizeVector srcDim;
    SizeVector srcDimPad;

    mkldnn::primitive_attr attr;
    std::vector<MKLDNNMemoryPtr> PostOpsIntBlobMemory;

    InferenceEngine::Precision inputPrec, outputPrec;
    size_t srcDataSize, dstDataSize;

    std::vector<int> indexTable;

    std::shared_ptr<jit_uni_interpolate_kernel> interpolateKernel;
};

}  // namespace MKLDNNPlugin
