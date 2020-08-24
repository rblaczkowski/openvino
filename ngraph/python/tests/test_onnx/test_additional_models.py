# ******************************************************************************
# Copyright 2018-2020 Intel Corporation
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

import tests
from operator import itemgetter
from pathlib import Path
import os

from tests.test_onnx.utils import OpenVinoOnnxBackend
from tests.test_onnx.utils.model_importer import ModelImportRunner, ONNX_HOME


def _get_default_additional_models_dir():
    return os.path.join(ONNX_HOME, "")


MODELS_ROOT_DIR = tests.ADDITIONAL_MODELS_DIR
if len(MODELS_ROOT_DIR) == 0:
    MODELS_ROOT_DIR = _get_default_additional_models_dir()

zoo_models = []
# rglob doesn't work for symlinks, so models have to be physically somwhere inside "MODELS_ROOT_DIR"
for path in Path(MODELS_ROOT_DIR).rglob("*.onnx"):
    mdir, file = os.path.split(str(path))
    if not file.startswith("."):
        zoo_models.append({"model_name": path, "model_file": file, "dir": str(mdir)})

if len(zoo_models) > 0:
    sorted(zoo_models, key=itemgetter("model_name"))

    # Set backend device name to be used instead of hardcoded by ONNX BackendTest class ones.
    OpenVinoOnnxBackend.backend_name = tests.BACKEND_NAME

    # import all test cases at global scope to make them visible to pytest
    backend_test = ModelImportRunner(OpenVinoOnnxBackend, zoo_models, __name__)
    test_cases = backend_test.test_cases["OnnxBackendValidationModelImportTest"]

    test_cases_list = [
        "onnxrt/20191107/opset10/mlperf_ssd_mobilenet_300/ssd_mobilenet_v1_coco_2018_01_28.onnx_cpu",
        "onnxrt/20191107/opset10/mask_rcnn_keras/mask_rcnn_keras.onnx_cpu",

        "onnxrt/20191107/opset10/mlperf_mobilenet/mobilenet_v1_1.0_224.onnx_cpu",
        "onnxrt/20191107/opset10/mlperf_resnet/resnet50_v1.onnx_cpu",
        "onnxrt/20191107/opset10/tf_mobilenet_v2_1.0_224/model.onnx_cpu",
        "onnxrt/20191107/opset10/tf_mobilenet_v2_1.4_224/model.onnx_cpu",
        "onnxrt/20191107/opset10/tf_inception_v2/model.onnx_cpu",
        "onnxrt/20191107/opset10/tf_resnet_v1_50/model.onnx_cpu",
        "onnxrt/20191107/opset10/tf_inception_v3/model.onnx_cpu",
        "onnxrt/20191107/opset10/tf_resnet_v2_101/model.onnx_cpu",
        "onnxrt/20191107/opset10/tf_inception_v4/model.onnx_cpu",
        "onnxrt/20191107/opset10/tf_mobilenet_v1_1.0_224/model.onnx_cpu",
        "onnxrt/20191107/opset10/tf_nasnet_large/model.onnx_cpu",
        "onnxrt/20191107/opset11/tf_inception_v4/model.onnx_cpu",
        "onnxrt/20191107/opset11/tf_resnet_v2_50/model.onnx_cpu",
        "onnxrt/20191107/opset10/tf_resnet_v2_152/model.onnx_cpu",
        "onnxrt/20191107/opset10/tf_nasnet_mobile/model.onnx_cpu",
        "onnxrt/20191107/opset11/tf_resnet_v1_152/model.onnx_cpu",
        "onnxrt/20191107/opset10/tf_inception_resnet_v2/model.onnx_cpu",
        "onnxrt/20191107/opset10/mask_rcnn/mask_rcnn_R_50_FPN_1x.onnx_cpu",
        "onnxrt/20191107/opset10/tf_inception_v1/model.onnx_cpu",
        "onnxrt/20191107/opset10/tf_pnasnet_large/model.onnx_cpu",
        "onnxrt/20191107/opset11/tf_resnet_v2_101/model.onnx_cpu",
        "onnxrt/20191107/opset10/tf_resnet_v2_50/model.onnx_cpu",
        "onnxrt/20191107/opset11/tf_resnet_v1_50/model.onnx_cpu",
        "onnxrt/20191107/opset11/tf_pnasnet_large/model.onnx_cpu",
        "onnxrt/20191107/opset7/test_tiny_yolov2/model.onnx_cpu",
        "onnxrt/20191107/opset7/tf_inception_v1/model.onnx_cpu",
        "onnxrt/20191107/opset11/tf_inception_resnet_v2/model.onnx_cpu",
        "onnxrt/20191107/opset11/tf_inception_v1/model.onnx_cpu",
        "onnxrt/20191107/opset7/tf_inception_v2/model.onnx_cpu",
        "onnxrt/20191107/opset7/tf_mobilenet_v1_1.0_224/model.onnx_cpu",
        "onnxrt/20191107/opset7/tf_mobilenet_v2_1.0_224/model.onnx_cpu",
        "onnxrt/20191107/opset10/tf_resnet_v1_101/model.onnx_cpu",
        "onnxrt/20191107/opset7/tf_mobilenet_v2_1.4_224/model.onnx_cpu",
        "onnxrt/20191107/opset11/tf_nasnet_large/model.onnx_cpu",
        "onnxrt/20191107/opset7/tf_nasnet_mobile/model.onnx_cpu",
        "onnxrt/20191107/opset11/tf_resnet_v1_101/model.onnx_cpu",
        "onnxrt/20191107/opset11/tf_nasnet_mobile/model.onnx_cpu",
        "onnxrt/20191107/opset7/tf_inception_v3/model.onnx_cpu",
        "onnxrt/20191107/opset11/tf_inception_v2/model.onnx_cpu",
        "onnxrt/20191107/opset11/tf_resnet_v2_152/model.onnx_cpu",
        "onnxrt/20191107/opset7/tf_inception_resnet_v2/model.onnx_cpu",
        "onnxrt/20191107/opset7/tf_inception_v4/model.onnx_cpu",
        "onnxrt/20191107/opset7/tf_resnet_v1_101/model.onnx_cpu",
        "onnxrt/20191107/opset7/tf_resnet_v2_50/model.onnx_cpu",
        "onnxrt/20191107/opset11/tf_inception_v3/model.onnx_cpu",
        "onnxrt/20191107/opset7/tf_resnet_v2_101/model.onnx_cpu",
        "onnxrt/20191107/opset10/tf_resnet_v1_152/model.onnx_cpu",
        "onnxrt/20191107/opset7/tf_resnet_v1_152/model.onnx_cpu",
        "onnxrt/20191107/opset8/test_tiny_yolov2/model.onnx_cpu",
        "onnxrt/20191107/opset8/tf_inception_v1/model.onnx_cpu",
        "onnxrt/20191107/opset8/tf_inception_v2/model.onnx_cpu",
        "onnxrt/20191107/opset8/tf_mobilenet_v1_1.0_224/model.onnx_cpu",
        "onnxrt/20191107/opset10/faster_rcnn/faster_rcnn_R_50_FPN_1x.onnx_cpu",
        "onnxrt/20191107/opset8/tf_mobilenet_v2_1.0_224/model.onnx_cpu",
        "onnxrt/20191107/opset8/tf_mobilenet_v2_1.4_224/model.onnx_cpu",
        "onnxrt/20191107/opset8/tf_inception_v3/model.onnx_cpu",
        "onnxrt/20191107/opset8/tf_nasnet_mobile/model.onnx_cpu",
        "onnxrt/20191107/opset7/tf_resnet_v1_50/model.onnx_cpu",
        "onnxrt/20191107/opset8/tf_resnet_v1_50/model.onnx_cpu",
        "onnxrt/20191107/opset9/cgan/cgan.onnx_cpu",
        "onnxrt/20191107/opset8/tf_resnet_v2_50/model.onnx_cpu",
        "onnxrt/20191107/opset9/tf_inception_v1/model.onnx_cpu",
        "onnxrt/20191107/opset7/tf_resnet_v2_152/model.onnx_cpu",
        "onnxrt/20191107/opset8/tf_resnet_v1_101/model.onnx_cpu",
        "onnxrt/20191107/opset9/tf_inception_v2/model.onnx_cpu",
        "onnxrt/20191107/opset9/tf_mobilenet_v2_1.0_224/model.onnx_cpu",
        "onnxrt/20191107/opset8/tf_resnet_v2_101/model.onnx_cpu",
        "onnxrt/20191107/opset7/tf_nasnet_large/model.onnx_cpu",
        "onnxrt/20191107/opset8/tf_inception_v4/model.onnx_cpu",
        "onnxrt/20191107/opset9/tf_inception_v3/model.onnx_cpu",
        "onnxrt/20191107/opset9/tf_mobilenet_v1_1.0_224/model.onnx_cpu",
        "onnxrt/20191107/opset9/tf_nasnet_mobile/model.onnx_cpu",
        "onnxrt/20191107/opset8/tf_resnet_v1_152/model.onnx_cpu",
        "onnxrt/20191107/opset9/tf_inception_v4/model.onnx_cpu",
        "onnxrt/20191107/opset7/tf_pnasnet_large/model.onnx_cpu",
        "onnxrt/20191107/opset9/tf_resnet_v1_50/model.onnx_cpu",
        "onnxrt/20191107/opset9/tf_resnet_v2_50/model.onnx_cpu",
        "onnxrt/20191107/opset9/tf_resnet_v1_101/model.onnx_cpu",
        "onnxrt/20191107/opset8/tf_pnasnet_large/model.onnx_cpu",
        "onnxrt/20191107/opset9/tf_inception_resnet_v2/model.onnx_cpu",
        "onnxrt/20191107/opset8/tf_nasnet_large/model.onnx_cpu",
        "onnxrt/20191107/opset9/tf_resnet_v2_101/model.onnx_cpu",
        "onnxrt/20191107/opset8/tf_resnet_v2_152/model.onnx_cpu",
        "onnxrt/20191107/opset9/tf_mobilenet_v2_1.4_224/model.onnx_cpu",
        "onnxrt/20191107/opset9/tf_resnet_v1_152/model.onnx_cpu",
        "onnxrt/20191107/opset9/tf_pnasnet_large/model.onnx_cpu",
        "onnxrt/20191107/opset9/tf_nasnet_large/model.onnx_cpu",
        "onnxrt/20191107/opset9/tf_resnet_v2_152/model.onnx_cpu"]
    # for test_case in test_cases_list:
    #     import pytest
    #     pytest.mark.xfail(backend_test.get_testcase("ValidationModelExecution", test_case))
    # assert 1 == 9
    # pytest.mark.skip(backend_test._test_items["ValidationModelExecution"]["test_/root/.onnx/additional_models/onnxrt/20191107/opset10/mask_rcnn_keras/mask_rcnn_keras.onnx_cpu"].func)  # noqa
    # del test_cases

    # test_cases = backend_test.test_cases["OnnxBackendValidationModelExecutionTest"]
    # # assert 1 == 0
    # del test_cases
    # test_cases_list = [
    #     test_cases.
    #     ]

    globals().update(backend_test.enable_report().test_cases)
