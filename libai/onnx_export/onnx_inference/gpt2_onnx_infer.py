# coding=utf-8
# Copyright 2021 The OneFlow Authors. All rights reserved.
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


from collections import OrderedDict
from typing import List

import numpy as np
import onnxruntime as ort


class OnnxModel:
    def __init__(
        self,
        onnx_filename,
        providers: List[str] = None,
        ort_optimize: bool = True,
    ):
        ort_sess_opt = ort.SessionOptions()
        ort_sess_opt.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            if ort_optimize
            else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        )
        if providers is None:
            if ort.__version__ > "1.9.0":
                providers = [
                    "TensorrtExecutionProvider",
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ]
            else:
                providers = ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(
            onnx_filename, sess_options=ort_sess_opt, providers=providers
        )

    def forward(self, input_list):
        ipt_dict = OrderedDict()
        for idx, ipt in enumerate(self.sess.get_inputs()):
            ipt_dict[ipt.name] = input_list[idx]
        onnx_res = self.sess.run([], ipt_dict)
        return onnx_res


if __name__ == "__main__":
    onnx_model = OnnxModel("model.onnx")
    input_list = [
        np.ones((1, 5)).astype(np.int64).astype(np.int64),
    ]

    print(onnx_model.forward(input_list))
