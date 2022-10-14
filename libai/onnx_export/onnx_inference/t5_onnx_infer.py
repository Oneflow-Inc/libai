import numpy as np
import onnxruntime as ort
from typing import List
from collections import OrderedDict

class OnnxModel:
    def __init__(
        self, 
        onnx_filename,
        providers: List[str] = None,
        ort_optimize: bool = True,
    ):
        ort_sess_opt = ort.SessionOptions()
        ort_sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED if \
            ort_optimize else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if providers is None:
            if ort.__version__ > "1.9.0":
                providers = [
                    "TensorrtExecutionProvider",
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ]
            else:
                providers = ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(onnx_filename, sess_options=ort_sess_opt, providers=providers)

    def forward(self, input_list):
        ipt_dict = OrderedDict()
        for idx, ipt in enumerate(self.sess.get_inputs()):
            ipt_dict[ipt.name] = input_list[idx]
        onnx_res = self.sess.run([], ipt_dict)
        return onnx_res

if __name__ == "__main__":
    onnx_model = OnnxModel("model.onnx")
    input_list = [
        np.ones((1, 5)).astype(np.int64),
        np.ones((1, 3)).astype(np.int64),
        np.ones((1, 5, 5)).astype(bool),
        np.ones((1, 3, 3)).astype(bool),
        np.ones((1, 3, 5)).astype(bool),
    ]
        
    print(onnx_model.forward(input_list))