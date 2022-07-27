
from ast import Constant
import onnx
import numpy as np

# 1. 读取不同模块的值
model = onnx.load("./02_onnx_process/05_onnx_merge/yolov5s.onnx")
# for item in model.graph.initializer:
#     print(item.name)


# 2. 改元素值
for item in model.graph.node:
    if item.op_type == "Constant":
        if "362" in item.output:
            t = item.attribute[0].t  
            print(t)  # t.data_type: 7
            print(type(t)) # 获取t类型是 onnx.onnx_ml_pb2.TensorProto,可在pb文件中查看该类型
            print(np.frombuffer(t.raw_data, dtype=np.int64))
            t.raw_data = np.array([100], dtype=np.int64).tobytes()

# onnx.save(model, "new.onnx")

########
# 3 替换节点