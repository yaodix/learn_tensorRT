
from ast import Constant
import onnx
import numpy as np
import onnx.helper as helper
##################################################################

# 1. 读取不同模块的值
model = onnx.load("yolov5s.onnx")
# for item in model.graph.initializer:
#     print(item.name)

##################################################################

# 2. 改元素值
# for item in model.graph.node:
#     if item.op_type == "Constant":
#         if "362" in item.output:
#             t = item.attribute[0].t  
#             print(t)  # t.data_type: 7
#             print(type(t)) # 获取t类型是 onnx.onnx_ml_pb2.TensorProto,可在pb文件中查看该类型
#             print(np.frombuffer(t.raw_data, dtype=np.int64))
#             t.raw_data = np.array([100], dtype=np.int64).tobytes()

# onnx.save(model, "new.onnx")

##################################################################
# 3 替换节点
# for item in model.graph.node:
#     if item.name == "Reshape_231":
#         print(item)
#         newitem = helper.make_node("Reshape", ["366", "367"], ["368"], "Reshape_55xx")
#         item.CopyFrom(newitem)  # CopyFrom的使用


##################################################################
# 4 删除节点

# find_node_with_input  = lambda name: [item for item in model.graph.node if name in item.input][0]
# find_node_with_output  = lambda name: [item for item in model.graph.node if name in item.output][0]

# remove_node = []  # 避免一边删除一边遍历，造成迭代器失效
# for item in model.graph.node:
#     if item.name == "Transpose_235":
#         # 上一个节点的输出是当前节点的输入
#         prev = find_node_with_output(item.input[0])

#         # 下一个节点的输入是当前节点的输出
#         next = find_node_with_input(item.output[0])
#         next.input[0] = prev.output[0]
#         remove_node.append(item)

# for item in remove_node[::-1]:
    # model.graph.node.remove(item)


##################################################################
# 5 修改模型输入/输出
# print(model.graph.input)
# print(type(model.graph.input[0]))  # ValueInfoProto
# new_input = helper.make_tensor_value_info("images", 1, [3,3,640,640])  # [3,3,“height”,"width"], 字符串表示动态
# model.graph.input[0].CopyFrom(new_input)

# new_output = helper.make_tensor_value_info("output", 1, [3, 25200, 85])
# model.graph.output[0].CopyFrom(new_output)


##################################################################
# 6 加入preprocess到onnx


onnx.save(model, "new.onnx")