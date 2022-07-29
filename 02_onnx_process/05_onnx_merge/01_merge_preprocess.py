from turtle import forward
import torch
import numpy as np
import onnx
class Preprocess(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = torch.rand(1,1,1,3)  # 成员变量使节点更加清晰
        self.std = torch.rand(1,1,1,3)

    def forward(self,x):
        x = x.float()  # 转换成float32,原本是float64
        x = (x / 255.0 - self.mean) / self.std
        x = x.permute(0, 3, 1, 2)
        return x


pre = Preprocess()
torch.onnx.export(pre, (torch.zeros(1,640,640,3, dtype=torch.uint8),), "preprocess.onnx")

# 0. 先把pre_onnx的所有节点及其输入输出名称都加上前缀
# 1. pre_onnx模型的输出节点--> yolov5s第一个操作节点的输入
# 2. 把pre_onnx的node全部放到yolov5s的node中
# 3. 把pre_onnx模型的input作为yolov5s模型的input(copyfrom)

pre_onnx = onnx.load("preprocess.onnx")
model = onnx.load("yolov5s.onnx")

for n in pre_onnx.graph.node:
    n.name = f"pre/{n.name}"     # node 的大部分属性都是string，包括 name、input、output
    for i in range(len(n.input)):
        n.input[i] = f"pre/{n.input[i]}"
    for i in range(len(n.output)):
        n.output[i] = f"pre/{n.output[i]}" 
        # print(type(n.output[i]))  # class str


for n in model.graph.node:
    if n.name == "Conv_0":
        n.input[0] = "pre/" + pre_onnx.graph.output[0].name  # 通过name的修改进行节点对接

for n in pre_onnx.graph.node:
    model.graph.node.append(n)  # 加载模型末端应该怎么加？

input_name = "pre/" + pre_onnx.graph.input[0].name
model.graph.input[0].CopyFrom(pre_onnx.graph.input[0])
model.graph.input[0].name = input_name


# onnx.save(model, "preyolov5.onnx")

    