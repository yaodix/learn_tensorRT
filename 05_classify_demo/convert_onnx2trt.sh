 # 转换模型倾向使用trtexec 而不是onnx2trt
 
 cd /home/yao/TensorRT-7.2.3.4/targets/x86_64-linux-gnu/bin/
./trtexec --onnx=/home/yao/workspace/learn_tensorRT/08_module_depoly/01_memory/workspace/classifier.onnx \
    --saveEngine=/home/yao/workspace/learn_tensorRT/08_module_depoly/01_memory/workspace/classifier.trt
