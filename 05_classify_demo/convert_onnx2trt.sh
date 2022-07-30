 # 转换模型倾向使用trtexec 而不是onnx2trt
 
 cd /home/yao/TensorRT-7.2.3.4/targets/x86_64-linux-gnu/bin/
./trtexec --onnx=/home/yao/workspace/learn_tensorRT/05_classify_demo/workspace/classifier.onnx 
    --saveEngine=/home/yao/workspace/learn_tensorRT/05_classify_demo/workspace/classifier.trt
