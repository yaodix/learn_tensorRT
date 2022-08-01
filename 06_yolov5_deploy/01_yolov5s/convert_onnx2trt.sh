 # 转换模型倾向使用trtexec 而不是onnx2trt
 
 cd /home/yao/TensorRT-7.2.3.4/targets/x86_64-linux-gnu/bin/
./trtexec --onnx=/home/yao/workspace/learn_tensorRT/06_yolov5_deploy/01_yolov5s/yolov5-6.0/yolov5s.onnx --saveEngine=/home/yao/workspace/learn_tensorRT/06_yolov5_deploy/01_yolov5s/yolov5-6.0/yolov5s.trt
