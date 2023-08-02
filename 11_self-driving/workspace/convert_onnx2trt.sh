 # 转换模型倾向使用trtexec 而不是onnx2trt
 
 cd /opt/TensorRT-7.2.3.4/targets/x86_64-linux-gnu/bin/
./trtexec --onnx=/home/yao/workspace/learn_tensorRT/11_self-driving/workspace/ldrn_kitti_resnext101_pretrained_data_grad_256x512.onnx \
    --saveEngine=/home/yao/workspace/learn_tensorRT/11_self-driving/workspace/ldrn_kitti_resnext101_pretrained_data_grad_256x512.trt

./trtexec --onnx=/home/yao/workspace/learn_tensorRT/11_self-driving/workspace/road-segmentation-adas.onnx \
    --saveEngine=/home/yao/workspace/learn_tensorRT/11_self-driving/workspace/road-segmentation-adas.trt

./trtexec --onnx=/home/yao/workspace/learn_tensorRT/11_self-driving/workspace/ultra_fast_lane_detection_culane_288x800.onnx \
    --saveEngine=/home/yao/workspace/learn_tensorRT/11_self-driving/workspace/ultra_fast_lane_detection_culane_288x800.trt

./trtexec --onnx=/home/yao/workspace/learn_tensorRT/11_self-driving/workspace/yolov5s.onnx \
    --saveEngine=/home/yao/workspace/learn_tensorRT/11_self-driving/workspace/yolov5s.trt
