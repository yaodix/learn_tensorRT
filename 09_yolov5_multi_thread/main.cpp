
// tensorRT include
// 编译用的头文件
#include <NvInfer.h>

// cuda include
#include <cuda_runtime.h>

// system include
#include <stdio.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <functional>
#include <unistd.h>
#include <opencv2/opencv.hpp>

#include "simple-logger.hpp"
#include "yolov5.hpp"

using namespace std;

static const char* cocolabels[] = {
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};

static bool exists(const string& path){

#ifdef _WIN32
    return ::PathFileExistsA(path.c_str());
#else
    return access(path.c_str(), R_OK) == 0;
#endif
}

static void inference(){

    auto image = cv::imread("/home/yao/workspace/learn_tensorRT/09_yolov5_multi_thread/workspace/rq.jpg");
    auto yolov5 = YoloV5::create_infer("/home/yao/workspace/learn_tensorRT/06_yolov5_deploy/01_yolov5s/yolov5-6.0/yolov5s.trt");
    auto boxes = yolov5->commit(image).get();

    for(auto& box : boxes){
        cv::Scalar color(0, 255, 0);
        cv::rectangle(image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), color, 3);

        auto name      = cocolabels[box.class_label];
        auto caption   = cv::format("%s %.2f", name, box.confidence);
        int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::rectangle(image, cv::Point(box.left-3, box.top-33), cv::Point(box.left + text_width, box.top), color, -1);
        cv::putText(image, caption, cv::Point(box.left, box.top-5), 0, 1, cv::Scalar::all(0), 2, 16);
    }
    cv::imwrite("/home/yao/workspace/learn_tensorRT/09_yolov5_multi_thread/workspace/image-draw.jpg", image);
}

int main(){

    inference();
    return 0;
}