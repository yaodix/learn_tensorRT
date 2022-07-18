
#include <string.h>
#include <iostream>
#include <stdio.h> // 因为要使用printf

// CUDA驱动头文件cuda.h
#include <cuda.h>

// 使用有参宏定义检查cuda driver是否被正常初始化, 并定位程序出错的文件名、行数和错误信息
// 宏定义中带do...while循环可保证程序的正确性
/*
#define checkDriver(op)    \
    do{                    \
        auto code = (op);  \
        if(code != CUresult::CUDA_SUCCESS){     \
            const char* err_name = nullptr;     \
            const char* err_message = nullptr;  \
            cuGetErrorName(code, &err_name);    \
            cuGetErrorString(code, &err_message);   \
            printf("%s:%d  %s failed. \n  code = %s, message = %s\n", __FILE__, __LINE__, #op, err_name, err_message);   \
            return -1;   \
        }                \
    }while(0)
*/

// 很明显，下面这种代码封装方式，更加的便于使用
//宏定义 #define <宏名>（<参数表>） <宏体>
#define checkDriver(op)  __check_cuda_driver((op), #op, __FILE__, __LINE__)

bool __check_cuda_driver(CUresult code, const char* op, const char* file, int line){

    if(code != CUresult::CUDA_SUCCESS){    
        const char* err_name = nullptr;    
        const char* err_message = nullptr;  
        cuGetErrorName(code, &err_name);    
        cuGetErrorString(code, &err_message);   
        printf("%s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);   
        return false;
    }
    return true;
}

int main() {
    /* 
    cuInit(int flags), 这里的flags目前必须给0;
        对于cuda的所有函数，必须先调用cuInit，否则其他API都会返回CUDA_ERROR_NOT_INITIALIZED
        https://docs.nvidia.com/cuda/archive/11.2.0/cuda-driver-api/group__CUDA__INITIALIZE.html
     */

    // 检查cuda driver的初始化
    // 实际调用的是__check_cuda_driver这个函数
      if (!checkDriver(cuInit(0))) {  // cuInit返回CUresult 类型：用于接收一些可能的错误代码
    // if (!checkDriver(cuInit(2))) {
        std::cout << "init fail\n";
        return -1;
    }
    
    /* 
    测试获取当前cuda驱动的版本
    显卡、CUDA、CUDA Toolkit

        1. 显卡驱动版本，比如：Driver Version: 460.84
        2. CUDA驱动版本：比如：CUDA Version: 11.2
        3. CUDA Toolkit版本：比如自行下载时选择的10.2、11.2等；这与前两个不是一回事, 
                CUDA Toolkit的每个版本都需要最低版本的CUDA驱动程序
        
        三者版本之间有依赖关系, 可参照https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
        nvidia-smi显示的是显卡驱动版本和此驱动最高支持的CUDA驱动版本
        
     */
    
    int driver_version = 0;
    if(!checkDriver(cuDriverGetVersion(&driver_version))){
        return -1;
    }
    printf("Driver version is %d\n", driver_version);

    // 测试获取当前设备信息
    char device_name[100]; // char 数组
    CUdevice device = 0;
    checkDriver(cuDeviceGetName(device_name, sizeof(device_name), device));
    printf("Device %d name is %s\n", device, device_name);
    return 0;
}