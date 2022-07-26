
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
// 宏定义 #define <宏名>（<参数表>） <宏体>
// 即每次我执行checkDriver(my_func)的时候，我真正执行的是__check_cuda_driver 这个函数，
// 这函数传入的参数有my_func输出的结果（op），它的名字，当前所编译的文件和行数（方便之后报错时可以定位错误源）
// #op中的#作用是把参数（变量）变成字符串
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
    // if (!checkDriver(cuInit(2))) {  // 必须用0初始化
        std::cout << "init fail\n";
        return -1;
    }
    
    int driver_version = 0;
    if(!checkDriver(cuDriverGetVersion(&driver_version))){  // 若driver_version为11020指的是11.2
        return -1;
    }
    printf("Driver version is %d\n", driver_version);

    // 测试获取当前设备信息
    char device_name[100]; // char 数组, s数组名device_name当作指针
    CUdevice device = 0;
    // 获取设备名称、型号如：Tesla V100-SXM2-32GB
    checkDriver(cuDeviceGetName(device_name, sizeof(device_name), device));
    printf("Device %d name is %s\n", device, device_name);
    return 0;
}