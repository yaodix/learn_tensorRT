
/*
验证cuda runtimeAPI 懒加载特性。
1、 CUDA Runtime是封装了CUDA Driver的高级别更友好的API
2、 cudaruntime需要引入cudart这个so文件
3、上下文管理：
  3.1. 使用cuDevicePrimaryCtxRetain为每个设备设置context，不再手工管理context，并且不提供直接管理context的API
  3.2. 任何依赖CUcontext的API被调用时，会触发CUcontext的创建和对设备的绑定
    此后任何API调用时，会以设备id为基准，调取绑定好的CUcontext
    因此被称为懒加载模式，避免了手动维护CUcontext的麻烦
4、cuda的状态返回值，都是cudaError_t类型，通过check宏捕获状态并处理是一种通用方式
    官方案例采用宏，而非这里的函数加宏
    函数加宏具有更加好的便利性
*/
// CUDA运行时头文件
#include <cuda_runtime.h>  // 

// CUDA驱动头文件
#include <cuda.h>
#include <stdio.h>
#include <string.h>

#define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line){
    if(code != cudaSuccess){    
        const char* err_name = cudaGetErrorName(code);    
        const char* err_message = cudaGetErrorString(code);  
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);   
        return false;
    }
    return true;
}

int main(){

    CUcontext context = nullptr;
    cuCtxGetCurrent(&context);
    printf("Current context = %p，当前无context\n", context);

    // cuda runtime是以cuda为基准开发的运行时库
    // cuda runtime所使用的CUcontext是基于cuDevicePrimaryCtxRetain函数获取的
    // 即，cuDevicePrimaryCtxRetain会为每个设备关联一个context，通过cuDevicePrimaryCtxRetain函数可以获取到
    // 而context初始化的时机是懒加载模式，即当你调用一个runtime api时，会触发创建动作
    // 也因此，避免了cu驱动级别的init和destroy操作。使得api的调用更加容易
    int device_count = 0;
    checkRuntime(cudaGetDeviceCount(&device_count));
    printf("device_count = %d\n", device_count);

    // 取而代之，是使用setdevice来控制当前上下文，当你要使用不同设备时
    // 使用不同的device id
    // 注意，context是线程内作用的，其他线程不相关的, 一个线程一个context stack
    int device_id = 0;
    printf("set current device to : %d，这个API依赖CUcontext，触发创建并设置\n", device_id);
    checkRuntime(cudaSetDevice(device_id));

    // 注意，是由于set device函数是“第一个执行的需要context的函数”，所以他会执行cuDevicePrimaryCtxRetain
    // 并设置当前context，这一切都是默认执行的。注意：cudaGetDeviceCount是一个不需要context的函数
    // 你可以认为绝大部分runtime api都是需要context的，所以第一个执行的cuda runtime函数，会创建context并设置上下文
    cuCtxGetCurrent(&context);
    printf("SetDevice after, Current context = %p，获取当前context\n", context);

    int current_device = 0;
    checkRuntime(cudaGetDevice(&current_device));
    printf("current_device = %d\n", current_device);
    return 0;
}