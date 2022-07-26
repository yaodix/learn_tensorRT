/*
流的操作示例

*/
// CUDA运行时头文件
#include <cuda_runtime.h>

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

    int device_id = 0;
    checkRuntime(cudaSetDevice(device_id));

    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreate(&stream));  // 流是任务队列的抽象！！！

    // 在GPU上开辟空间
    float* memory_device = nullptr;
    checkRuntime(cudaMalloc(&memory_device, 100 * sizeof(float)));

    // 在CPU上开辟空间并且放数据进去，将数据复制到GPU
    float* memory_host = new float[100];
    memory_host[2] = 520.25;
    // 异步复制操作，发出指令立即返回，主线程不需要等待复制结束才继续，异步操作比同步多了一个参数stream
    checkRuntime(cudaMemcpyAsync(memory_device, memory_host, sizeof(float) * 100, cudaMemcpyHostToDevice, stream)); 

    // 在CPU上开辟pin memory,并将GPU上的数据复制回来 
    float* memory_page_locked = nullptr;
    checkRuntime(cudaMallocHost(&memory_page_locked, 100 * sizeof(float)));

    // 同样不用等待复制完成，但在流中排队
    checkRuntime(cudaMemcpyAsync(memory_page_locked, memory_device, sizeof(float) * 100,
            cudaMemcpyDeviceToHost, stream)); // 异步复制操作，主线程不需要等待复制结束才继续
    
    // 统一等待流队列中所有操作结束
    checkRuntime(cudaStreamSynchronize(stream));
    
    printf("%f\n", memory_page_locked[2]);
    
    // 释放内存
    checkRuntime(cudaFreeHost(memory_page_locked));
    checkRuntime(cudaFree(memory_device));
    checkRuntime(cudaStreamDestroy(stream));
    delete [] memory_host;
    return 0;
}