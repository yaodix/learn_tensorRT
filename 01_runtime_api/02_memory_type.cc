/* 
host\device内存分配和转移示例
cudaMalloc： Allocates memory on device
cudaMallocHost： Allocates page-locked memory on the host

new： Allocates pageable memory
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
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n",
             file, line, op, err_name, err_message);   
        return false;
    }
    return true;
}

int main(){

    int device_id = 0;
    checkRuntime(cudaSetDevice(device_id));  // 可以不设置

    float* memory_device = nullptr;
    checkRuntime(cudaMalloc(&memory_device, 100 * sizeof(float))); // pointer to device

    float* memory_host = new float[100];  // pageable memory
    memory_host[2] = 520.25;
    // 返回的地址是开辟的device地址，存放在memory_device
    checkRuntime(cudaMemcpy(memory_device, memory_host, sizeof(float) * 100, cudaMemcpyHostToDevice)); 

    float* memory_page_locked = nullptr;
    // 返回的地址是被开辟的pin memory的地址，存放在memory_page_locked(pinned memory)    
    checkRuntime(cudaMallocHost(&memory_page_locked, 100 * sizeof(float)));  // Allocates page-locked memory on the host
    checkRuntime(cudaMemcpy(memory_page_locked, memory_device, sizeof(float) * 100,
         cudaMemcpyDeviceToHost)); // 

    printf("%f\n", memory_page_locked[2]);
    checkRuntime(cudaFreeHost(memory_page_locked));
    delete [] memory_host;
    checkRuntime(cudaFree(memory_device)); 

    return 0;
}