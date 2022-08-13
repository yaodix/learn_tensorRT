/*
  重点是解决内存复用问题，使得内存分配、复制自动管理。
    在代码中声明MixMemory变量后，循环使用该变量不用重复分配和销毁内存(allocated_size <= to_alloc_size)
    第一次使用变量是才会分配内存
*/

#ifndef MEMORY_HPP
#define MEMORY_HPP

#include <stddef.h>

#define CURRENT_DEVICE_ID   -1

namespace TRT{

    class MixMemory {
    public:
        MixMemory(int device_id = CURRENT_DEVICE_ID);  // 设定内存存放的设备id
        MixMemory(void* cpu, size_t cpu_size, void* gpu, size_t gpu_size, int device_id = CURRENT_DEVICE_ID);
        virtual ~MixMemory();

        template<typename _T>
        _T* gpu(size_t size){ return (_T*)gpu(size * sizeof(_T)); }

        template<typename _T>
        _T* cpu(size_t size){ return (_T*)cpu(size * sizeof(_T)); };


        // 是否属于MixMemory自己分配的gpu/cpu
        inline bool owner_gpu() const{return owner_gpu_;}
        inline bool owner_cpu() const{return owner_cpu_;}

        inline size_t cpu_size() const{return cpu_size_;}
        inline size_t gpu_size() const{return gpu_size_;}
        inline int device_id() const{return device_id_;}

        inline void* gpu() const { return gpu_; }

        // Pinned Memory
        inline void* cpu() const { return cpu_; }

        template<typename _T>
        inline _T* gpu() const { return (_T*)gpu_; }

        // Pinned Memory
        template<typename _T>
        inline _T* cpu() const { return (_T*)cpu_; }
        
        // 引用外部分配好的内存
        void reference_data(void* cpu, size_t cpu_size, void* gpu, size_t gpu_size, int device_id = CURRENT_DEVICE_ID);

        void* gpu(size_t size);  // 返回或重新分配内存空间
        void* cpu(size_t size);
        void release_gpu();
        void release_cpu();
        void release_all();

    private:
        void* cpu_ = nullptr;
        size_t cpu_size_ = 0;
        bool owner_cpu_ = true;
        int device_id_ = 0;

        void* gpu_ = nullptr;
        size_t gpu_size_ = 0;
        bool owner_gpu_ = true;
    };
};

#endif // MEMORY_HPP