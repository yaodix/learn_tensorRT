#include "infer.hpp"
/*


*/
void infer_test(){
    auto infer = create_infer("trtfile"); // 创建及初始化 抖音网页短视频辅助讲解: 创建及初始化推理器 https://v.douyin.com/NfJvWdW/
    if(infer == nullptr){                       
        printf("Infer is nullptr.\n");          
        return;
    }
    // 将任务提交给推理器（推理器执行commit），同时推理器（infer）也等着获取（get）结果
    printf("commit msg = %s\n", infer->commit("msg").get().c_str()); 
}

int main(){

    infer_test();
    /* 
    多线程推理设计思路
        建议先阅读代码，若有疑问，可点击抖音短视频进行辅助讲解(建议1.5倍速观看)
        https://v.douyin.com/NfJ9kYy/
     */
    return 0;
}

