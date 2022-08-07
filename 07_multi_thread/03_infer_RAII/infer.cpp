
#include "infer.hpp"
#include <iostream>
#include <thread>
#include <vector>
#include <condition_variable>
#include <mutex>
#include <string>
#include <future>
#include <queue>
#include <functional>

// 封装接口类
using namespace std;

struct Job{
    shared_ptr<promise<string>> pro;
    string input;
};

class InferImpl : public Infer{
public:
    virtual ~InferImpl(){
        stop();
    }

    void stop(){
        if(running_){
            running_ = false;
            cv_.notify_one();  // 如果在worker的while中运行等待，可以运行wait()后，退出while
        }

        if(worker_thread_.joinable())
            worker_thread_.join();
    }

    bool startup(const string& file){

        file_ = file;
        running_ = true; // 启动后，运行状态设置为true

        // 线程传递promise的目的，是获得线程是否初始化成功的状态
        // 而在线程内做初始化，好处是，初始化跟释放在同一个线程内
        // 代码可读性好，资源管理方便
        promise<bool> pro;
        worker_thread_ = thread(&InferImpl::worker, this, std::ref(pro));
        /* 
            注意：这里thread 一构建好后，worker函数就开始执行了
            第一个参数是该线程要执行的worker函数，第二个参数是this指的是class InferImpl，
            第三个参数指的是传引用，因为我们在worker函数里要修改pro。  
         */
        return pro.get_future().get();  // worker修改完pro,立刻能get到
    }

    virtual shared_future<string> commit(const string& input) override{
        /* 
        建议先阅读代码，若有疑问，可点击抖音短视频进行辅助讲解(建议1.5倍速观看)
            commit 函数 https://v.douyin.com/NfJvHxm/
         */
        Job job;
        job.input = input;
        job.pro.reset(new promise<string>());

        shared_future<string> fut = job.pro->get_future();  // fut存放处理结果,shared_future允许复制
        {
            lock_guard<mutex> l(lock_);
            jobs_.emplace(std::move(job));
        }
        cv_.notify_one();
        return fut;
    }

    // 尽量保证模型在哪里分配就在哪里释放，哪里使用，这样能使得程序足够的简单，清晰
    void worker(promise<bool>& pro){

        // load model
        if(file_ != "trtfile"){
            // failed
            pro.set_value(false);
            printf("Load model failed: %s\n", file_.c_str());
            return;
        }

        // load success

        // 这里的promise用来负责确认infer初始化成功了, 此处set_value设置成功，pro.get_future().get()可以立刻获取值
        pro.set_value(true); 

        // this_thread::sleep_for(chrono::milliseconds(3000));
        // std::cout << "milliseconds 3000\n";

        int batch_size = 5;
        vector<Job> fetched_jobs;  // 存放一个batch的推理数据
        while(running_) {            
            {
                unique_lock<mutex> l(lock_);
                // 一直等着，cv_.wait(lock, predicate) // 如果 running不在运行状态 或者说 jobs_有东西 而且接收到了notify one的信号
                cv_.wait(l, [&](){return !running_ || !jobs_.empty();});  //  !running_作用 退出机制

                if(!running_) break; // 如果 不在运行 就直接结束循环
                
                // while (jobs_.size() < batch_size && !jobs_.empty()) {  // batch > 1
                //     fetched_jobs.emplace_back(std::move(jobs_.front())); 
                //     jobs_.pop();                                        
                // }
                for(int i = 0; i < batch_size && !jobs_.empty(); ++i){   // jobs_不为空的时候 // 如果生产频率高于消费频率，一般可以产生max_batch
                    fetched_jobs.emplace_back(std::move(jobs_.front())); // 就往里面fetched_jobs里塞东西
                    jobs_.pop();                                         // fetched_jobs塞进来一个，jobs_那边就要pop掉一个。（因为move）
                }
            }

            // 一次加载一批，并进行批处理
            // forward(fetched_jobs)
            // if (fetched_jobs.size() >= batch_size) // 可能会漏掉最后不足max_batchsize的任务
             { 
                std::cout << "batch size " << fetched_jobs.size() << std::endl;
                for(auto& job : fetched_jobs){
                    job.pro->set_value(job.input + "---processed");
                }
                fetched_jobs.clear();
            }
        }
        printf("Infer worker done.\n");
    }

private:
    atomic<bool> running_{false};
    string file_;
    thread worker_thread_;
    queue<Job> jobs_;
    mutex lock_;
    condition_variable cv_;
};

shared_ptr<Infer> create_infer(const string& file){
    /* 
        1. 避免外部单独执行模型加载功能
        2. 一般只执行一次
        3. forward函数中不必要判断加载模型成功（执行到这个步骤表示一定有存在的模型）
     */
    shared_ptr<InferImpl> instance(new InferImpl()); // 实例化一个推理器的实现类（inferImpl），以指针形式返回 
    if(!instance->startup(file)){                    // 推理器实现类实例(instance)启动。这里的file是engine file
        instance.reset();                            // 如果启动不成功就reset
    }
    std::cout << "startup success\n";
    return instance;    
}

