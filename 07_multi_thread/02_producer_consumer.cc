/*
生产消费者模型
  1. 队列
  2. 数据保护
  3. 条件变量
  4. 生产消费频率设置

*/

#include <stdio.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <string>
#include <iostream>
#include <memory>
#include <future>

using namespace std;

struct Job
{
  /* data */
  shared_ptr<promise<string>> pro;  // 避免多份拷贝
  string input;
};


deque<Job> jobs;  // not threa safety

mutex job_lock;
condition_variable cond;

void VideoCaputure() {
  int pic_id = 0;
  while(true) {
      Job j;
    {
      unique_lock<mutex> l(job_lock);
      std::string msg = "pic_id " + to_string(pic_id++);
      cout << "生产了一个新图片，" <<  msg << "，size is " << jobs.size() << endl;

      // 队列满，则不生成，等待需要生成的时候
      cond.wait(l, []() { 
        return jobs.size() < 5;  // false： 解锁互斥量，等待（阻塞）线程，true: 结束阻塞状态，加锁
        });
      j.pro.reset(new promise<string>());
      j.input = msg;
      jobs.push_back(j);
      // 异步模式
      // detection->push
      // face->push
      // feature->push


    }
    // 一次进行3个结果的回收，然后进行处理
    // 等待这个job处理完毕，那结果
    // .get()：实现等待(阻塞线程)，直到promise->set_value被执行了，有返回结果，这里的返回值就是res
    auto res = j.pro->get_future().get();  
    std::cout << "job " << j.input << "-> " << res << std::endl;

    this_thread::sleep_for(chrono::milliseconds(500));  // 生产变快
  }
}

void InferWork() {
  while (true) {
    if (!jobs.empty()) {
      {
        lock_guard<mutex> l(job_lock);
        auto job = jobs.front();
        jobs.pop_front();        
        // 消费掉就可以通知生成者
        cond.notify_one();
        cout<< "消费了一个新图片，" << job.input << endl;

        auto res = job.input + "---infer";
        job.pro->set_value(res);
      }
      this_thread::sleep_for(chrono::milliseconds(1000));

    }
    this_thread::yield();  // 交出时间片给其他线程, 有线程等待时使用
  }
  
}

int main() {

thread t1(VideoCaputure);
thread t2(InferWork);

t1.join();
t2.join();

}