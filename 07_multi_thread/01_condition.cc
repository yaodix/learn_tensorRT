#include <thread>
#include <stdio.h>
#include <chrono>


using namespace std;
void worker() {
    printf("hello thread\n");
    this_thread::sleep_for(chrono::milliseconds(1000));
    printf("thread done\n");
}

class Infer {
public:
  Infer() {
    work_thread_ = thread(&Infer::_inferWorker, this);
  }

  ~Infer() {
    if ((work_thread_.joinable())) {
      work_thread_.join();
    }    
  }
private:
  void _inferWorker() {
    printf("in worker\n");
  }

  private:
    thread work_thread_;
};

int main() {
  Infer infer;
  return 0;
}