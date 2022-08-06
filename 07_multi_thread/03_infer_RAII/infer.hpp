/*
    接口模式封装代码
        1.解决成员函数被外部使用的问题
        2. 解决头文件污染，如cuda相关

    原则：
        1. 头文件只包含重要的部分
        2. 外界不需要的尽量不要写在类中，如private成员变量。
        3. 不要在头文件中写namespace，可以在cpp文件中使用
*/
#ifndef INFER_HPP
#define INFER_HPP

#include <string>
#include <future>
#include <memory>

/////////////////////////////////////////////////////////////////////////////////////////
// 封装接口类
class Infer{
public:
    virtual std::shared_future<std::string> commit(const std::string& input) = 0;
};

std::shared_ptr<Infer> create_infer(const std::string& file);

#endif // INFER_HPP