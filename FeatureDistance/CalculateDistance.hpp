/*
**计算两个向量a和b之间的距离。公共接口由CalculateDistance类提供。*
**具体类实例由CreateCalculator类负责生成        *
*/
#ifndef CALCULATEDISTANCE_HPP_
#define CALCULATEDISTANCE_HPP_

#include <string>
#include <memory>

using std::string;
using std::shared_ptr;

//计算两个向量之间的距离
template <typename T>
class CalculateDistance{
public:
    virtual ~CalculateDistance(){}
    virtual std::string type() = 0;
    virtual T calculate(const T *a, const T *b,int n )= 0;
};

//用于生成具体类的简单工厂
//这儿模板类的引用，有点问题，暂时使用下面的具体类来生成类实例
template <typename T>
class CreateCalculator{
public:
    shared_ptr<CalculateDistance<T>> create(string type);
};

class CreateCalculatorFloat{
public:
    shared_ptr<CalculateDistance<float>> create(string type);
};

//计算两个向量之间的余弦距离
template <typename T>
class CosineDistance :public CalculateDistance<T>{
public:
    virtual std::string type() {return "Cosine";}
    virtual T calculate(const T *a, const T *b, int n);    
};
#endif