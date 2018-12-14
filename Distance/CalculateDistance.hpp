/*
**计算两个向量a和b之间的距离。公共接口由CalculateDistance类提供。*
**具体类实例由CreateCalculator类负责生成        *
*/
#ifndef CALCULATEDISTANCE_HPP_
#define CALCULATEDISTANCE_HPP_

#include <string>
#include <memory>
#include <cmath>
#include "caffe/util/math_functions.hpp"

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


template <typename T>
T CosineDistance<T>::calculate(const T *a, const T *b, int n)
{
    T dot_product = caffe::caffe_cpu_dot(n,a,b);
    return 1 - dot_product / 
            (std::sqrt(caffe::caffe_cpu_dot(n,a,a)) *
            std::sqrt(caffe::caffe_cpu_dot(n,b,b)));
}

template <typename T>
shared_ptr<CalculateDistance<T>> CreateCalculator<T>::create(string type)
{
    if(type == "Cosine")
        return shared_ptr<CalculateDistance<T>>(new CosineDistance<T>());
    else{
        std::cerr << "Unknown distance type" << std::endl;
        exit(1);
    }
}
#endif