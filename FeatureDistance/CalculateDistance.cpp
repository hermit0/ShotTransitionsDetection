#include "CalculateDistance.hpp"
#include <cmath>
#include "caffe/util/math_functions.hpp"

template <typename T>
T CosineDistance<T>::calculate(const T *a, const T *b, int n)
{
    T dot_product = caffe::caffe_cpu_dot(n,a,b);
    return dot_product / 
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

shared_ptr<CalculateDistance<float>> CreateCalculatorFloat::create(string type)
{
    if(type == "Cosine")
        return shared_ptr<CalculateDistance<float>>(new CosineDistance<float>());
    else{
        std::cerr << "Unknown distance type" << std::endl;
        exit(1);
    }
}