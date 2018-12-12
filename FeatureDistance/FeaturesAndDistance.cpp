#include <iostream>
#include <string>
#include <utility>
#include <fstream>
#include "CalculateDistance.hpp"
#include "ExtractDataFromDB.hpp"


using std::pair;
using std::vector;
using std::string;
//从db文件中获取采样的视频的特征，计算相邻帧之间的距离,获得相似度序列
//features_db:包含单个视频中所有帧图像的特征的db文件
//db_type: db文件的类型 leveldb, lmdb
//sampleRate: 采样率
//type: 距离度量的类型，目前有Cosine
//similarities: 相似度序列，每一项表示(帧序号，和下一个采样帧的相似度)

void getSimilaritiesSquence(const string &features_db, const string &db_type, int sampleRate, string type,
    vector<pair<int,float>> &similarities)
{
    ExtractDataFromDB extractor(features_db, db_type);
    caffe::Datum features[2];   //用于存放相邻两帧的特征，原始数据
    string keys[2];//用于存放相邻两帧的索引  
    float* featureVectors[2];//用于存放特征向量  
    int index = 0;  //当前计算的采样帧存放在features[index]中

    if(extractor.getKey(keys[index]))
    {
        extractor.getRecord(features[index]);
        //计算特征向量的维度
        const int datum_channels = features[index].channels();
        const int datum_height = features[index].height();
        const int datum_width = features[index].width();
        const int nums = datum_channels * datum_height *datum_width;
        featureVectors[0] = new float[nums];
        featureVectors[1] = new float[nums];

        for(int i = 0; i < nums; ++i)
            featureVectors[index][i] = features[index].float_data(i);
        
        shared_ptr<CalculateDistance<float>> calculator = CreateCalculatorFloat().create(type);

        //获得下一个采样帧
        int step = 0;
        while(step < sampleRate)
        {
            extractor.next();
            if(!extractor.valid())
                break;
            ++step;
        }
        similarities.clear();
        if(step == sampleRate)
        {
            int other = index ^ 1;
            while(extractor.getKey(keys[other]))
            {
                extractor.getRecord(features[other]);
                for(int i = 0; i < nums; ++i)
                    featureVectors[other][i] = features[other].float_data(i);
                float distance = calculator->calculate(featureVectors[index],featureVectors[other],nums);
                int frame_no = std::stoi(keys[index]);
                similarities.push_back(std::make_pair(frame_no,distance));
                index = other;
                other = index ^ 1;
                //获得下一个采样帧
                step = 0;
                while(step < sampleRate)
                {
                    extractor.next();
                    if(!extractor.valid())
                        break;
                    ++step;
                }
                if(step < sampleRate)
                    break;
            }
        }
        delete featureVectors[0];
        delete featureVectors[1];
        

    }
}