#include <iostream>
#include <string>
#include <utility>
#include <fstream>
#include <iomanip>
#include <list>

#include <glog/logging.h>

#include "boost/algorithm/string.hpp"

#include "CalculateDistance.hpp"
#include "ExtractDataFromDB.hpp"


using std::pair;
using std::vector;
using std::string;

vector<int> merger_candidates(vector<vector<int>> &candidates_at_all_sampleRates);
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
        
        shared_ptr<CalculateDistance<float>> calculator = CreateCalculator<float>().create(type);

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

//candidate seletction
//算法1
//T = local_mean + a * local_sigma * (1 + ln(global_mean / local_mean))
//当d(i) > T 或者 d(i)比其相邻的大很多时，认为该帧是candidate
vector<int> filtering(vector<pair<int,float>> &distances, float sigma, int window_size)
{
    float window_sum = 0.0;
    //int start_frame = window_size - 1;
    size_t window_start = 0;
    size_t window_end = 2 * window_size - 2;
    
    vector<int> candidates;
    if(distances.empty())
        return candidates;

    //计算全局平均值
    float global_mean = 0.0;
    float sum = 0.0;
    for(size_t i = 0; i < distances.size();++i)
        sum += distances[i].second;
    global_mean = sum / distances.size();
    
    int frame_no = window_start + window_size - 1;
    for(size_t i = window_start; i <= window_end;++i)
        window_sum += distances[i].second;
    while(window_end < distances.size())
    {
        //计算local mean ,local standard deviation
        float local_mean = window_sum / (2 * window_size -1); 
        float local_d = 0.0;
        for(size_t i = window_start; i < window_end;++i)
        {
            local_d = (distances[i].second - local_mean) * (distances[i].second - local_mean);

        }
        local_d = std::sqrt(local_d / (2 * window_size - 2));
        //计算threshold
        float a = 0.7;
        float threshold = local_mean + a * local_d * (1 + std::log(global_mean / local_mean));
        if(distances[frame_no] .second > threshold)
            candidates.push_back(distances[frame_no].first);
        else{
            if((distances[frame_no].second > 3 * distances[frame_no - 1].second 
                || distances[frame_no].second > 3 * distances[frame_no + 1].second)
                && distances[frame_no].second > 0.8 * global_mean)
                candidates.push_back(distances[frame_no].first);
        }
        //滑动窗口
        ++frame_no;
        window_sum -= distances[window_start].second;
        ++window_end;
        if(window_end < distances.size())
            window_sum += distances[window_end].second;

    }
    return candidates;
    
}

int main(int argc, char **argv)
{
    ::google::InitGoogleLogging(argv[0]);
    const int num_required_args = 5;
    if(argc < num_required_args)
    {
        LOG(ERROR) << 
        "This program is used to calculate  frame distances\n"
        "Usage:FeaturesAndDistance features_db db_type sampleRates type\n"
        "features_db:包含单个视频中所有帧图像的特征的db文件\n"
        "db_type: db文件的类型 leveldb, lmdb\n"
        "sampleRate: 采样率,用逗号隔开的采样率序列\n"
        "type: 距离度量的类型，目前有Cosine\n";
        return 1;
    }
    int arg_pos = 0;
    string features_db(argv[++arg_pos]);
    string db_type(argv[++arg_pos]);
    string sampleRateString(argv[++arg_pos]);
    vector<string> temp;
    boost::split(temp,sampleRateString,boost::is_any_of(","));
    vector<int> sampleRates;
    for(size_t i = 0; i < temp.size();++i)
        sampleRates.push_back(std::stoi(temp[i]));
    string distance_type(argv[++arg_pos]);
    vector<vector<pair<int,float>>> all_distances(sampleRates.size());

    for(size_t i = 0; i < sampleRates.size();++i)
    {
        getSimilaritiesSquence(features_db,db_type,sampleRates[i],distance_type,all_distances[i]);
        // string output_file("distance");
        // output_file = output_file + std::to_string(i);
        // std::ofstream out(output_file);
        // if(!out.is_open())
        // {
        //     LOG(ERROR) << "cannot create the file " << output_file;
        //     return 1;
        // }
        // for(size_t j = 0; j < all_distances[i].size();++j)
        // {
        //     out << std::setw(10) << std::setfill('0') << all_distances[i][j].first << " " << all_distances[i][j].second
        //         << std::endl;
        // }
        // out.close();
    }
    vector<vector<int>> initial_candidates;
    //float t = 0.5;
    float sigma = 0.05;
    int window_size = 16;
    //进行初步的筛选
    for(size_t i = 0; i < sampleRates.size();++i)
    {
        vector<int> temp = filtering(all_distances[i], sigma, window_size);
        //打印以供调试
        // string candidate_file_name("candidates_at_sample_");
        // candidate_file_name.append(std::to_string(sampleRates[i]));
        // std::ofstream candidate_file(candidate_file_name);
        // if(candidate_file.is_open())
        // {
        //     for(auto frame_no : temp)
        //         candidate_file << frame_no << std::endl;
        //     candidate_file.close();
        // }
        initial_candidates.push_back(temp);

    }
    vector<int> all = merger_candidates(initial_candidates);
    string output_file_name(features_db);
    output_file_name.append("_candidates");
    std::ofstream all_file(output_file_name);
    if(all_file.is_open())
    {
        for(auto frame_no: all)
            all_file << frame_no << std::endl;
        all_file.close();
    }

    return 0;
}

//合并不同采样率得到的candidate
vector<int> merger_candidates(vector<vector<int>> &candidates_at_all_sampleRates)
{
    //测试，简单的合并所有的结果
    vector<int> result;
    if(candidates_at_all_sampleRates.empty())
        return result;
    std::list<int> temp(candidates_at_all_sampleRates[0].begin(),candidates_at_all_sampleRates[0].end());
    const int min_space = 5;    //不同采样率之间的候选帧之间的最小间隔
    for(size_t i = 1; i < candidates_at_all_sampleRates.size();++i)
    {
        auto it = temp.begin();
        auto prev_it = temp.end();
        for(size_t j = 0; j < candidates_at_all_sampleRates[i].size();++j)
        {
            while(it != temp.end() && *it < candidates_at_all_sampleRates[i][j])
            {
                prev_it = it;
                ++it;
            }
           
            //如果不同采样率的candidates相隔太近，则只保留低采样率的candidates
            if(it == temp.begin())
            {
                if(*it >= candidates_at_all_sampleRates[i][j] + min_space)
                    temp.insert(it, candidates_at_all_sampleRates[i][j]);
            }else if(it == temp.end())
            {
                if(candidates_at_all_sampleRates[i][j] >= *prev_it + min_space)
                    temp.insert(it, candidates_at_all_sampleRates[i][j]);
            }else
            {
                if(*it >= candidates_at_all_sampleRates[i][j] + min_space 
                    && candidates_at_all_sampleRates[i][j] >= *prev_it + min_space)
                    temp.insert(it, candidates_at_all_sampleRates[i][j]);
            }
            
        }
    }
    result.resize(temp.size());
    std::copy(temp.begin(),temp.end(),result.begin());
    return result;
}