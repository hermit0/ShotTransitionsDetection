#include <cstring>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <sstream>

#include "boost/algorithm/string.hpp"
#include <glog/logging.h>
//#define CPU_ONLY
#include  "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/format.hpp"


using caffe::Caffe;
using caffe::Net;
using caffe::Blob;

template<typename Dtype> 
int feature_extract(int argc, char** argv);
/*
* 提取全连接层的输出作为特征
*  使用方式： ExtractFeature-FC pretrained_net_param feature_extraction_proto_file
*                              extract_feature_blob_name save_feature_file_name num_mini_batches
*                              [CPU/GPU]  [device_id]
*  输出特征文件save_feature_file_name中每一行形式为 index: feature(用[]表示的向量)
*  包含可执行文件名字在内的所有命令行参数至少有6个
*/
int main(int argc, char** argv)
{
    return feature_extract<float>(argc, argv);
}

template<typename Dtype>
int feature_extract(int argc, char** argv)
{
    ::google::InitGoogleLogging(argv[0]);
    const int num_required_args = 6;
    if(argc < num_required_args){
        LOG(ERROR) <<
        "This program is used to extract features from fully connected layer"
        "usage: ExtractFeature-FC pretrained_net_param feature_extraction_proto_file"
        " extract_feature_blob_name save_feature_file_name num_mini_batches"
        " [CPU/GPU]  [device_id]\n";
        return 1;
    }
    int arg_pos = num_required_args;
    if(argc > arg_pos && std::strcmp(argv[arg_pos],"GPU") == 0)
    {
        LOG(ERROR) << "Using GPU";
        int device_id = 0;
        if(argc > arg_pos + 1){
            device_id = atoi(argv[arg_pos + 1]);
            CHECK_GE(device_id, 0);
        }
        LOG(ERROR) << "Using device_id = " << device_id;
        Caffe::SetDevice(device_id);
        Caffe::set_mode(Caffe::GPU);
    }else{
        LOG(ERROR) << "Using CPU";
        Caffe::set_mode(Caffe::CPU);
    }
    arg_pos = 0;
    
    std::string pretrained_binary_proto(argv[++arg_pos]);
    std::string feature_extraction_proto(argv[++arg_pos]);
    boost::shared_ptr<Net<Dtype> >  feature_extraction_net(
        new Net<Dtype>(feature_extraction_proto,caffe::TEST));
    feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);

    std::string extract_feature_blob_names(argv[++arg_pos]);
    std::vector<std::string> blob_names;
    boost::split(blob_names,extract_feature_blob_names,boost::is_any_of(","));

    std::string save_feature_file_names(argv[++arg_pos]);
    std::vector<std::string> file_names;
    boost::split(file_names,save_feature_file_names,boost::is_any_of(","));

    CHECK_EQ(blob_names.size(),file_names.size()) <<
        "the number of blob names and file names must be equal";
    
    size_t feature_nums = blob_names.size();
    
    //判断blob name 是否有效
    for(size_t i = 0; i < feature_nums; ++i){
        CHECK(feature_extraction_net->has_blob(blob_names[i]))
            << "Unknown feature blob name " << blob_names[i]
            << " in the network " << feature_extraction_proto;
    }

    int num_mini_batches = std::atoi(argv[++arg_pos]);

    std::vector<boost::shared_ptr<std::ofstream> > output_fstreams;
    for(size_t i = 0; i < feature_nums; ++i){
        boost::shared_ptr<std::ofstream> temp(new std::ofstream(file_names[i]));
        if(!temp->is_open())
        {
            LOG(ERROR) << "cann't open " << file_names[i];
            return 1;
        }
        output_fstreams.push_back(temp);
    }

    LOG(ERROR) << "Extracting features";
    std::vector<int> image_indices(feature_nums,0);
    for(int batch_index = 0; batch_index < num_mini_batches; ++batch_index){
        feature_extraction_net->Forward();
        for(size_t i = 0; i < feature_nums; ++i){
            const boost::shared_ptr<Blob<Dtype> > feature_blob = 
                feature_extraction_net->blob_by_name(blob_names[i]);
            int batch_size = feature_blob->num();
            int dim_features = feature_blob->count()/batch_size;
            for(int n = 0; n < batch_size;++n){
                const Dtype *feature_blob_data = feature_blob->cpu_data() +
                    feature_blob->offset(n);
                std::stringstream temp;
                temp << caffe::format_int(image_indices[i],10) << ":";
                temp << " [";
                for(int d = 0; d < dim_features; ++d)
                {
                    if(d != 0)
                        temp << ", ";
                    temp << feature_blob_data[d];
                    
                }
                temp << "]\n";
                std::string line;
                temp >> line;
                *(output_fstreams[i]) << line;
                ++image_indices[i];    
            }
        }
    }
    for(int i = 0; i < feature_nums; ++i)
        output_fstreams[i]->close();

    LOG(ERROR) << "Successfully extracted the features!";
    return 0;

}