#ifndef EXTRACTFEATURES_HPP_
#define EXTRACTFEATURES_HPP_

#include <cstring>
#include <string>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/net.hpp"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"

using caffe::Caffe;
using caffe::Net;
using caffe::Blob;
using caffe::Datum;
using std::string;
namespace db = caffe::db;

/*
* 提取神经网络的某一层或若干层的输出作为特征，存放在文本文件中
*  使用方式： ExtractFeature-FC pretrained_net_param feature_extraction_proto_file
*                              extract_feature_blob_name save_feature_file_name num_mini_batches
*                              [CPU/GPU]  [device_id]
*  输出特征文件save_feature_file_name中每一行形式为 index: feature(用[]表示的向量)
*  包含可执行文件名字在内的所有命令行参数至少有6个
*/
template<typename Dtype> 
int feature_extract(int argc, char** argv);

/*
* 提取神经网络的某一层或若干层的输出作为特征，存放在db文件中）
*  使用方式： ExtractFeature-FC pretrained_net_param feature_extraction_proto_file
*                              extract_feature_blob_name save_feature_file_name  num_mini_batches
*                              leveldb/lmdb [CPU/GPU]  [device_id]
*  输出特征文件save_feature_file_name的每一个record对应单个样本的特征，其key是图像的索引（用10位数字表示），其value是对应输出blob的Datum对象
*  包含可执行文件名字在内的所有命令行参数至少有7个
*/
template<typename Dtype> 
int feature_extract_to_db(int argc, char** argv);

template<typename Dtype>
int feature_extract_to_db(const string &pretrained_net_param, const string &feature_extraction_proto_file, const string &extract_feature_blob_names,
    const string &save_feature_file_names, const int num_mini_batches, const int num_of_frames,const string &db_backend, const string mode = "CPU", const int device_id = 0);

// int feature_extract_to_db_float(const string &pretrained_net_param, const string &feature_extraction_proto_file, const string &extract_feature_blob_names,
//      const string &save_feature_file_names, const int num_mini_batches,const int num_of_frames, const string &db_backend, const string mode = "CPU", const int device_id = 0);

template<typename Dtype>
int feature_extract_to_db(const string &pretrained_net_param, const string &feature_extraction_proto_file, const string &extract_feature_blob_names,
    const string &save_feature_file_names, const int num_mini_batches, const int num_of_frames,const string &db_backend, const string mode, const int device_id)
{
    if(mode == "GPU")
    {
        LOG(ERROR)<< "Using GPU";
        LOG(ERROR) << "Using Device_id=" << device_id;
        Caffe::SetDevice(device_id);
        Caffe::set_mode(Caffe::GPU);
    }else
    {
        LOG(ERROR) << "Using CPU";
        Caffe::set_mode(Caffe::CPU);
    }
    boost::shared_ptr<Net<Dtype> > feature_extraction_net(
      new Net<Dtype>(feature_extraction_proto_file, caffe::TEST));
    feature_extraction_net->CopyTrainedLayersFrom(pretrained_net_param);

    std::vector<std::string> blob_names;
    boost::split(blob_names, extract_feature_blob_names, boost::is_any_of(","));

    std::vector<std::string> dataset_names;
    boost::split(dataset_names, save_feature_file_names,
               boost::is_any_of(","));
    CHECK_EQ(blob_names.size(), dataset_names.size()) <<
      " the number of blob names and dataset names must be equal";
    size_t num_features = blob_names.size();

    for (size_t i = 0; i < num_features; i++) {
        CHECK(feature_extraction_net->has_blob(blob_names[i]))
            << "Unknown feature blob name " << blob_names[i]
            << " in the network " << feature_extraction_proto_file;
    }

    std::vector<boost::shared_ptr<db::DB> > feature_dbs;
    std::vector<boost::shared_ptr<db::Transaction> > txns;

    for (size_t i = 0; i < num_features; ++i) {
        LOG(INFO)<< "Opening dataset " << dataset_names[i];
        boost::shared_ptr<db::DB> db(db::GetDB(db_backend));
        db->Open(dataset_names.at(i), db::NEW);
        feature_dbs.push_back(db);
        boost::shared_ptr<db::Transaction> txn(db->NewTransaction());
        txns.push_back(txn);
    }
    LOG(ERROR)<< "Extracting Features";
    Datum datum;
    std::vector<int> image_indices(num_features, 0);
    for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
        feature_extraction_net->Forward();
        for (int i = 0; i < num_features; ++i) {
            const boost::shared_ptr<Blob<Dtype> > feature_blob =
                feature_extraction_net->blob_by_name(blob_names[i]);
            int batch_size = feature_blob->num();
            int dim_features = feature_blob->count() / batch_size;
            const Dtype* feature_blob_data;
            for (int n = 0; n < batch_size; ++n) {
                datum.set_height(feature_blob->height());
                datum.set_width(feature_blob->width());
                datum.set_channels(feature_blob->channels());
                datum.clear_data();
                datum.clear_float_data();
                feature_blob_data = feature_blob->cpu_data() +
                    feature_blob->offset(n);
                for (int d = 0; d < dim_features; ++d) {
                    datum.add_float_data(feature_blob_data[d]);
                }
                if(image_indices[i] < num_of_frames)
                {
                    //是视频中的帧图像的特征
                    string key_str = caffe::format_int(image_indices[i], 10);

                    string out;
                    CHECK(datum.SerializeToString(&out));
                    txns.at(i)->Put(key_str, out);
                    ++image_indices[i];
                    if (image_indices[i] % 1000 == 0) {
                        txns.at(i)->Commit();
                        txns.at(i).reset(feature_dbs.at(i)->NewTransaction());
                        LOG(ERROR)<< "Extracted features of " << image_indices[i] <<
                        " query images for feature blob " << blob_names[i];
                    }
                }
                
            }  // for (int n = 0; n < batch_size; ++n)
        }
    }
    for (int i = 0; i < num_features; ++i) {
        if (image_indices[i] % 1000 != 0) {
            txns.at(i)->Commit();
        }
        LOG(ERROR)<< "Extracted features of " << image_indices[i] <<
            " query images for feature blob " << blob_names[i];
        feature_dbs.at(i)->Close();
    }

    LOG(ERROR)<< "Successfully extracted the features!";
    return 0;
        

}

template<typename Dtype> 
int feature_extract_to_db(int argc, char** argv)
{
    ::google::InitGoogleLogging(argv[0]);
    const int num_required_args = 7;
    if(argc < num_required_args){
        LOG(ERROR) <<
         "This program takes in a trained network and an input data layer, and then"
        " extract features of the input data produced by the net.\n"
        "usage: ExtractFeature pretrained_net_param feature_extraction_proto_file"
        " extract_feature_blob_name save_feature_file_name num_mini_batches"
        " [leveldb/lmdb] [CPU/GPU]  [device_id]\n"
        "Note: you can extract multiple features in one pass by specifying"
        " multiple feature blob names and dataset names separated by ','."
        " The names cannot contain white space characters and the number of blobs"
        " and datasets must be equal.";
        return 1;
    }
    
    int arg_pos = num_required_args;
    string mode = "CPU";
    int device_id = 0;
    if (argc > arg_pos && strcmp(argv[arg_pos], "GPU") == 0) {
        mode = "GPU";
        if (argc > arg_pos + 1) {
            device_id = atoi(argv[arg_pos + 1]);
            CHECK_GE(device_id, 0);
        }
        
    } 

    arg_pos = 0;  // the name of the executable
    std::string pretrained_binary_proto(argv[++arg_pos]);
    std::string feature_extraction_proto(argv[++arg_pos]);
    

    std::string extract_feature_blob_names(argv[++arg_pos]);
    

    std::string save_feature_dataset_names(argv[++arg_pos]);
    
    

    int num_mini_batches = atoi(argv[++arg_pos]);

    
    const char* db_type = argv[++arg_pos];
    
    return feature_extract_to_db<Dtype>(pretrained_binary_proto,feature_extraction_proto,extract_feature_blob_names,save_feature_dataset_names,
        num_mini_batches,db_type,mode,device_id);
    
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
#endif