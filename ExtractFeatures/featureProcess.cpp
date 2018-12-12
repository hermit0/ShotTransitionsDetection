#include <string>
#include <fstream>

#include "boost/filesystem.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/videoio.hpp>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/common.hpp"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"

#include "ExtractFeatures.hpp"

using caffe::Datum;

using std::string;

namespace db = caffe::db;
using ::boost::filesystem::path;

int extractFeaturesForVideo(const string &video_file, const string &pretrained_binary_proto, const string &feature_extraction_proto,
    const string &extract_feature_blob_names, const string &db_backend, const int batch_size,const string mode, const int device_id,
    int new_height, int new_width);

//对视频序列进行处理，提取需要的特征
//用法：featureProcess pretained_net_param net_protofile blob_names video_file_list db_backend
//pretrained_net_param:训练好的网络模型的参数
//net_protofile:网络的proto txt文件
//blob_name :要提取的特征对应的blob的名字,用逗号分隔开
//video_file_list:包含所有视频文件路径的文本文件
int processAllVideos(int argc, char **argv)
{
    ::google::InitGoogleLogging(argv[0]);
    const int num_required_args = 9;
    if(argc < num_required_args){
        LOG(ERROR) <<
        "This program is used to extract features for a list of videos\n"
        "用法：featureProcess pretained_net_param net_protofile blob_names video_file_list db_backend batch_size new_height new_width [CPU/GPU] [device_id]"
        "pretrained_net_param:训练好的网络模型的参数\n"
        "net_protofile:网络的proto txt文件\n"
        "blob_names :要提取的特征对应的blob的名字,用逗号隔开\n"
        "video_file_list:包含所有视频文件路径的文本文件\n"
        "db_backend:leveldb还是lmdb\n"
        "batch_size:提取特征时使用的batch的大小\n"
        "new_height:缩放后的图像高度\n"
        "new_width:缩放后的图像宽度\n";
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
    arg_pos = 0;
    std::string pretrained_binary_proto(argv[++arg_pos]);
    std::string feature_extraction_proto(argv[++arg_pos]);
    std::string extract_feature_blob_names(argv[++arg_pos]);
    

    std::string contain_videos_file(argv[++arg_pos]);
    std::string db_backend(argv[++arg_pos]);
    int batch_size = atoi(argv[++arg_pos]);
    std::ifstream videos_stream(contain_videos_file);
    int new_height = atoi(argv[++arg_pos]);
    int new_width = atoi(argv[++arg_pos]); 
    string video_name;
    while(videos_stream >> video_name)
    {
        //处理单个视频
        if(extractFeaturesForVideo(video_name, pretrained_binary_proto, feature_extraction_proto,extract_feature_blob_names,db_backend,
            batch_size,mode,device_id,new_height,new_width))
            LOG(ERROR) << "cannot extract features for video " << video_name;
    }
    return 0;
}

//对单个视频进行处理，获得视频所有帧的特征文件
//成功返回0,失败返回1
int extractFeaturesForVideo(const string &video_file, const string &pretrained_binary_proto, const string &feature_extraction_proto,
    const string &extract_feature_blob_names, const string &db_backend, const int batch_size,const string mode, const int device_id,
    int new_height, int new_width)
{
    std::vector<std::string> blob_names;
    boost::split(blob_names,extract_feature_blob_names,boost::is_any_of(","));
    string file_names;
    string file_name;
    for(int i = 0; i < blob_names.size();++i)
    {
        if(i > 0)
            file_names.push_back(',');
        file_name.clear();
        file_name.append(video_file);
        file_name.push_back('_');
        file_name.append(blob_names[i]);
        file_names.append(file_name);
        file_names.append(db_backend);
    }
    
    cv::VideoCapture cap;
    cv::Mat img, img_origin;

    cap.open(video_file);
    if(!cap.isOpened())
    {
        LOG(ERROR) << "Cannot open " << video_file;
        return 1;
    }
    int num_of_frames = cap.get(CV_CAP_PROP_FRAME_COUNT);
    int height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    int width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    //const int batch_size = 10;  //批的大小
    int num_mini_batches = num_of_frames / batch_size + ((num_of_frames % batch_size) > 0 ? 1:0);
    int frame_no = 0;
    Datum datum;
    
    boost::shared_ptr<db::DB> videoDB(db::GetDB(db_backend));
    string db_name(video_file);
    db_name.append("_ImageDb");
    videoDB->Open(db_name,db::NEW);
    boost::shared_ptr<db::Transaction> txn(videoDB->NewTransaction());
    cap >> img_origin;
    datum.set_channels(3);
    if(new_height)
    {
        datum.set_height(new_height);
        datum.set_width(new_width);
    }else
    {
        datum.set_height(height);
        datum.set_width(width);
    }
    while(!img_origin.empty())
    {
        
        
        datum.clear_data();
        if(new_height)
            cv::resize(img_origin,img,cv::Size(new_width,new_height));
        else
            img_origin.copyTo(img);
        caffe::CVMatToDatum(img,&datum); //将图像保存到datum中
        //将datum保存到db中
        string key_str = caffe::format_int(frame_no++,10);
        string out;
        CHECK(datum.SerializeToString(&out));
        txn->Put(key_str,out);
        if(frame_no % 100 == 0)
        {
            txn->Commit();
            txn.reset(videoDB->NewTransaction());
            LOG(ERROR) << "Save frame " << frame_no - 100 << "to frame "
                << frame_no - 1 <<  "of " << video_file << " to " << db_name;
        }
        cap >> img_origin;
    }
    if(frame_no % 100 != 0)
        txn->Commit();
    LOG(ERROR) << "Save total" << frame_no << " frames of " << video_file << " to " << db_name;
    videoDB->Close();
    //修改net的数据层的源文件的路径
    caffe::NetParameter net_param;
    if(!ReadProtoFromTextFile(feature_extraction_proto, &net_param))
    {
        LOG(ERROR) << "Failed to parse input text file as NetParameter: "
            << feature_extraction_proto;
        return 1;
    }
    bool has_update = false;
        
    for(int i = 0; i < net_param.layer_size();++i)
    {
        if(net_param.layer(i).type() == "Data" && net_param.layer(i).has_data_param())
        {
            caffe::DataParameter *data_param = net_param.mutable_layer(i)->mutable_data_param();
            data_param->set_source(db_name);
            if(db_backend == "leveldb")
                data_param->set_backend(caffe::DataParameter_DB_LEVELDB);
            else
                data_param->set_backend(caffe::DataParameter_DB_LMDB);
            data_param->set_batch_size(batch_size);
            has_update = true;
        }
    }    
    if(has_update)
    {
        caffe::WriteProtoToTextFile(net_param,feature_extraction_proto);
        LOG(INFO) << "Wrote updated NetParameter text proto to " << db_name;
    }

    //提取该视频的特征,存放到db文件中
    // feature_extract_to_db<float>(pretrained_binary_proto, feature_extraction_proto, extract_feature_blob_names,file_names,num_mini_batches,
    //     db_backend,mode,device_id);

    //删除视频帧的db文件
    path file_path = db_name;
    if(!remove_all(file_path))
    {
        LOG(ERROR) << "cannot remove the temp output file " << db_name;
        return 1;
    }
    return 0;
    
}

int main(int argc, char **argv)
{
    return processAllVideos(argc, argv);
}
