#include <string>
#include <cstring>
#include <fstream>
#include <vector>
#include <iomanip>

#include <glog/logging.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio.hpp>
#include "boost/algorithm/string.hpp"

#include "caffe/proto/caffe.pb.h"
#include "caffe/blob.hpp"
#include "caffe/net.hpp"
#include "CalculateDistance.hpp"
#include "caffe/util/io.hpp"

using std::string;
using std::vector;
using std::pair;
using caffe::Caffe;
using caffe::InputParameter;
using caffe::Net;

int processVideo(const string &video_file, const string &pretrained_binary_proto, const string &feature_extraction_proto,
    const string &extract_feature_blob_names, const string &distance_type,int new_height, int new_width, const vector<int> &all_rates, const string &mode,int device_id);
//启动的主函数
int main(int argc, char **argv)
{
    ::google::InitGoogleLogging(argv[0]);
    const int num_required_args = 9;
    if(argc < num_required_args){
        LOG(ERROR) <<
        "This program is used to extract features for a list of videos\n"
        "用法：featureProcess pretained_net_param net_protofile blob_names video_file_list new_height new_width distance_type sampleRates [CPU/GPU] [device_id]"
        "pretrained_net_param:训练好的网络模型的参数\n"
        "net_protofile:网络的proto txt文件\n"
        "blob_names :要提取的特征对应的blob的名字,用逗号隔开\n"
        "video_file_list:包含所有视频文件路径的文本文件\n"
        "new_height:缩放后的图像高度\n"
        "new_width:缩放后的图像宽度\n"
        "distance_type: 距离度量的类型，目前有Cosine\n"
        "sampleRates:采样率序列，用逗号隔开\n"
        "可选的[CPU/GPU] [device_id]";

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
    int new_height = atoi(argv[++arg_pos]);
    int new_width = atoi(argv[++arg_pos]); 
    string distance_type(argv[++arg_pos]);

    //获得采样率序列
    string sampleRates(argv[++arg_pos]);
    vector<string> temp;
    boost::split(temp,sampleRates,boost::is_any_of(","));
    vector<int> all_rates;
    for(size_t i = 0; i < temp.size();++i)
    {
        int rate = std::stoi(temp[i]);
        CHECK_GE(rate, 1) << " the sample rate must >= 1";
        all_rates.push_back(rate);
    }
    std::sort(all_rates.begin(),all_rates.end());   //确保采样率是递增的

    //读取视频文件并依次处理单个视频
    std::ifstream videos_stream(contain_videos_file);
    if(videos_stream.is_open())
    {
        string video_name;
        while(videos_stream >> video_name)
        {
            LOG(ERROR) << "start  processing " << video_name;
            if(processVideo(video_name,pretrained_binary_proto, feature_extraction_proto,extract_feature_blob_names,distance_type,new_height,
                new_width, all_rates, mode, device_id))
                LOG(ERROR) << "cannot calculate distances sequence for video " << video_name;


        }
    }else
        LOG(ERROR) << "cannot open the file " << contain_videos_file;
    return 0;
}

//计算单个视频图像帧之间的距离序列
//成功返回1，失败返回0
//输入参数：
// video_file: 视频文件的路径
// pretrained_binary_proto: 训练好的网络模型
// feature_extraction_proto:网络的proto txt文件
// extract_feature_blob_names:要提取的特征名字，用逗号隔开
// distance_type:用于度量图像帧之间距离的类型，目前有Cosine
// new_height,new_width: 视频帧用作网络输入时应转换成的新大小
// all_rates: 对视频进行采样的采样率序列
// mode: CPU/GPU
// device_id: GPU的设备id
// 输出：包含距离序列的一系列文件，文件命令方式：视频名_特征名_采样率
int processVideo(const string &video_file, const string &pretrained_binary_proto, const string &feature_extraction_proto,
    const string &extract_feature_blob_names, const string &distance_type,int new_height, int new_width, const vector<int> &all_rates, const string &mode,int device_id)
{
    
    cv::VideoCapture cap;
    cv::Mat img, img_origin;

    cap.open(video_file);
    if(!cap.isOpened())
    {
        LOG(ERROR) << "Cannot open " << video_file;
        return 1;
    }
    int num_of_frames = cap.get(CV_CAP_PROP_FRAME_COUNT);

    //获得batch_size
    caffe::NetParameter net_param;
    if(!ReadProtoFromTextFile(feature_extraction_proto, &net_param))
    {
        LOG(ERROR) << "Failed to parse input text file as NetParameter: "
            << feature_extraction_proto;
        return 1;
    }
    int batch_size = 1;
    int width = 0, height = 0, channels = 0;
    for(int i  = 0; i < net_param.layer_size();++i)
    {
        if(net_param.layer(i).name() == "data" && net_param.layer(i).type() == "Input")
        {
            CHECK(net_param.layer(i).has_input_param())
                << "input layer data must have input_param";
            const InputParameter &input_param = net_param.layer(i).input_param();
            CHECK_GE(input_param.shape_size(),1) << "the input_param must specify the shape of input blob";
            const caffe::BlobShape &input_shape = input_param.shape(0);
            batch_size = input_shape.dim(0);
            channels = input_shape.dim(1);
            height = input_shape.dim(2);
            width = input_shape.dim(3);
            CHECK_EQ(height,new_height) << "new height must equal the height in input layer";
            CHECK_EQ(width,new_width) << "new width must equal the width in input layer";

        }
    }
    
    int num_mini_batches = num_of_frames / batch_size + ((num_of_frames % batch_size) > 0 ? 1:0);

    //初始化网络
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
    boost::shared_ptr<Net<float> > feature_extraction_net(
      new Net<float>(feature_extraction_proto, caffe::TEST));
    feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);
    std::vector<std::string> blob_names;
    boost::split(blob_names, extract_feature_blob_names, boost::is_any_of(","));
    size_t num_features = blob_names.size();
    for (size_t i = 0; i < num_features; i++) {
        CHECK(feature_extraction_net->has_blob(blob_names[i]))
            << "Unknown feature blob name " << blob_names[i]
            << " in the network " << feature_extraction_proto;
    }

    shared_ptr<CalculateDistance<float>> calculator = CreateCalculator<float>().create(distance_type);
    //all_distances[i]表示第i个特征的距离序列集合
    //all_distances[i][j]表示第i个特征在采样率j上的距离序列
    vector<vector<vector<pair<int,float>>>> all_distances(num_features,vector<vector<pair<int,float>>>(all_rates.size()));
    vector<vector<int>> to_compare(num_features);    //to_compare[i][j]第i特征在采样率j上要计算的帧的序号,初始均从0开始
    for(size_t i = 0; i < num_features; ++i)
    {
        for(size_t j = 0; j < all_rates.size();++j)
            to_compare[i].push_back(0);
    }
    vector<vector<shared_ptr<float>>> last_compare(num_features, vector<shared_ptr<float>>(all_rates.size()));    //存放不同特征在不同采样率上的前次滑动窗中的最后一帧的特征

    int window_begin = 0,window_end  = 0;
    for(int i = 0; i < num_mini_batches; ++i)
    {

        //更新输入blob
        boost::shared_ptr<caffe::Blob<float> > input_blob =
                feature_extraction_net->blob_by_name("data");
        float *top_data = input_blob->mutable_cpu_data();
        for(int j = 0; j < batch_size; ++j)
        {
            
            cap >> img_origin;
            if(!img_origin.empty())
            {
                cv::resize(img_origin,img,cv::Size(new_width,new_height));
                int offset = input_blob->offset(j);
                
                int top_index = offset;
                for(int h = 0; h < new_height; ++h)
                {
                    const uchar* ptr = img.ptr<uchar>(h);
                    int img_index = 0;
                    for(int w = 0; w < new_width;++w)
                        for(int c = 0; c < channels;++c)
                        {
                            top_index = offset + (c * height + h) * width + w;
                            top_data[top_index] = static_cast<float>(ptr[img_index++]);
                        }
                }
                ++window_end;
            }   
        }   //完成输入图像数据的设置
        LOG(ERROR) << "extract features of frame " << window_begin << " to frame " << window_end - 1;
        feature_extraction_net->Forward();//提取特征
        for(size_t feature_index = 0; feature_index < num_features;++feature_index)
        {
            const boost::shared_ptr<caffe::Blob<float> > feature_blob =
                feature_extraction_net->blob_by_name(blob_names[feature_index]);
            int dim_features = feature_blob->count() / batch_size;  //特征的维度
            const float *feature_blob_data = feature_blob->cpu_data();  //所有图像的特征数据
            //计算不同采样率上的距离
            for(size_t rate_index = 0; rate_index < all_rates.size(); ++rate_index )
            {
                const float *frame1 = nullptr;
                const float *frame2 = nullptr;
                while(to_compare[feature_index][rate_index] + all_rates[rate_index] < window_end)
                {
                    int frame1_no = to_compare[feature_index][rate_index];
                    if(frame1 == nullptr)
                    {
                        //该滑动窗口上的第一次比较
                        if(frame1_no >= window_begin)
                            frame1 = feature_blob_data + feature_blob->offset(frame1_no - window_begin);
                        else
                            frame1 = last_compare[feature_index][rate_index].get();
                    }
                    frame2 = feature_blob_data + feature_blob->offset(frame1_no + all_rates[rate_index] - window_begin);
                    float distance = calculator->calculate(frame1,frame2,dim_features);
                    
                    all_distances[feature_index][rate_index].push_back(std::make_pair(frame1_no,distance));
                    to_compare[feature_index][rate_index] += all_rates[rate_index];
                    frame1 = frame2;     
                }
                if(to_compare[feature_index][rate_index] >= window_begin)
                {
                    const float *last_frame = feature_blob_data + feature_blob->offset(to_compare[feature_index][rate_index] - window_begin);
                    //拷贝数据
                    float *feature_data = new float[dim_features];
                    for(int d = 0; d < dim_features; ++d)
                        feature_data[d] = last_frame[d];
                    last_compare[feature_index][rate_index].reset(feature_data);
                }
            }
                
        }//完成不同特征在不同采样率上的距离计算
        window_begin = window_end;
        window_end = window_begin;
    }
    //打印distance以调试
    for(size_t feature_index = 0; feature_index < num_features; ++feature_index)
    {
        string prefix(video_file + "_distance");
        prefix = prefix + blob_names[feature_index] + "_";
        for(size_t rate_index = 0; rate_index < all_rates.size();++rate_index)
        {
            string file_name(prefix + std::to_string(all_rates[rate_index]));
            std::ofstream of(file_name);
            if(!of.is_open())
            {
                LOG(ERROR) << "cannot create the file " << file_name;
                return 1;
            }
            for(size_t j = 0; j < all_distances[feature_index][rate_index].size();++j)
            {
                of << std::setw(10) << std::setfill('0') << all_distances[feature_index][rate_index][j].first << " " 
                    << all_distances[feature_index][rate_index][j].second << std::endl;
            }
            of.close();
        }
    }
    return 0;
}