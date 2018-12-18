main.cpp中实现的程序可以边解压边提取特征并计算距离序列，最后执行过滤算法，输出candidate transition center.
"用法：calculateDistance pretained_net_param net_protofile blob_names video_file_list new_height new_width distance_type sampleRates output_dir [CPU/GPU] [device_id]"
        "pretrained_net_param:训练好的网络模型的参数\n"
        "net_protofile:网络的proto txt文件\n"
        "blob_names :要提取的特征对应的blob的名字,用逗号隔开\n"
        "video_file_list:包含所有视频文件路径的文本文件\n"
        "new_height:缩放后的图像高度\n"
        "new_width:缩放后的图像宽度\n"
        "distance_type: 距离度量的类型，目前有Cosine\n"
        "sampleRates:采样率序列，用逗号隔开\n"
        "output_dir:输出目录\n"
        "可选的[CPU/GPU] [device_id]";
