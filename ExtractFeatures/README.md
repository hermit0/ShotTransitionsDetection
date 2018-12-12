本文件下的代码完成为视频提取深度特征的过程，执行过程中会对视频进行解压缩，并保存解压后的所有帧图像文件，然后为所有的帧图像提取特征，
并将提取到的特征保存到一个db文件中，再删除图像文件。因此运行过程中需要保证充足的硬盘空间，如果整个视频提取的特征的总大小特别大，不适合使用该程序。
具体的执行方式为 a.out pretrained_caffe_model net_proto_txt blob_names video_list_file db_backend batch_size new_height new_width [CPU/GPU] [device_id]
