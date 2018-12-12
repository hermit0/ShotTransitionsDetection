/*用于从db文件中提取样本的特征*/
#ifndef EXTRACTDATAFROMDB_HPP_
#define EXTRACTDATAFROMDB_HPP_

#include <string>
#include <memory>

#include "caffe/util/db.hpp"
#include "caffe/proto/caffe.pb.h"

using std::string;
using std::shared_ptr;
namespace db = caffe::db;

//用于从db文件中提取单个图像样本的特征
class ExtractDataFromDB{
private:
    shared_ptr<db::DB> m_db;
    shared_ptr<db::Cursor> m_cursor;
public:
    ExtractDataFromDB(const string &db_file, const string &db_backend);
    ~ExtractDataFromDB(){}
    bool getRecord(caffe::Datum &data);//获得一个图像样本的特征
    bool getKey(string &key);
    void next();
    bool valid();   //当前的记录是否有效    
};
#endif