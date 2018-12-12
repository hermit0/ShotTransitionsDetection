#include "ExtractDataFromDB.hpp"

ExtractDataFromDB::ExtractDataFromDB(const string& db_file,const string & db_backend)
{
    m_db.reset(db::GetDB(db_backend));
    m_db->Open(db_file,db::READ);
    m_cursor.reset(m_db->NewCursor());

}

//获得单个样本的特征，成功返回true
bool ExtractDataFromDB::getRecord(caffe::Datum &datum)
{
    if(m_cursor->valid()){

        datum.ParseFromString(m_cursor->value());
        return true;
    }else
        return false;
}

void ExtractDataFromDB::next()
{
    if(m_cursor->valid())
        m_cursor->Next();
}

bool ExtractDataFromDB::valid()
{
    return m_cursor->valid();
}

bool ExtractDataFromDB::getKey(string &key)
{
    if(m_cursor->valid()){

        key = m_cursor->key();
        return true;
    }else
        return false;
}