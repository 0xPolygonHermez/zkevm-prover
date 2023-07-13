#ifndef DATABASE_ASSOCIATIVE_CACHE_HPP
#define DATABASE_ASSOCIATIVE_CACHE_HPP
#include <vector>
#include "goldilocks_base_field.hpp"
#include <nlohmann/json.hpp>
#include <mutex>
#include "zklog.hpp"

using namespace std;
using json = nlohmann::json;

struct DatabaseAssociativeCacheRecord {
    string remainingKey; 
    void* value;
    string leftChildKey;
    string rightChildKey;
    uint64_t size;
};

class DatabaseAssociativeCache
{
protected:
    recursive_mutex mlock;

    int  nKeyBits;
    int size;
    vector<DatabaseAssociativeCacheRecord*> buffer;
    
    uint64_t attempts;
    uint64_t hits;
    string name;

    DatabaseAssociativeCache() :
        nKeyBits(0),
        attempts(0),
        hits(0)
        {}; 
    ~DatabaseAssociativeCache(){};
    inline bool addKeyValue(const string &key, const void * value, const bool update, const string& leftChildkey,const string& rightChildKey); // returns true if cache is full
    bool addKeyValue(const uint64_t index, const string &remainingKey, const void * value, const bool update, const string& leftChildkey,const string& rightChildKey);
    inline bool findKey(const string &key, DatabaseAssociativeCacheRecord* &record); 
    bool findKey(const uint64_t index, const string &remainingKey, DatabaseAssociativeCacheRecord* &record);
    inline void splitKey(const string &key, string &remainingKey, uint64_t &index);
    
public:
    DatabaseAssociativeCache(int nKeyBits_, string name_) :
        attempts(0),
        hits(0)
        {
            nKeyBits = nKeyBits_;
            if(nKeyBits % 4 != 0)
            {
                zklog.error("DatabaseAssociativeCache::DatabaseAssociativeCache() nKeyBits must be a multiple of 4");
                exit(1);
            }
            name = name_;
            size = 1 << nKeyBits;
            buffer.assign(size,NULL);
        };
    inline void postConstruct(int nKeyBits_, string name_);
    virtual DatabaseAssociativeCacheRecord* allocRecord(const string remainingKey, const void * value, const string& leftChildkey,const string& rightChildKey) = 0;
    virtual void freeRecord(DatabaseAssociativeCacheRecord* record) = 0;
    virtual void updateRecord(DatabaseAssociativeCacheRecord* record, const void * value, const string& leftChildkey,const string& rightChildKey) = 0;
    inline uint64_t getSize(void) { return size; };
    bool enabled() {return (nKeyBits > 0);};
};

class DatabaseMTAssociativeCache : public DatabaseAssociativeCache
{
public:
    ~DatabaseMTAssociativeCache(){};
    inline bool add(const string &key, const vector<Goldilocks::Element> &value, const bool update, const string& leftChildkey,const string& rightChildKey); // returns true if cache is full
    inline bool find(const string &key, vector<Goldilocks::Element> &value, string& leftChildkey, string& rightChildKey);
    DatabaseAssociativeCacheRecord* allocRecord(const string key, const void * value, const string& leftChildkey,const string& rightChildKey) override;
    inline void freeRecord(DatabaseAssociativeCacheRecord* record) override;
    inline void updateRecord(DatabaseAssociativeCacheRecord* record, const void * value, const string& leftChildkey,const string& rightChildKey) override;
};

class DatabaseProgramAssociativeCache : public DatabaseAssociativeCache
{
public:  
    ~DatabaseProgramAssociativeCache(){};
    inline bool add(const string &key, const vector<uint8_t> &value, const bool update, const string& leftChildkey,const string& rightChildKey); // returns true if cache is full
    inline bool find(const string &key, vector<uint8_t> &value, string& leftChildkey, string& rightChildKey);
    DatabaseAssociativeCacheRecord* allocRecord(const string key, const void * value, const string& leftChildkey,const string& rightChildKey) override;
    inline void freeRecord(DatabaseAssociativeCacheRecord* record) override;
    inline void updateRecord(DatabaseAssociativeCacheRecord* record, const void * value, const string& leftChildkey,const string& rightChildKey) override;
};


// DatabaseAssociativeCache inlines 
void DatabaseAssociativeCache::postConstruct(int nKeyBits_, string name_){
        nKeyBits = nKeyBits_;
            if(nKeyBits % 4 != 0)
            {
                zklog.error("DatabaseAssociativeCache::DatabaseAssociativeCache() nKeyBits must be a multiple of 4");
                exit(1);
            }
            name = name_;
            size = 1 << nKeyBits;
            buffer.assign(size,NULL);
            attempts = 0;
            hits = 0;
}
void DatabaseAssociativeCache::splitKey(const string &key, string &remainingKey, uint64_t &index){
    index = 0;
    int ndigits = nKeyBits/4;
    remainingKey = key.substr(0,key.length() - ndigits);
    string cachekey = key.substr(key.length() - ndigits);
    for(int i = 0; i < ndigits; i++)
    {
        index = index << 4;
        index += (uint64_t)strtol(cachekey.substr(i,1).c_str(), NULL, 16);
    }
    //std::cout<<" Cache: "<<key<<" "<<remainingKey<<" "<<cachekey<<" "<<index<<std::endl;

}
bool DatabaseAssociativeCache::addKeyValue(const string &key, const void * value, const bool update, const string& leftChildkey, const string& rightChildKey){
    string remainingKey;
    uint64_t index;
    splitKey(key, remainingKey, index);
    return addKeyValue(index, remainingKey, value, update, leftChildkey, rightChildKey);
}
bool DatabaseAssociativeCache::findKey(const string &key, DatabaseAssociativeCacheRecord* &record){
    string remainingKey;
    uint64_t index;
    splitKey(key, remainingKey, index);
    return findKey(index, remainingKey, record);
}

// DatabaseMTAssociativeCache inlines
bool DatabaseMTAssociativeCache::add(const string &key, const vector<Goldilocks::Element> &value, const bool update, const string& leftChildkey,const string& rightChildKey)
{
    lock_guard<recursive_mutex> guard(mlock); //rick: on es l'unlock?
    return addKeyValue(key, (const void *)&value, update, leftChildkey, rightChildKey);
}
bool DatabaseMTAssociativeCache::find(const string &key, vector<Goldilocks::Element> &value, string& leftChildkey, string& rightChildKey)
{
    lock_guard<recursive_mutex> guard(mlock);
    DatabaseAssociativeCacheRecord* record;
    bool found = findKey(key, record);
    if (found) 
    {
        value = *((vector<Goldilocks::Element>*) record->value);
    }
    leftChildkey = record->leftChildKey;
    rightChildKey = record->rightChildKey;
    return found;
}
void DatabaseMTAssociativeCache::freeRecord(DatabaseAssociativeCacheRecord* record)
{
    delete (vector<Goldilocks::Element>*)(record->value);
    delete record;
}
void DatabaseMTAssociativeCache::updateRecord(DatabaseAssociativeCacheRecord* record, const void * value, const string& leftChildkey,const string& rightChildKey)
{
    *(vector<Goldilocks::Element>*)(record->value) = *(vector<Goldilocks::Element>*)(value);
    record->leftChildKey = leftChildkey;
    record->rightChildKey = rightChildKey;
}

// DatabaseProgramAssociativeCache inlines 
bool DatabaseProgramAssociativeCache::add(const string &key, const vector<uint8_t> &value, const bool update, const string& leftChildkey,const string& rightChildKey)
{
    lock_guard<recursive_mutex> guard(mlock);
    return addKeyValue(key, (const void *)&value, update,leftChildkey, rightChildKey);
}
bool DatabaseProgramAssociativeCache::find(const string &key, vector<uint8_t> &value, string& leftChildkey, string& rightChildKey)
{
    lock_guard<recursive_mutex> guard(mlock);
    DatabaseAssociativeCacheRecord* record;
    bool found = findKey(key, record);
    if (found) 
    {
        value = *((vector<uint8_t>*) record->value);
    }
    leftChildkey = record->leftChildKey;
    rightChildKey = record->rightChildKey;
    return found;
}
void DatabaseProgramAssociativeCache::freeRecord(DatabaseAssociativeCacheRecord* record)
{
    delete (vector<uint8_t>*)(record->value);
    delete record;
}
void DatabaseProgramAssociativeCache::updateRecord(DatabaseAssociativeCacheRecord* record, const void * value, const string& leftChildkey,const string& rightChildKey)
{
    *(vector<uint8_t>*)(record->value) = *(vector<uint8_t>*)(value);
    record->leftChildKey = leftChildkey;
    record->rightChildKey = rightChildKey;
}
#endif



