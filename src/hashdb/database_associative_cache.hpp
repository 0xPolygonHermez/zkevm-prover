#ifndef DATABASE_ASSOCIATIVE_CACHE_HPP
#define DATABASE_ASSOCIATIVE_CACHE_HPP
#include <vector>
#include "goldilocks_base_field.hpp"
#include <nlohmann/json.hpp>
#include <mutex>
#include "zklog.hpp"

using namespace std;
using json = nlohmann::json;

struct DatabaseAssociativeCacheRecord
{
    string remainingKey;
    uint64_t key[4];
    void *value;
    string leftChildKey;
    string rightChildKey;
};

class DatabaseAssociativeCache
{
protected:
    recursive_mutex mlock;

    int nKeyBits;
    int size;
    vector<DatabaseAssociativeCacheRecord *> buffer;
    uint32_t  *indices_;
    uint64_t  *buffer_;
    int buffer_pos;

    uint64_t attempts;
    uint64_t hits;
    string name;

    uint64_t indexMask;

    DatabaseAssociativeCache() : nKeyBits(0),
                                 attempts(0),
                                 hits(0)
    {
        buffer_ = NULL;
    };
    ~DatabaseAssociativeCache(){
        delete[] buffer_;
        delete[] indices_;
    };
    inline bool addKeyValue(const string &key, const void *value, const bool update, const string &leftChildkey, const string &rightChildKey); // returns true if cache is full
    bool addKeyValue(const uint64_t index, const string &remainingKey, const void *value, const bool update, const string &leftChildkey, const string &rightChildKey);
    inline bool findKey(const string &key, DatabaseAssociativeCacheRecord *&record);
    bool findKey(const uint64_t index, const string &remainingKey, DatabaseAssociativeCacheRecord *&record);
    inline void splitKey(const string &key, string &remainingKey, uint64_t &index);

public:

    bool addKeyValue(Goldilocks::Element (&key)[4], const vector<Goldilocks::Element> &value, const bool update = true);
    bool findKey(Goldilocks::Element (&key)[4], vector<Goldilocks::Element> &value);
    DatabaseAssociativeCache(int nKeyBits_, string name_) : attempts(0),
                                                            hits(0)
    {
        buffer_pos = 0;
        nKeyBits = nKeyBits_;
        if (nKeyBits % 4 != 0)
        {
            zklog.error("DatabaseAssociativeCache::DatabaseAssociativeCache() nKeyBits must be a multiple of 4");
            exit(1);
        }
        name = name_;
        size = 1 << nKeyBits;
        // buffer.assign(size,NULL); //rick: això ho hauré de treure
        indices_ = new uint32_t[size * 16];
        int buffer_size = 1<<16;
        buffer_ = new uint64_t[buffer_size];

        indexMask = 0;
        for (int i = 0; i < nKeyBits; i++)
        {
            indexMask = indexMask << 1;
            indexMask += 1;
        }
    };
    inline void postConstruct(int nKeyBits_, string name_);
    virtual DatabaseAssociativeCacheRecord *allocRecord(const string remainingKey, const void *value, const string &leftChildkey, const string &rightChildKey) = 0;
    virtual void freeRecord(DatabaseAssociativeCacheRecord *record) = 0;
    virtual void updateRecord(DatabaseAssociativeCacheRecord *record, const void *value, const string &leftChildkey, const string &rightChildKey) = 0;
    inline uint64_t getSize(void) { return size; };
    bool enabled() { return (nKeyBits > 0); };
};

class DatabaseMTAssociativeCache : public DatabaseAssociativeCache
{
public:
    ~DatabaseMTAssociativeCache(){};
    inline bool add(const string &key, const vector<Goldilocks::Element> &value, const bool update, const string &leftChildkey, const string &rightChildKey); // returns true if cache is full
    inline bool find(const string &key, vector<Goldilocks::Element> &value, string &leftChildkey, string &rightChildKey);
    DatabaseAssociativeCacheRecord *allocRecord(const string key, const void *value, const string &leftChildkey, const string &rightChildKey) override;
    inline void freeRecord(DatabaseAssociativeCacheRecord *record) override;
    inline void updateRecord(DatabaseAssociativeCacheRecord *record, const void *value, const string &leftChildkey, const string &rightChildKey) override;
};

class DatabaseProgramAssociativeCache : public DatabaseAssociativeCache
{
public:
    ~DatabaseProgramAssociativeCache(){};
    inline bool add(const string &key, const vector<uint8_t> &value, const bool update, const string &leftChildkey, const string &rightChildKey); // returns true if cache is full
    inline bool find(const string &key, vector<uint8_t> &value, string &leftChildkey, string &rightChildKey);
    DatabaseAssociativeCacheRecord *allocRecord(const string key, const void *value, const string &leftChildkey, const string &rightChildKey) override;
    inline void freeRecord(DatabaseAssociativeCacheRecord *record) override;
    inline void updateRecord(DatabaseAssociativeCacheRecord *record, const void *value, const string &leftChildkey, const string &rightChildKey) override;
};

// DatabaseAssociativeCache inlines
void DatabaseAssociativeCache::postConstruct(int nKeyBits_, string name_)
{
    buffer_pos = 0;
    nKeyBits = nKeyBits_;
    name = name_;
    size = 1 << nKeyBits;
    indices_ = new uint32_t[size];
    int buffer_size = 16*1<<16; //rick
    buffer_ = new uint64_t[buffer_size*16];
    for(int i=0; i<size; i++)
        indices_[i] = 0;

    attempts = 0;
    hits = 0;

    indexMask = 0;
    for (int i = 0; i < nKeyBits; i++)
    {
        indexMask = indexMask << 1;
        indexMask += 1;
    }
}
void DatabaseAssociativeCache::splitKey(const string &key, string &remainingKey, uint64_t &index)
{
    index = 0;
    int ndigits = nKeyBits / 4;
    remainingKey = key.substr(0, key.length() - ndigits);
    string cachekey = key.substr(key.length() - ndigits);
    for (int i = 0; i < ndigits; i++)
    {
        index = index << 4;
        index += (uint64_t)strtol(cachekey.substr(i, 1).c_str(), NULL, 16);
    }
    // std::cout<<" Cache: "<<key<<" "<<remainingKey<<" "<<cachekey<<" "<<index<<std::endl;
}
bool DatabaseAssociativeCache::addKeyValue(const string &key, const void *value, const bool update, const string &leftChildkey, const string &rightChildKey)
{
    string remainingKey;
    uint64_t index;
    splitKey(key, remainingKey, index);
    return addKeyValue(index, remainingKey, value, update, leftChildkey, rightChildKey);
}
bool DatabaseAssociativeCache::findKey(const string &key, DatabaseAssociativeCacheRecord *&record)
{
    string remainingKey;
    uint64_t index;
    splitKey(key, remainingKey, index);
    return findKey(index, remainingKey, record);
}

// DatabaseMTAssociativeCache inlines
bool DatabaseMTAssociativeCache::add(const string &key, const vector<Goldilocks::Element> &value, const bool update, const string &leftChildkey, const string &rightChildKey)
{
    lock_guard<recursive_mutex> guard(mlock); // rick: on es l'unlock?
    return addKeyValue(key, (const void *)&value, update, leftChildkey, rightChildKey);
}
bool DatabaseMTAssociativeCache::find(const string &key, vector<Goldilocks::Element> &value, string &leftChildkey, string &rightChildKey)
{
    lock_guard<recursive_mutex> guard(mlock);
    DatabaseAssociativeCacheRecord *record;
    bool found = findKey(key, record);
    if (found)
    {
        value = *((vector<Goldilocks::Element> *)record->value);
    }
    leftChildkey = record->leftChildKey;
    rightChildKey = record->rightChildKey;
    return found;
}
void DatabaseMTAssociativeCache::freeRecord(DatabaseAssociativeCacheRecord *record)
{
    delete (vector<Goldilocks::Element> *)(record->value);
    delete record;
}
void DatabaseMTAssociativeCache::updateRecord(DatabaseAssociativeCacheRecord *record, const void *value, const string &leftChildkey, const string &rightChildKey)
{
    *(vector<Goldilocks::Element> *)(record->value) = *(vector<Goldilocks::Element> *)(value);
    record->leftChildKey = leftChildkey;
    record->rightChildKey = rightChildKey;
}

// DatabaseProgramAssociativeCache inlines
bool DatabaseProgramAssociativeCache::add(const string &key, const vector<uint8_t> &value, const bool update, const string &leftChildkey, const string &rightChildKey)
{
    lock_guard<recursive_mutex> guard(mlock);
    return addKeyValue(key, (const void *)&value, update, leftChildkey, rightChildKey);
}
bool DatabaseProgramAssociativeCache::find(const string &key, vector<uint8_t> &value, string &leftChildkey, string &rightChildKey)
{
    lock_guard<recursive_mutex> guard(mlock);
    DatabaseAssociativeCacheRecord *record;
    bool found = findKey(key, record);
    if (found)
    {
        value = *((vector<uint8_t> *)record->value);
    }
    leftChildkey = record->leftChildKey;
    rightChildKey = record->rightChildKey;
    return found;
}
void DatabaseProgramAssociativeCache::freeRecord(DatabaseAssociativeCacheRecord *record)
{
    delete (vector<uint8_t> *)(record->value);
    delete record;
}
void DatabaseProgramAssociativeCache::updateRecord(DatabaseAssociativeCacheRecord *record, const void *value, const string &leftChildkey, const string &rightChildKey)
{
    *(vector<uint8_t> *)(record->value) = *(vector<uint8_t> *)(value);
    record->leftChildKey = leftChildkey;
    record->rightChildKey = rightChildKey;
}
#endif



