#ifndef DATABASE_ASSOCIATIVE_CACHE_HPP
#define DATABASE_ASSOCIATIVE_CACHE_1_HPP
#include <vector>
#include "goldilocks_base_field.hpp"
#include <nlohmann/json.hpp>
#include <mutex>
#include "zklog.hpp"

using namespace std;
using json = nlohmann::json;

struct DatabaseAssociativeCache1Record
{
    string remainingKey;
    uint64_t key[4];
    void *value;
    string leftChildKey;
    string rightChildKey;
};

class DatabaseAssociativeCache1
{
protected:
    recursive_mutex mlock;

    int nKeyBits;
    int size;
    vector<DatabaseAssociativeCache1Record *> buffer;
    uint64_t *buffer_;

    uint64_t attempts;
    uint64_t hits;
    string name;

    uint64_t indexMask;

    DatabaseAssociativeCache1() : nKeyBits(0),
                                 attempts(0),
                                 hits(0)
    {
        buffer_ = NULL;
    };
    ~DatabaseAssociativeCache1(){
        delete[] buffer_;
    };
    inline bool addKeyValue(const string &key, const void *value, const bool update, const string &leftChildkey, const string &rightChildKey); // returns true if cache is full
    bool addKeyValue(const uint64_t index, const string &remainingKey, const void *value, const bool update, const string &leftChildkey, const string &rightChildKey);
    inline bool findKey(const string &key, DatabaseAssociativeCache1Record *&record);
    bool findKey(const uint64_t index, const string &remainingKey, DatabaseAssociativeCache1Record *&record);
    inline void splitKey(const string &key, string &remainingKey, uint64_t &index);

public:

    bool addKeyValue(Goldilocks::Element (&key)[4], const vector<Goldilocks::Element> &value, const bool update);
    bool findKey(Goldilocks::Element (&key)[4], vector<Goldilocks::Element> &value);
    DatabaseAssociativeCache1(int nKeyBits_, string name_) : attempts(0),
                                                            hits(0)
    {
        nKeyBits = nKeyBits_;
        if (nKeyBits % 4 != 0)
        {
            zklog.error("DatabaseAssociativeCache1::DatabaseAssociativeCache1() nKeyBits must be a multiple of 4");
            exit(1);
        }
        name = name_;
        size = 1 << nKeyBits;
        // buffer.assign(size,NULL); //rick: això ho hauré de treure
        buffer_ = new uint64_t[size * 16];

        indexMask = 0;
        for (int i = 0; i < nKeyBits; i++)
        {
            indexMask = indexMask << 1;
            indexMask += 1;
        }
    };
    inline void postConstruct(int nKeyBits_, string name_);
    virtual DatabaseAssociativeCache1Record *allocRecord(const string remainingKey, const void *value, const string &leftChildkey, const string &rightChildKey) = 0;
    virtual void freeRecord(DatabaseAssociativeCache1Record *record) = 0;
    virtual void updateRecord(DatabaseAssociativeCache1Record *record, const void *value, const string &leftChildkey, const string &rightChildKey) = 0;
    inline uint64_t getSize(void) { return size; };
    bool enabled() { return (nKeyBits > 0); };
};

class DatabaseMTAssociativeCache1 : public DatabaseAssociativeCache1
{
public:
    ~DatabaseMTAssociativeCache1(){};
    inline bool add(const string &key, const vector<Goldilocks::Element> &value, const bool update, const string &leftChildkey, const string &rightChildKey); // returns true if cache is full
    inline bool find(const string &key, vector<Goldilocks::Element> &value, string &leftChildkey, string &rightChildKey);
    DatabaseAssociativeCache1Record *allocRecord(const string key, const void *value, const string &leftChildkey, const string &rightChildKey) override;
    inline void freeRecord(DatabaseAssociativeCache1Record *record) override;
    inline void updateRecord(DatabaseAssociativeCache1Record *record, const void *value, const string &leftChildkey, const string &rightChildKey) override;
};

class DatabaseProgramAssociativeCache1 : public DatabaseAssociativeCache1
{
public:
    ~DatabaseProgramAssociativeCache1(){};
    inline bool add(const string &key, const vector<uint8_t> &value, const bool update, const string &leftChildkey, const string &rightChildKey); // returns true if cache is full
    inline bool find(const string &key, vector<uint8_t> &value, string &leftChildkey, string &rightChildKey);
    DatabaseAssociativeCache1Record *allocRecord(const string key, const void *value, const string &leftChildkey, const string &rightChildKey) override;
    inline void freeRecord(DatabaseAssociativeCache1Record *record) override;
    inline void updateRecord(DatabaseAssociativeCache1Record *record, const void *value, const string &leftChildkey, const string &rightChildKey) override;
};

// DatabaseAssociativeCache1 inlines
void DatabaseAssociativeCache1::postConstruct(int nKeyBits_, string name_)
{
    nKeyBits = nKeyBits_;
    /*if (nKeyBits % 4 != 0) // rick: això ho hauré de treure
    {
        zklog.error("DatabaseAssociativeCache1::DatabaseAssociativeCache1() nKeyBits must be a multiple of 4");
        exit(1);
    }*/
    name = name_;
    size = 1 << nKeyBits;
    buffer.assign(size, NULL);
    buffer_ = new uint64_t[size * 16];

    attempts = 0;
    hits = 0;

    indexMask = 0;
    for (int i = 0; i < nKeyBits; i++)
    {
        indexMask = indexMask << 1;
        indexMask += 1;
    }
}
void DatabaseAssociativeCache1::splitKey(const string &key, string &remainingKey, uint64_t &index)
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
bool DatabaseAssociativeCache1::addKeyValue(const string &key, const void *value, const bool update, const string &leftChildkey, const string &rightChildKey)
{
    string remainingKey;
    uint64_t index;
    splitKey(key, remainingKey, index);
    return addKeyValue(index, remainingKey, value, update, leftChildkey, rightChildKey);
}
bool DatabaseAssociativeCache1::findKey(const string &key, DatabaseAssociativeCache1Record *&record)
{
    string remainingKey;
    uint64_t index;
    splitKey(key, remainingKey, index);
    return findKey(index, remainingKey, record);
}

// DatabaseMTAssociativeCache1 inlines
bool DatabaseMTAssociativeCache1::add(const string &key, const vector<Goldilocks::Element> &value, const bool update, const string &leftChildkey, const string &rightChildKey)
{
    lock_guard<recursive_mutex> guard(mlock); // rick: on es l'unlock?
    return addKeyValue(key, (const void *)&value, update, leftChildkey, rightChildKey);
}
bool DatabaseMTAssociativeCache1::find(const string &key, vector<Goldilocks::Element> &value, string &leftChildkey, string &rightChildKey)
{
    lock_guard<recursive_mutex> guard(mlock);
    DatabaseAssociativeCache1Record *record;
    bool found = findKey(key, record);
    if (found)
    {
        value = *((vector<Goldilocks::Element> *)record->value);
    }
    leftChildkey = record->leftChildKey;
    rightChildKey = record->rightChildKey;
    return found;
}
void DatabaseMTAssociativeCache1::freeRecord(DatabaseAssociativeCache1Record *record)
{
    delete (vector<Goldilocks::Element> *)(record->value);
    delete record;
}
void DatabaseMTAssociativeCache1::updateRecord(DatabaseAssociativeCache1Record *record, const void *value, const string &leftChildkey, const string &rightChildKey)
{
    *(vector<Goldilocks::Element> *)(record->value) = *(vector<Goldilocks::Element> *)(value);
    record->leftChildKey = leftChildkey;
    record->rightChildKey = rightChildKey;
}

// DatabaseProgramAssociativeCache1 inlines
bool DatabaseProgramAssociativeCache1::add(const string &key, const vector<uint8_t> &value, const bool update, const string &leftChildkey, const string &rightChildKey)
{
    lock_guard<recursive_mutex> guard(mlock);
    return addKeyValue(key, (const void *)&value, update, leftChildkey, rightChildKey);
}
bool DatabaseProgramAssociativeCache1::find(const string &key, vector<uint8_t> &value, string &leftChildkey, string &rightChildKey)
{
    lock_guard<recursive_mutex> guard(mlock);
    DatabaseAssociativeCache1Record *record;
    bool found = findKey(key, record);
    if (found)
    {
        value = *((vector<uint8_t> *)record->value);
    }
    leftChildkey = record->leftChildKey;
    rightChildKey = record->rightChildKey;
    return found;
}
void DatabaseProgramAssociativeCache1::freeRecord(DatabaseAssociativeCache1Record *record)
{
    delete (vector<uint8_t> *)(record->value);
    delete record;
}
void DatabaseProgramAssociativeCache1::updateRecord(DatabaseAssociativeCache1Record *record, const void *value, const string &leftChildkey, const string &rightChildKey)
{
    *(vector<uint8_t> *)(record->value) = *(vector<uint8_t> *)(value);
    record->leftChildKey = leftChildkey;
    record->rightChildKey = rightChildKey;
}
#endif



