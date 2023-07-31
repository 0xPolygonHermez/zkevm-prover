#ifndef DATABASE_CACHE_HPP
#define DATABASE_CACHE_HPP

#include <vector>
#include "goldilocks_base_field.hpp"
#include <nlohmann/json.hpp>
#include <mutex>
#include "zklog.hpp"

using namespace std;
using json = nlohmann::json;

struct DatabaseCacheRecord {
    string key; // TODO: use *char instead? it uses less memory than a string, but we need to convert
    void* value;
    DatabaseCacheRecord* next;
    DatabaseCacheRecord* prev;
    uint64_t size;
};

class DatabaseCache
{
protected:
    recursive_mutex mlock;
    uint64_t maxSize;
    uint64_t currentSize;
    unordered_map<string, DatabaseCacheRecord*> cacheMap;
    DatabaseCacheRecord * head;
    DatabaseCacheRecord * last;
    uint64_t attempts;
    uint64_t hits;
    string name;

    DatabaseCache() :
        maxSize(0),
        currentSize(0),
        head(NULL),
        last(NULL),
        attempts(0),
        hits(0)
        {};
    ~DatabaseCache();
    bool addKeyValue(const string &key, const void * value, const bool update); // returns true if cache is full
    bool findKey(const string &key, DatabaseCacheRecord* &record);

public:
    virtual DatabaseCacheRecord* allocRecord(const string key, const void * value) = 0;
    virtual void freeRecord(DatabaseCacheRecord* record) = 0;
    virtual void updateRecord(DatabaseCacheRecord* record, const void * value) = 0;

public:
    uint64_t getMaxSize(void) { return maxSize; };
    uint64_t getCurrentSize(void) { return currentSize; };
    bool enabled() {return (maxSize > 0);};
    void setMaxSize(int64_t size) { maxSize = size; }; // size is in bytes, 0 = no cache
    void setName(const char * pChar) { name = pChar; };
    void print(bool printContent);
    void clear(void);
};

class DatabaseMTCache : public DatabaseCache
{
public:
    ~DatabaseMTCache();
    bool add(const string &key, const vector<Goldilocks::Element> &value, const bool update); // returns true if cache is full
    bool find(const string &key, vector<Goldilocks::Element> &value);
    DatabaseCacheRecord* allocRecord(const string key, const void * value) override;
    void freeRecord(DatabaseCacheRecord* record) override;
    void updateRecord(DatabaseCacheRecord* record, const void * value) override;
};

class DatabaseProgramCache : public DatabaseCache
{
public:  
    ~DatabaseProgramCache();
    bool add(const string &key, const vector<uint8_t> &value, const bool update); // returns true if cache is full
    bool find(const string &key, vector<uint8_t> &value);
    DatabaseCacheRecord* allocRecord(const string key, const void * value) override;
    void freeRecord(DatabaseCacheRecord* record) override;
    void updateRecord(DatabaseCacheRecord* record, const void * value) override;
};

#endif