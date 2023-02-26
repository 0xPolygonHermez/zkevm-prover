#ifndef DATABASE_CACHE_HPP
#define DATABASE_CACHE_HPP

#include <vector>
#include "goldilocks_base_field.hpp"
#include <nlohmann/json.hpp>
#include <mutex>

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
    int64_t cacheSize = 0;
    int64_t cacheCurrentSize = 0;
    unordered_map<string, DatabaseCacheRecord*> cacheMap;
    DatabaseCacheRecord* head = NULL;
    DatabaseCacheRecord* last = NULL;

    DatabaseCache() : cacheCurrentSize(0), head(NULL), last(NULL) {};
    ~DatabaseCache();
    bool addRecord(const string key, DatabaseCacheRecord* record); // returns true if cache is full
    bool findKey(const string key, DatabaseCacheRecord* &record);
    virtual void freeRecord(DatabaseCacheRecord* record) {};

public:
    bool enabled() {return ((cacheSize == -1) || (cacheSize > 0));};
    void setCacheSize(int64_t size) {cacheSize = size;}; // size is in bytes. 0 = no cache; -1 = cache no limit
    void print(bool printContent);
};

class DatabaseMTCache : public DatabaseCache
{
public:  
    bool add(const string key, vector<Goldilocks::Element> value); // returns true if cache is full
    bool find(const string key, vector<Goldilocks::Element> &value);
    void freeRecord(DatabaseCacheRecord* record);
};

class DatabaseProgramCache : public DatabaseCache
{
public:  
    bool add(const string key, vector<uint8_t> value); // returns true if cache is full
    bool find(const string key, vector<uint8_t> &value);
    void freeRecord(DatabaseCacheRecord* record);
};

#endif