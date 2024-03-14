#ifndef DATABASE_CACHE_64_HPP
#define DATABASE_CACHE_64_HPP

#include <vector>
#include "goldilocks_base_field.hpp"
#include <nlohmann/json.hpp>
#include <mutex>
#include "zklog.hpp"

using namespace std;
using json = nlohmann::json;

struct DatabaseCacheRecord64 {
    string key; // TODO: use *char instead? it uses less memory than a string, but we need to convert
    void* value;
    DatabaseCacheRecord64* next;
    DatabaseCacheRecord64* prev;
    uint64_t size;
};

class DatabaseCache64
{
protected:
    recursive_mutex mlock;
    uint64_t maxSize;
    uint64_t currentSize;
    unordered_map<string, DatabaseCacheRecord64*> cacheMap;
    DatabaseCacheRecord64 * head;
    DatabaseCacheRecord64 * last;
    uint64_t attempts;
    uint64_t hits;
    string name;

    DatabaseCache64() :
        maxSize(0),
        currentSize(0),
        head(NULL),
        last(NULL),
        attempts(0),
        hits(0)
        {};
    ~DatabaseCache64();
    bool addKeyValue(const string &key, const void * value, const bool update); // returns true if cache is full
    bool findKey(const string &key, DatabaseCacheRecord64* &record);

public:
    virtual DatabaseCacheRecord64* allocRecord(const string key, const void * value) = 0;
    virtual void freeRecord(DatabaseCacheRecord64* record) = 0;
    virtual void updateRecord(DatabaseCacheRecord64* record, const void * value) = 0;

public:
    uint64_t getMaxSize(void) { return maxSize; };
    uint64_t getCurrentSize(void) { return currentSize; };
    bool enabled() {return (maxSize > 0);};
    void setMaxSize(int64_t size) { maxSize = size; }; // size is in bytes, 0 = no cache
    void setName(const char * pChar) { name = pChar; };
    void print(bool printContent);
    void clear(void);
};

class DatabaseMTCache64 : public DatabaseCache64
{
public:
    ~DatabaseMTCache64();
    bool add(const string &key, const vector<Goldilocks::Element> &value, const bool update); // returns true if cache is full
    bool find(const string &key, vector<Goldilocks::Element> &value);
    DatabaseCacheRecord64* allocRecord(const string key, const void * value) override;
    void freeRecord(DatabaseCacheRecord64* record) override;
    void updateRecord(DatabaseCacheRecord64* record, const void * value) override;
};

class DatabaseProgramCache64 : public DatabaseCache64
{
public:  
    ~DatabaseProgramCache64();
    bool add(const string &key, const vector<uint8_t> &value, const bool update); // returns true if cache is full
    bool find(const string &key, vector<uint8_t> &value);
    DatabaseCacheRecord64* allocRecord(const string key, const void * value) override;
    void freeRecord(DatabaseCacheRecord64* record) override;
    void updateRecord(DatabaseCacheRecord64* record, const void * value) override;
};

#endif