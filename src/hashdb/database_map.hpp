#ifndef DATABASE_MAP_HPP
#define DATABASE_MAP_HPP

#include <vector>
#include "goldilocks_base_field.hpp"
#include <nlohmann/json.hpp>
#include <mutex>

using namespace std;
using json = nlohmann::json;

class DatabaseMap;


class DatabaseMap
{
public:
    typedef unordered_map<string, vector<Goldilocks::Element>> MTMap;
    typedef unordered_map<string, string> MT64Map;
    typedef unordered_map<string, mpz_class> MT64KVMap;
    typedef unordered_map<string, uint64_t> MT64VersionMap;
    typedef unordered_map<string, vector<uint8_t>> ProgramMap;

private:
    typedef void(*onChangeCallbackFunctionPtr)(void*, DatabaseMap *dbMap);

    recursive_mutex mlock;
    MTMap mtDB;
    MT64Map mt64DB;
    MT64KVMap mt64KVDB;
    MT64VersionMap mt64VersionDB;
    ProgramMap programDB;
    bool callbackOnChange = false;
    bool saveKeys = false;
    onChangeCallbackFunctionPtr cbFunction = NULL;
    void *cbInstance = NULL;

    uint64_t mtCachedTimes;
    uint64_t mtCachedTime;
    uint64_t mtDbTimes;
    uint64_t mtDbTime;
    uint64_t mt64CachedTimes;
    uint64_t mt64CachedTime;
    uint64_t mt64DbTimes;
    uint64_t mt64DbTime;
    uint64_t mt64KVCachedTimes;
    uint64_t mt64KVCachedTime;
    uint64_t mt64KVDbTimes;
    uint64_t mt64KVDbTime;
    uint64_t mt64VersionCachedTimes;
    uint64_t mt64VersionCachedTime;
    uint64_t mt64VersionDbTimes;
    uint64_t mt64VersionDbTime;
    uint64_t programCachedTimes;
    uint64_t programCachedTime;
    uint64_t programDbTimes;
    uint64_t programDbTime;
    uint64_t getTreeTimes;
    uint64_t getTreeTime;
    uint64_t getTreeFields;
    

    void onChangeCallback();

public:
    DatabaseMap() :
        mtCachedTimes(0),
        mtCachedTime(0),
        mtDbTimes(0),
        mtDbTime(0),
        mt64CachedTimes(0),
        mt64CachedTime(0),
        mt64DbTimes(0),
        mt64DbTime(0),
        mt64KVCachedTimes(0),
        mt64KVCachedTime(0),
        mt64KVDbTimes(0),
        mt64KVDbTime(0),
        mt64VersionCachedTimes(0),
        mt64VersionCachedTime(0),
        mt64VersionDbTimes(0),
        mt64VersionDbTime(0),
        programCachedTimes(0),
        programCachedTime(0),
        programDbTimes(0),
        programDbTime(0),
        getTreeTimes(0),
        getTreeTime(0),
        getTreeFields(0)
    {

    };
    inline void add(const string& key, const vector<Goldilocks::Element>& value, const bool cached, const uint64_t time);
    inline void add(const string& key, const string& value, const bool cached, const uint64_t time);
    inline void add(const string& key, const vector<uint8_t>& value, const bool cached, const uint64_t time);
    inline void add(const string& key, const mpz_class& value, const bool cached, const uint64_t time);
    inline void add(const string& key, const uint64_t version, const bool cached, const uint64_t time);
    inline void addGetTree(const uint64_t time, const uint64_t numberOfFields);
    void add(MTMap &db);
    void add(ProgramMap &db);
    void add(MT64Map &db);
    void add(MT64KVMap &db);
    void add(MT64VersionMap &db);
    bool findMT(const string& key, vector<Goldilocks::Element> &value);
    bool findMT64(const string& key, string &value);
    bool findMT64KV(const string& key, mpz_class &value);
    bool findMT64Version(const string& key, uint64_t &value);
    bool findProgram(const string& key, vector<uint8_t> &value);
    MTMap getMTDB();
    MT64Map getMT64DB();
    MT64KVMap getMT64KVDB();
    MT64VersionMap getMT64VersionDB();
    ProgramMap getProgramDB();
    void setOnChangeCallback(void *instance, onChangeCallbackFunctionPtr function);
    inline void setSaveKeys(const bool saveKeys_){ saveKeys = saveKeys_; };
    inline bool getSaveKeys(){ return saveKeys; };
    void print(void);
};

void DatabaseMap::add(const string& key, const vector<Goldilocks::Element>& value, const bool cached, const uint64_t time)
{
    lock_guard<recursive_mutex> guard(mlock);
    if(saveKeys) mtDB[key] = value;
    if (cached)
    {
        mtCachedTimes += 1;
        mtCachedTime += time;
    }
    else
    {
        mtDbTimes += 1;
        mtDbTime += time;
    }
    if (callbackOnChange) onChangeCallback();
}

void DatabaseMap::add(const string& key, const string &value, const bool cached, const uint64_t time)
{
    lock_guard<recursive_mutex> guard(mlock);
    if(saveKeys) mt64DB[key] = value;
    if (cached)
    {
        mt64CachedTimes += 1;
        mt64CachedTime += time;
    }
    else
    {
        mt64DbTimes += 1;
        mt64DbTime += time;
    }
    if (callbackOnChange) onChangeCallback();
}

void DatabaseMap::add(const string& key, const vector<uint8_t>& value, const bool cached, const uint64_t time)
{
    lock_guard<recursive_mutex> guard(mlock);

    if(saveKeys) programDB[key] = value;
    if (cached)
    {
        programCachedTimes += 1;
        programCachedTime += time;
    }
    else
    {
        programDbTimes += 1;
        programDbTime += time;
    }
    if (callbackOnChange) onChangeCallback();
}

void DatabaseMap::add(const string& key, const mpz_class& value, const bool cached, const uint64_t time)
{
    lock_guard<recursive_mutex> guard(mlock);
    if(saveKeys) mt64KVDB[key] = value;
    if (cached)
    {
        mt64KVCachedTimes += 1;
        mt64KVCachedTime += time;
    }
    else
    {
        mt64KVDbTimes += 1;
        mt64KVDbTime += time;
    }
    if (callbackOnChange) onChangeCallback();
}

void DatabaseMap::add(const string& key, const uint64_t version, const bool cached, const uint64_t time)
{
    lock_guard<recursive_mutex> guard(mlock);
    if(saveKeys) mt64VersionDB[key] = version;
    if (cached)
    {
        mt64VersionCachedTimes += 1;
        mt64VersionCachedTime += time;
    }
    else
    {
        mt64VersionDbTimes += 1;
        mt64VersionDbTime += time;
    }
    if (callbackOnChange) onChangeCallback();
}

void DatabaseMap::addGetTree(const uint64_t time, const uint64_t numberOfFields)
{
    lock_guard<recursive_mutex> guard(mlock);
    getTreeTimes += 1;
    getTreeTime += time;
    getTreeFields += numberOfFields;
}

#endif