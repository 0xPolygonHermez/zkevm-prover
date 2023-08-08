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
    typedef unordered_map<string, vector<uint8_t>> ProgramMap;

private:
    typedef void(*onChangeCallbackFunctionPtr)(void*, DatabaseMap *dbMap);

    recursive_mutex mlock;
    MTMap mtDB;
    ProgramMap programDB;
    bool callbackOnChange = false;
    bool saveKeys = false;
    onChangeCallbackFunctionPtr cbFunction = NULL;
    void *cbInstance = NULL;

    uint64_t mtCachedTimes;
    uint64_t mtCachedTime;
    uint64_t mtDbTimes;
    uint64_t mtDbTime;
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
    inline void add(const string& key, const vector<uint8_t>& value, const bool cached, const uint64_t time);
    inline void addGetTree(const uint64_t time, const uint64_t numberOfFields);
    void add(MTMap &db);
    void add(ProgramMap &db);
    bool findMT(const string& key, vector<Goldilocks::Element> &value);
    bool findProgram(const string& key, vector<uint8_t> &value);
    MTMap getMTDB();
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

void DatabaseMap::addGetTree(const uint64_t time, const uint64_t numberOfFields)
{
    lock_guard<recursive_mutex> guard(mlock);
    getTreeTimes += 1;
    getTreeTime += time;
    getTreeFields += numberOfFields;
}

#endif