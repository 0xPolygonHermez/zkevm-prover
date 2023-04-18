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
    void add(const string key, vector<Goldilocks::Element> value, const bool cached, const uint64_t time);
    void add(const string key, vector<uint8_t> value, const bool cached, const uint64_t time);
    void addGetTree(const uint64_t time, const uint64_t numberOfFields);
    void add(MTMap &db);
    void add(ProgramMap &db);
    bool findMT(const string key, vector<Goldilocks::Element> &value);
    bool findProgram(const string key, vector<uint8_t> &value);
    MTMap getMTDB();
    ProgramMap getProgramDB();
    void setOnChangeCallback(void *instance, onChangeCallbackFunctionPtr function);
    void print(void);
};

#endif