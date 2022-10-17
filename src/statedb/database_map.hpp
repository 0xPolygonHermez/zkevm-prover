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

    void onChangeCallback();

public:
    DatabaseMap(){};
    void add(const string key, vector<Goldilocks::Element> value);
    void add(const string key, vector<uint8_t> value);
    void add(MTMap &db);
    void add(ProgramMap &db);
    bool findMT(const string key, vector<Goldilocks::Element> &value);
    bool findProgram(const string key, vector<uint8_t> &value);
    MTMap getMTDB();
    ProgramMap getProgramDB();
    void setOnChangeCallback(void *instance, onChangeCallbackFunctionPtr function);
};

#endif