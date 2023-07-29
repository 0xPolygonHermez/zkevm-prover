#include "database_map.hpp"
#include "utils.hpp"
#include "scalar.hpp"
#include "zklog.hpp"
#include "zkmax.hpp"



void DatabaseMap::add(MTMap &db)
{
    lock_guard<recursive_mutex> guard(mlock);

    mtDB.insert(db.begin(), db.end());
    if (callbackOnChange) onChangeCallback();
}

void DatabaseMap::add(ProgramMap &db)
{
    lock_guard<recursive_mutex> guard(mlock);

    programDB.insert(db.begin(), db.end());
    if (callbackOnChange) onChangeCallback();
}

bool DatabaseMap::findMT(const string& key, vector<Goldilocks::Element> &value)
{
    lock_guard<recursive_mutex> guard(mlock);

    DatabaseMap::MTMap::iterator it = mtDB.find(key);

    if (it != mtDB.end())
    {
        value = it->second;
        return true;
    }

    return false;
}

bool DatabaseMap::findProgram(const string& key, vector<uint8_t> &value)
{
    lock_guard<recursive_mutex> guard(mlock);

    DatabaseMap::ProgramMap::iterator it = programDB.find(key);

    if (it != programDB.end())
    {
        value = it->second;
        return true;
    }

    return false;
}

DatabaseMap::MTMap DatabaseMap::getMTDB()
{
    lock_guard<recursive_mutex> guard(mlock);

    return mtDB;
}

DatabaseMap::ProgramMap DatabaseMap::getProgramDB()
{
    lock_guard<recursive_mutex> guard(mlock);

    return programDB;
}

void DatabaseMap::setOnChangeCallback(void *instance, onChangeCallbackFunctionPtr function)
{
    lock_guard<recursive_mutex> guard(mlock);

    if ((instance != NULL) && (function != NULL))
    {
        cbFunction = function;
        cbInstance = instance;
        callbackOnChange = true;
    } else callbackOnChange = false;
}

void DatabaseMap::onChangeCallback()
{
    cbFunction(cbInstance, this);
}

void DatabaseMap::print(void)
{
    zklog.info(
        string("Database map: dbMetrics") +
        " MT.size=" + to_string(mtDB.size()) +
        " cached=" + to_string(mtCachedTimes) + "times=" + to_string(mtCachedTime) + "us=" + to_string(mtCachedTime/zkmax(mtCachedTimes,1)) + "us/time" +
        " db=" + to_string(mtDbTimes) + "times=" + to_string(mtDbTime) + "us=" + to_string(mtDbTime/zkmax(mtDbTimes,1)) + "us/time" +
        " cacheHitRatio=" + to_string(mtCachedTimes*100/zkmax(mtCachedTimes+mtDbTimes,1)) + "%" +
        " PROGRAM.size=" + to_string(programDB.size()) +
        " cached=" + to_string(programCachedTimes) + "times=" + to_string(programCachedTime) + "us=" + to_string(programCachedTime/zkmax(programCachedTimes,1)) + "us/time" +
        " db=" + to_string(programDbTimes) + "times=" + to_string(programDbTime) + "us=" + to_string(programDbTime/zkmax(programDbTimes,1)) + "us/time" +
        " cacheHitRatio=" + to_string(programCachedTimes*100/zkmax(programCachedTimes+programDbTimes,1)) + "%" +
        " GET_TREE " + to_string(getTreeTimes) + "times=" + to_string(getTreeTime) + "us=" + to_string(getTreeTime/zkmax(getTreeTimes,1)) + "us/time=" + to_string(getTreeFields) + "fields=" + to_string(double(getTreeFields)/zkmax(getTreeTimes,1)) + "fields/time=" + to_string(getTreeTime/zkmax(getTreeFields,1)) +"us/field"
        );
}