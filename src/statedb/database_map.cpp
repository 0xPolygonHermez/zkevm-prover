#include "database_map.hpp"
#include "utils.hpp"
#include "scalar.hpp"

void DatabaseMap::add(const string key, vector<Goldilocks::Element> value)
{
    lock_guard<recursive_mutex> guard(mlock);

    mtDB[key] = value;
    if (callbackOnChange) onChangeCallback();
}

void DatabaseMap::add(const string key, vector<uint8_t> value)
{
    lock_guard<recursive_mutex> guard(mlock);

    programDB[key] = value;
    if (callbackOnChange) onChangeCallback();
}

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

bool DatabaseMap::findMT(const string key, vector<Goldilocks::Element> &value)
{
    lock_guard<recursive_mutex> guard(mlock);

    if (mtDB.find(key) != mtDB.end())
    {
        value = mtDB[key];
        return true;
    }

    return false;
}

bool DatabaseMap::findProgram(const string key, vector<uint8_t> &value)
{
    lock_guard<recursive_mutex> guard(mlock);

    if (programDB.find(key) != programDB.end())
    {
        value = programDB[key];
        return true;
    }

    return false;
}

map<string, vector<Goldilocks::Element>> DatabaseMap::getMTDB()
{
    lock_guard<recursive_mutex> guard(mlock);

    return mtDB;
}

map<string, vector<uint8_t>> DatabaseMap::getProgramDB()
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
