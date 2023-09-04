#include "multi_write_64.hpp"
#include "scalar.hpp"
#include "zkassert.hpp"

using namespace std;

MultiWrite64::MultiWrite64(Goldilocks & fr) :
    fr(fr),
    lastFlushId(0),
    storedFlushId(0),
    storingFlushId(0),
    pendingToFlushDataIndex(0),
    storingDataIndex(2),
    synchronizingDataIndex(2)
{
    // Init mutex
    pthread_mutex_init(&mutex, NULL);

    // Reset data
    data[0].Reset();
    data[1].Reset();
    data[2].Reset();
};

string MultiWrite64::print(void)
{
    return "lastFlushId=" + to_string(lastFlushId) +
        " storedFlushId=" + to_string(storedFlushId) +
        " storingFlushId=" + to_string(storingFlushId) +
        " pendingToFlushDataIndex=" + to_string(pendingToFlushDataIndex) +
        " storingDataIndex=" + to_string(storingDataIndex) +
        " synchronizingDataIndex=" + to_string(synchronizingDataIndex);
}

bool MultiWrite64::findNode(const string &key, string &value)
{
    value.clear();
    bool bResult = false;
    Lock();

    unordered_map<string, string>::const_iterator it;

    // Search in data[pendingToFlushDataIndex].nodes
    if (bResult == false)
    {
        it = data[pendingToFlushDataIndex].nodes.find(key);
        if (it != data[pendingToFlushDataIndex].nodes.end())
        {
            value = it->second;
            bResult = true;

#ifdef LOG_DB_MULTI_WRITE_FIND_NODES
            zklog.info("MultiWrite64::findNodes() data[pendingToFlushDataIndex].nodes found key=" + key + " value=" + it->second);
#endif
        }
    }

    // Search in data[pendingToFlushDataIndex].nodesIntray
    if (bResult == false)
    {
        it = data[pendingToFlushDataIndex].nodesIntray.find(key);
        if (it != data[pendingToFlushDataIndex].nodesIntray.end())
        {
            value = it->second;
            bResult = true;

#ifdef LOG_DB_MULTI_WRITE_FIND_NODES
            zklog.info("MultiWrite64::findNodes() data[pendingToFlushDataIndex].nodesIntray found key=" + key + " value=" + it->second);
#endif
        }
    }

    //if (storingDataIndex != pendingToFlushDataIndex)
    // If there is still some data pending to be stored on database
    if (storingFlushId != storedFlushId)
    {
        // Search in data[storingDataIndex].nodes
        if (bResult == false)
        {
            it = data[storingDataIndex].nodes.find(key);
            if (it != data[storingDataIndex].nodes.end())
            {
                value = it->second;
                bResult = true;
#ifdef LOG_DB_MULTI_WRITE_FIND_NODES
                zklog.info("MultiWrite64::findNodes() data[pendingToFlushDataIndex].nodes found key=" + key + " value=" + it->second);
#endif
            }
        }

        // data[storingDataIndex].nodesIntray must be empty
        zkassert(data[storingDataIndex].nodesIntray.size() == 0);
    }

    Unlock();

    return bResult;
}

bool MultiWrite64::findProgram(const string &key, vector<uint8_t> &value)
{
    value.clear();
    bool bResult = false;
    Lock();

    unordered_map<string, string>::const_iterator it;

    // Search in data[pendingToFlushDataIndex].program
    if (bResult == false)
    {
        it = data[pendingToFlushDataIndex].program.find(key);
        if (it != data[pendingToFlushDataIndex].program.end())
        {
            if ((it->second.size() % 2) != 0)
            {
                zklog.error("MultiWrite64::findNode() data[pendingToFlushDataIndex].program found invalid node size=" + to_string(it->second.size()) + " for key=" + key);
            }
            else
            {
                string2ba(it->second, value);
                bResult = true;
            }
        }
    }

    // Search in data[pendingToFlushDataIndex].programIntray
    if (bResult == false)
    {
        it = data[pendingToFlushDataIndex].programIntray.find(key);
        if (it != data[pendingToFlushDataIndex].programIntray.end())
        {
            if ((it->second.size() % 2) != 0)
            {
                zklog.error("MultiWrite64::findNode() data[pendingToFlushDataIndex].programIntray found invalid node size=" + to_string(it->second.size()) + " for key=" + key);
            }
            else
            {
                string2ba(it->second, value);
                bResult = true;
            }
        }
    }

    // Search in data[storingDataIndex].program
    if (bResult == false)
    {
        it = data[storingDataIndex].program.find(key);
        if (it != data[storingDataIndex].program.end())
        {
            if ((it->second.size() % 2) != 0)
            {
                zklog.error("MultiWrite64::findNode() data[storingDataIndex].program found invalid node size=" + to_string(it->second.size()) + " for key=" + key);
            }
            else
            {
                string2ba(it->second, value);
                bResult = true;
            }
        }
    }

    // data[storingDataIndex].programIntray must be empty
    zkassert(data[storingDataIndex].programIntray.size() == 0);

    Unlock();

    return bResult;
}

bool MultiWrite64::findKeyValue(const uint64_t version, KeyValue &kv){

    bool bResult = false;
    Lock();

    unordered_map<uint64_t, KeyValue>::const_iterator it;

    // Search in data[pendingToFlushDataIndex].keyValue
    if (bResult == false)
    {
        it = data[pendingToFlushDataIndex].keyValue.find(version);
        if (it != data[pendingToFlushDataIndex].keyValue.end())
        {
            kv = it->second;
            bResult = true;

#ifdef LOG_DB_MULTI_WRITE_FIND_NODES
            zklog.info("MultiWrite64::findKeyValue() data[pendingToFlushDataIndex].keyValue found version=" + to_string(version) + " key=" + fea2string(fr, it->second.key.fe) + " value=" + it->second.value.get_str());
#endif
        }
    }

    // Search in data[pendingToFlushDataIndex].nodesIntray
    if (bResult == false)
    {
        it = data[pendingToFlushDataIndex].keyValueIntray.find(version);
        if (it != data[pendingToFlushDataIndex].keyValueIntray.end())
        {
            kv = it->second;
            bResult = true;

#ifdef LOG_DB_MULTI_WRITE_FIND_NODES
            zklog.info("MultiWrite64::findKeyValue() data[pendingToFlushDataIndex].keyValueIntray found version=" + to_string(version) + " key=" + fea2string(fr, it->second.key.fe) + " value=" + it->second.value.get_str());
#endif
        }
    }

    // If there is still some data pending to be stored on database
    if (storingFlushId != storedFlushId)
    {
        // Search in data[storingDataIndex].keyValue
        if (bResult == false)
        {
            it = data[storingDataIndex].keyValue.find(version);
            if (it != data[storingDataIndex].keyValue.end())
            {
                kv = it->second;
                bResult = true;
#ifdef LOG_DB_MULTI_WRITE_FIND_NODES
                zklog.info("MultiWrite64::findKeyValue() data[storingDataIndex].keyValueIntray found version=" + to_string(version) + " key=" + fea2string(fr, it->second.key.fe) + " value=" + it->second.value.get_str());
#endif
            }
        }

        // data[storingDataIndex].nodesIntray must be empty
        zkassert(data[storingDataIndex].keyValueIntray.size() == 0);
    }

    Unlock();

    return bResult;

}

bool MultiWrite64::findVersion(const string &key, uint64_t &version){
    
        bool bResult = false;
        Lock();
    
        unordered_map<string, uint64_t>::const_iterator it;
    
        // Search in data[pendingToFlushDataIndex].version
        if (bResult == false)
        {
            it = data[pendingToFlushDataIndex].version.find(key);
            if (it != data[pendingToFlushDataIndex].version.end())
            {
                version = it->second;
                bResult = true;
                #ifdef LOG_DB_MULTI_WRITE_FIND_NODES
                    zklog.info("MultiWrite64::findVersion() data[pendingToFlushDataIndex].version found key=" + key + " version=" + to_string(version));
                #endif
            }
        }

        // Search in data[pendingToFlushDataIndex].versionIntray
        if (bResult == false)
        {
            it = data[pendingToFlushDataIndex].versionIntray.find(key);
            if (it != data[pendingToFlushDataIndex].versionIntray.end())
            {
                version = it->second;
                bResult = true;
                #ifdef LOG_DB_MULTI_WRITE_FIND_NODES
                    zklog.info("MultiWrite64::findVersion() data[pendingToFlushDataIndex].versionIntray found key=" + key + " version=" + to_string(version));
                #endif
            }
        }

        // If there is still some data pending to be stored on database
        if (storingFlushId != storedFlushId)
        {
            // Search in data[storingDataIndex].version
            if (bResult == false)
            {
                it = data[storingDataIndex].version.find(key);
                if (it != data[storingDataIndex].version.end())
                {
                    version = it->second;
                    bResult = true;
                    #ifdef LOG_DB_MULTI_WRITE_FIND_NODES
                        zklog.info("MultiWrite64::findVersion() data[storingDataIndex].version found key=" + key + " version=" + to_string(version));
                    #endif
                }
            }

            // data[storingDataIndex].versionIntray must be empty
            zkassert(data[storingDataIndex].versionIntray.size() == 0);
        }

        Unlock();
        return bResult;

}