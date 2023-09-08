#include "multi_write_64.hpp"
#include "scalar.hpp"
#include "zkassert.hpp"

using namespace std;

uint64_t previousAvailableVersion(const uint64_t versionIn, const vector<uint64_t> &versions)
{
    uint64_t versionOut = 0;
    for (auto it = versions.begin(); it != versions.end(); ++it)
    {
        if (*it < versionIn && *it > versionOut)
        {
            versionOut = *it;
        }
    }
    return versionOut;
}

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

bool MultiWrite64::findKeyValue(const uint64_t version,const Goldilocks::Element (&key)[4], mpz_class &value){


    bool bResult = false;
    Lock();
    string keyStr_ = fea2string(fr, key[0], key[1], key[2], key[3]);
    string keyStr = NormalizeToNFormat(keyStr_, 64); 


    map<uint64_t, vector<KeyValue>>::const_iterator it;

    // Search in data[pendingToFlushDataIndex].keyValueA
    if (bResult == false)
    {
        uint64_t versionPrevious = previousAvailableVersion(version, data[pendingToFlushDataIndex].keyVersions[keyStr]);
        if(versionPrevious != 0){
            it = data[pendingToFlushDataIndex].keyValueA.find(versionPrevious);
            if (it != data[pendingToFlushDataIndex].keyValueA.end())
            {
                for(auto it2 = it->second.begin(); it2 != it->second.end(); ++it2){
                    if(it2->key[0]==key[0] && it2->key[1]==key[1] && it2->key[2]==key[2] && it2->key[3]==key[3]){
                        value = it2->value;
                        bResult = true;
                        break;
                    }
                }

    #ifdef LOG_DB_MULTI_WRITE_FIND_NODES
                zklog.info("MultiWrite64::findkeyValueA() data[pendingToFlushDataIndex].keyValueA found version=" + to_string(version) + " key=" + keyStr + " value=" + value.get_str());
    #endif
            }
        }
    }
    unordered_map<uint64_t, vector<KeyValue>>::const_iterator it_;

    // Search in data[pendingToFlushDataIndex].keyValueAIntray
    if (bResult == false)
    {
        uint64_t versionPrevious = previousAvailableVersion(version, data[pendingToFlushDataIndex].keyVersionsIntray[keyStr]);
        if(versionPrevious != 0){
            it_ = data[pendingToFlushDataIndex].keyValueAIntray.find(versionPrevious);
            if (it_ != data[pendingToFlushDataIndex].keyValueAIntray.end())
            {
                for(auto it2 = it_->second.begin(); it2 != it_->second.end(); ++it2){
                    if(it2->key[0]==key[0] && it2->key[1]==key[1] && it2->key[2]==key[2] && it2->key[3]==key[3]){
                        value = it2->value;
                        bResult = true;
                        break;
                    }
                }

    #ifdef LOG_DB_MULTI_WRITE_FIND_NODES
                zklog.info("MultiWrite64::findkeyValueAIntray() data[pendingToFlushDataIndex].keyValueAIntray found version=" + to_string(version) + " key=" + keyStr + " value=" + value.get_str());
    #endif
            }
        }
    }

    // If there is still some data pending to be stored on database
    if (storingFlushId != storedFlushId)
    {

        // Search in data[storingDataIndex].keyValueA
        if (bResult == false)
        {
            uint64_t versionPrevious = previousAvailableVersion(version, data[storingDataIndex].keyVersions[keyStr]);
            if(versionPrevious != 0){
                it = data[storingDataIndex].keyValueA.find(versionPrevious);
                if (it != data[storingDataIndex].keyValueA.end())
                {
                    for(auto it2 = it->second.begin(); it2 != it->second.end(); ++it2){
                        if(it2->key[0]==key[0] && it2->key[1]==key[1] && it2->key[2]==key[2] && it2->key[3]==key[3]){
                            value = it2->value;
                            bResult = true;
                            break;
                        }
                    }

        #ifdef LOG_DB_MULTI_WRITE_FIND_NODES
                    zklog.info("MultiWrite64::findkeyValueA() data[storingDataIndex].keyValueA found version=" + to_string(version) + " key=" + keyStr + " value=" + value.get_str());
        #endif
                }
            }
        }
        

        // data[storingDataIndex].keyValueIntray must be empty
        zkassert(data[storingDataIndex].keyValueAIntray.size() == 0);
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