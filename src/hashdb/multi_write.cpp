#include "multi_write.hpp"
#include "scalar.hpp"
#include "zkassert.hpp"

using namespace std;

MultiWrite::MultiWrite(Goldilocks & fr) :
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

string MultiWrite::print(void)
{
    return "lastFlushId=" + to_string(lastFlushId) +
        " storedFlushId=" + to_string(storedFlushId) +
        " storingFlushId=" + to_string(storingFlushId) +
        " pendingToFlushDataIndex=" + to_string(pendingToFlushDataIndex) +
        " storingDataIndex=" + to_string(storingDataIndex) +
        " synchronizingDataIndex=" + to_string(synchronizingDataIndex);
}

bool MultiWrite::findNode(const string &key, vector<Goldilocks::Element> &value)
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
            if ((it->second.size() % 16) != 0)
            {
                zklog.error("MultiWrite::findNode() data[pendingToFlushDataIndex].nodes found invalid node size=" + to_string(it->second.size()) + " for key=" + key);
            }
            else
            {
                for (uint64_t i=0; i<it->second.size(); i+=16)
                {
                    Goldilocks::Element fe = fr.fromString(it->second.substr(i, 16), 16);
                    value.push_back(fe);
                }
                bResult = true;

#ifdef LOG_DB_MULTI_WRITE_FIND_NODES
                zklog.info("MultiWrite::findNodes() data[pendingToFlushDataIndex].nodes found key=" + key + " value=" + it->second);
#endif
            }
        }
    }

    // Search in data[pendingToFlushDataIndex].nodesIntray
    if (bResult == false)
    {
        it = data[pendingToFlushDataIndex].nodesIntray.find(key);
        if (it != data[pendingToFlushDataIndex].nodesIntray.end())
        {
            if ((it->second.size() % 16) != 0)
            {
                zklog.error("MultiWrite::findNode() data[pendingToFlushDataIndex].nodesIntray found invalid node size=" + to_string(it->second.size()) + " for key=" + key);
            }
            else
            {
                for (uint64_t i=0; i<it->second.size(); i+=16)
                {
                    Goldilocks::Element fe = fr.fromString(it->second.substr(i, 16), 16);
                    value.push_back(fe);
                }
                bResult = true;

#ifdef LOG_DB_MULTI_WRITE_FIND_NODES
                zklog.info("MultiWrite::findNodes() data[pendingToFlushDataIndex].nodesIntray found key=" + key + " value=" + it->second);
#endif
            }
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
                if ((it->second.size() % 16) != 0)
                {
                    zklog.error("MultiWrite::findNode() data[storingDataIndex].nodes found invalid node size=" + to_string(it->second.size()) + " for key=" + key);
                }
                else
                {
                    for (uint64_t i=0; i<it->second.size(); i+=16)
                    {
                        Goldilocks::Element fe = fr.fromString(it->second.substr(i, 16), 16);
                        value.push_back(fe);
                    }
                    bResult = true;
#ifdef LOG_DB_MULTI_WRITE_FIND_NODES
                    zklog.info("MultiWrite::findNodes() data[pendingToFlushDataIndex].nodes found key=" + key + " value=" + it->second);
#endif
                }
            }
        }

        // data[storingDataIndex].nodesIntray must be empty
        zkassert(data[storingDataIndex].nodesIntray.size() == 0);
    }

    Unlock();

    return bResult;
}

bool MultiWrite::findProgram(const string &key, vector<uint8_t> &value)
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
                zklog.error("MultiWrite::findNode() data[pendingToFlushDataIndex].program found invalid node size=" + to_string(it->second.size()) + " for key=" + key);
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
                zklog.error("MultiWrite::findNode() data[pendingToFlushDataIndex].programIntray found invalid node size=" + to_string(it->second.size()) + " for key=" + key);
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
                zklog.error("MultiWrite::findNode() data[storingDataIndex].program found invalid node size=" + to_string(it->second.size()) + " for key=" + key);
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