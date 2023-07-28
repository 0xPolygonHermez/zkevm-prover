#include "database_associative_cache.hpp"
#include <vector>
#include "goldilocks_base_field.hpp"
#include <nlohmann/json.hpp>
#include <mutex>
#include "zklog.hpp"
#include "zkmax.hpp"
#include "exit_process.hpp"



DatabaseMTAssociativeCache::DatabaseMTAssociativeCache()
{
    nKeyBits = 0;
    indicesSize = 0;
    cacheSize = 0;
    indices = NULL;
    keys = NULL;
    values = NULL;
    isLeaf = NULL;
    currentCacheIndex = 0;
    attempts = 0;
    hits = 0;
    name = "";
};

DatabaseMTAssociativeCache::DatabaseMTAssociativeCache(int nKeyBits_, int cacheSize_, string name_)
{
    postConstruct(nKeyBits_, cacheSize_, name_);
};

DatabaseMTAssociativeCache::~DatabaseMTAssociativeCache()
{
    if (indices != NULL)
        delete[] indices;
    if (keys != NULL)
        delete[] keys;
    if (values != NULL)
        delete[] values;
    if (isLeaf != NULL)
        delete[] isLeaf;
};

void DatabaseMTAssociativeCache::postConstruct(int nKeyBits_, int cacheSize_, string name_)
{
    nKeyBits = nKeyBits_;
    if (nKeyBits_ > 64)
    {
        zklog.error("DatabaseMTAssociativeCache::DatabaseMTAssociativeCache() nKeyBits_ > 64");
        exit(1);
    }
    indicesSize = 1 << nKeyBits;
    cacheSize = cacheSize_;
    indices = new uint32_t[indicesSize];
    uint32_t initValue = cacheSize + 1;
    memset(indices, initValue, sizeof(uint32_t) * indicesSize);
    keys = new Goldilocks::Element[4 * cacheSize];
    values = new Goldilocks::Element[12 * cacheSize];
    isLeaf = new bool[cacheSize];
    currentCacheIndex = 0;
    attempts = 0;
    hits = 0;
    name = name_;

    indicesMask = 0;
    for (int i = 0; i < nKeyBits; i++)
    {
        indicesMask = indicesMask << 1;
        indicesMask += 1;
    }
};

void DatabaseMTAssociativeCache::addKeyValue(Goldilocks::Element (&key)[4], const vector<Goldilocks::Element> &value, bool update)
{
    //
    //  Statistics
    //
    attempts++; // must be atomic operation!! makes it sence? not really
    if (attempts << 44 == 0)
    {
        zklog.info("DatabaseMTAssociativeCache::addKeyValue() name=" + name + " indicesSize=" + to_string(indicesSize) + " cacheSize=" + to_string(cacheSize) + " attempts=" + to_string(attempts) + " hits=" + to_string(hits) + " hit ratio=" + to_string(double(hits) * 100.0 / double(zkmax(attempts, 1))) + "%");
    }

    //
    // Try to add in one of my 4 slots
    //
    for (int i = 0; i < 4; ++i)
    {
        uint32_t tableIndex = (uint32_t)(key[i].fe & indicesMask);
        uint32_t cacheIndex = indices[tableIndex];
        uint32_t cacheIndexKey, cacheIndexValue;
        bool write = false;

        if (currentCacheIndex - cacheIndex > cacheSize ||
            (currentCacheIndex < cacheSize && UINT32_MAX - cacheIndex + currentCacheIndex > cacheSize))
        {
            write = true;
            cacheIndex = currentCacheIndex;
            currentCacheIndex = (currentCacheIndex == UINT32_MAX) ? 0 : (currentCacheIndex + 1); // atomic!
            indices[tableIndex] = cacheIndex;
            cacheIndexKey = cacheIndex * 4;
            cacheIndexValue = cacheIndex * 12;
        }
        else
        {
            cacheIndexKey = cacheIndex * 4;
            cacheIndexValue = cacheIndex * 12;

            if (keys[cacheIndexKey + 0].fe == key[0].fe &&
                keys[cacheIndexKey + 1].fe == key[1].fe &&
                keys[cacheIndexKey + 2].fe == key[2].fe &&
                keys[cacheIndexKey + 3].fe == key[3].fe)
            {
                hits++;
                write = update;
            }
            else
            {
                continue;
            }
        }
        if (write) // must be atomic operation!!
        {
            isLeaf[cacheIndex] = (value.size() > 8);
            keys[cacheIndexKey + 0].fe = key[0].fe;
            keys[cacheIndexKey + 1].fe = key[1].fe;
            keys[cacheIndexKey + 2].fe = key[2].fe;
            keys[cacheIndexKey + 3].fe = key[3].fe;
            values[cacheIndexValue + 0] = value[0];
            values[cacheIndexValue + 1] = value[1];
            values[cacheIndexValue + 2] = value[2];
            values[cacheIndexValue + 3] = value[3];
            values[cacheIndexValue + 4] = value[4];
            values[cacheIndexValue + 5] = value[5];
            values[cacheIndexValue + 6] = value[6];
            values[cacheIndexValue + 7] = value[7];
            if (isLeaf[indices[tableIndex]])
            {
                values[cacheIndexValue + 8] = value[8];
                values[cacheIndexValue + 9] = value[9];
                values[cacheIndexValue + 10] = value[10];
                values[cacheIndexValue + 11] = value[11];
            }
            return;
        }
    }
    //
    // forced entry insertion
    //
    uint32_t cacheIndex = currentCacheIndex;
    currentCacheIndex = (currentCacheIndex == UINT32_MAX) ? 0 : (currentCacheIndex + 1); // atomic!
    uint32_t cacheIndexKey = cacheIndex * 4;
    uint32_t cacheIndexValue = cacheIndex * 12;
    isLeaf[cacheIndex] = (value.size() > 8);
    keys[cacheIndexKey + 0].fe = key[0].fe;
    keys[cacheIndexKey + 1].fe = key[1].fe;
    keys[cacheIndexKey + 2].fe = key[2].fe;
    keys[cacheIndexKey + 3].fe = key[3].fe;
    values[cacheIndexValue + 0] = value[0];
    values[cacheIndexValue + 1] = value[1];
    values[cacheIndexValue + 2] = value[2];
    values[cacheIndexValue + 3] = value[3];
    values[cacheIndexValue + 4] = value[4];
    values[cacheIndexValue + 5] = value[5];
    values[cacheIndexValue + 6] = value[6];
    values[cacheIndexValue + 7] = value[7];
    if (isLeaf[cacheIndex])
    {
        values[cacheIndexValue + 8] = value[8];
        values[cacheIndexValue + 9] = value[9];
        values[cacheIndexValue + 10] = value[10];
        values[cacheIndexValue + 11] = value[11];
    }
    //
    // Forced index insertion
    //
    int iters = 0;
    forcedInsertion(cacheIndex, iters);
}

void DatabaseMTAssociativeCache::forcedInsertion(uint32_t index, int &iters)
{
    //
    // avoid infinite loop
    //
    iters++;
    if (iters > 5)
    {
        zklog.error("forcedInsertion() more than 5 iterations required!");
        exitProcess();
    }    

    //
    // find a slot into my indices
    //
    Goldilocks::Element *key = &keys[index * 4];
    uint32_t minCacheIndex = UINT32_MAX;
    int pos = 0;

    for (int i = 0; i < 4; ++i)
    {
        uint32_t tableIndex = (uint32_t)(key[i].fe & indicesMask);
        uint32_t cacheIndex = indices[tableIndex];
        if (currentCacheIndex - cacheIndex > cacheSize ||
            (currentCacheIndex < cacheSize && UINT32_MAX - cacheIndex + currentCacheIndex > cacheSize))
        {
            indices[tableIndex] = index;
            return;
        }
        else
        {
            if (cacheIndex < minCacheIndex && cacheIndex != index)
            {
                minCacheIndex = cacheIndex;
                pos = i;
            }
        }
    }
    indices[(uint32_t)(key[pos].fe & indicesMask)] = index;
    forcedInsertion(minCacheIndex, iters);
}

bool DatabaseMTAssociativeCache::findKey(Goldilocks::Element (&key)[4], vector<Goldilocks::Element> &value)
{
    for (int i = 0; i < 4; i++)
    {
        uint32_t tableIndex = (uint32_t)(key[i].fe & indicesMask);
        uint32_t cacheIndex = indices[tableIndex];
        if (currentCacheIndex - cacheIndex > cacheSize ||
            (currentCacheIndex < cacheSize && UINT32_MAX - cacheIndex + currentCacheIndex > cacheSize))
            continue;
            
        uint32_t cacheIndexKey = cacheIndex * 4;

        if (keys[cacheIndexKey + 0].fe == key[0].fe &&
            keys[cacheIndexKey + 1].fe == key[1].fe &&
            keys[cacheIndexKey + 2].fe == key[2].fe &&
            keys[cacheIndexKey + 3].fe == key[3].fe)
        {
            uint32_t cacheIndexValue = cacheIndex * 12;
            ++hits; // must be atomic operation!! makes is sence?
            if (isLeaf[cacheIndex])
            {
                value.resize(12); // would like to avoid stl in the future...
            }
            else
            {
                value.resize(8);
            }
            value[0] = values[cacheIndexValue];
            value[1] = values[cacheIndexValue + 1];
            value[2] = values[cacheIndexValue + 2];
            value[3] = values[cacheIndexValue + 3];
            value[4] = values[cacheIndexValue + 4];
            value[5] = values[cacheIndexValue + 5];
            value[6] = values[cacheIndexValue + 6];
            value[7] = values[cacheIndexValue + 7];
            if (isLeaf[indices[tableIndex]])
            {
                value[8] = values[cacheIndexValue + 8];
                value[9] = values[cacheIndexValue + 9];
                value[10] = values[cacheIndexValue + 10];
                value[11] = values[cacheIndexValue + 11];
            }
            return true;
        }
    }
    return false;
}
