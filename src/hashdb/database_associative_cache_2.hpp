#ifndef DATABASE_ASSOCIATIVE_CACHE_2_HPP
#define DATABASE_ASSOCIATIVE_CACHE_2_HPP
#include <vector>
#include "goldilocks_base_field.hpp"
#include <nlohmann/json.hpp>
#include <mutex>
#include "zklog.hpp"

using namespace std;
using json = nlohmann::json;
template <class T>
class DatabaseAssociativeCache2
{
private:
    recursive_mutex mlock;

    int nKeyBits;
    int indicesSize;
    int cacheSize;

    uint32_t *indices;
    uint64_t *keys;
    T *values;
    int nextSlot;

    uint64_t attempts;
    uint64_t hits;
    string name;

    uint64_t indicesMask;

public:
    DatabaseAssociativeCache2();
    DatabaseAssociativeCache2(int nKeyBits_, int cacheSize_, string name_);
    ~DatabaseAssociativeCache2();

    void postConstruct(int nKeyBits_, int cacheSize_, string name_);
    void addKeyValue(Goldilocks::Element (&key)[4], const vector<Goldilocks::Element> &value);
    bool findKey(Goldilocks::Element (&key)[4], vector<Goldilocks::Element> &value);
};

// Methods:

template <class T>
DatabaseAssociativeCache2::DatabaseAssociativeCache2()
{
    nKeyBits = 0;
    indicesSize = 0;
    logCacheSize = 0;
    cacheSize = 0;
    indices = NULL;
    keys = NULL;
    values = NULL;
    nextSlot = 0;
    attempts = 0;
    hits = 0;
    name = "";
};

template <class T>
DatabaseAssociativeCache2::DatabaseAssociativeCache2(int nKeyBits_, int cacheSize_, string name_){
    postConstruct(nKeyBits_, cacheSize_, name_)};

template <class T>
DatabaseAssociativeCache2::~DatabaseAssociativeCache2()
{
    if (indices != NULL)
        delete[] indices;
    if (keys != NULL)
        delete[] keys;
    if (values != NULL)
        delete[] values;
};

template <class T>
void DatabaseAssociativeCache2::postConstruct(int nKeyBits_, int cacheSize_, string name_)
{
    nKeyBits = nKeyBits_;
    if (nKeyBits_ > 64)
    {
        zklog.error("DatabaseAssociativeCache2::DatabaseAssociativeCache2() nKeyBits_ > 64");
        exit(1);
    }
    indicesSize = 1 << nKeyBits;
    cacheSize = cacheSize_;
    indices = new uint32_t[indicesSize];
    uint32_t initValue = cacheSize + 1;
    memset(indices, initValue, sizeof(uint32_t) * indicesSize);
    keys = new uint64_t[4 * cacheSize];
    values = new T[12 * cacheSize];
    nextSlot = 0;
    attempts = 0;
    hits = 0;
    name = name_;

    indicesMask = 0;
    for (int i = 0; i < nKeyBits - 1; i++)
    {
        indicesMask = indicesMask << 1;
        indicesMask += 1;
    }
    cacheMask = 0;
    for (int i = 0; i < logCacheSize - 1; i++)
    {
        cacheMask = cacheMask << 1;
        cacheMask += 1;
    }
};

template <class T>
void DatabaseAssociativeCache2::addKeyValue(Goldilocks::Element (&key)[4], const vector<T> &value)
{

    attempts++; // must be atomic operation!! makes it sence? not really
    if (attempts << 44 == 0)
    {
        zklog.info("DatabaseAssociativeCache2::addKeyValue() name=" + name + " indicesSize=" + to_string(indicesSize) + " cacheSize=" + to_string(cacheSize) + " attempts=" + to_string(attempts) + " hits=" + to_string(hits) + " hit ratio=" + to_string(double(hits) * 100.0 / double(zkmax(attempts, 1))) + "%");
    }

    uint32_t offsetIndices = (uint32_t)(key[0].fe & indicesMask);
    uint32_t offsetKeys, offsetValues;
    bool update = false;
    if (indices[offsetIndices] > cacheSize)
    {
        update = true;
        indices[offsetIndices] = nextSlot; // must be atomic operation!!
        nextSlot++;                        // must be atomic operation!!
        if (nextSlot >= cacheSize)         // must be atomic operation!!
        {
            nextSlot = 0;
        }
        offsetKeys = indices[offsetIndices] * 4;
        offsetValues = indices[offsetIndices] * 12;
    }
    else
    {
        offsetKeys = indices[offsetIndices] * 4;
        offsetValues = indices[offsetIndices] * 12;
        if (keys[offsetKeys] == key[0].fe && keys[offsetKeys + 1] == key[1].fe && keys[offsetKeys + 2] == key[2].fe && keys[offsetKeys + 3] == key[3].fe)
        {
            hits++; // must be atomic operation!! makes is sence?
            update = false;
        }
        else
        {
            update = true;
        }
    }

    bool isleaf = value.size() > 8;
    if (update) // must be atomic operation!!
    {
        keys[offsetKeys] = key[0].fe;
        keys[offsetKeys + 1] = key[1].fe;
        keys[offsetKeys + 2] = key[2].fe;
        keys[offsetKeys + 3] = key[3].fe;
        values[offsetValues] = value[0];
        values[offsetValues + 1] = value[1];
        values[offsetValues + 2] = value[2];
        values[offsetValues + 3] = value[3];
        values[offsetValues + 4] = value[4];
        values[offsetValues + 5] = value[5];
        values[offsetValues + 6] = value[6];
        values[offsetValues + 7] = value[7];
        if (isleaf)
        {
            values[offsetValues + 8] = value[8];
            values[offsetValues + 9] = value[9];
            values[offsetValues + 10] = value[10];
            values[offsetValues + 11] = value[11];
        }
        else
        {
            values[offsetValues + 8] = 0xFFFFFFFFFFFFFFFF;
        }
    }
}

template <class T>
bool DatabaseAssociativeCache2::findKey(Goldilocks::Element (&key)[4], vector<T> &value)
{
    uint32_t offsetIndices = (uint32_t)(key[0].fe & indicesMask);
    if (indices[offsetIndices] > cacheSize)
    {
        return false;
    }
    uint32_t offsetKeys = indices[offsetIndices] * 4;
    if (keys[offsetKeys] == key[0].fe && keys[offsetKeys + 1] == key[1].fe && keys[offsetKeys + 2] == key[2].fe && keys[offsetKeys + 3] == key[3].fe)
    {
        ++hits; // must be atomic operation!! makes is sence?
        bool isleaf = (buffer_[offset + 12] != 0xFFFFFFFFFFFFFFFF);
        if (isleaf)
        {
            value.resize(12); // would like to avoid stl in the future...
        }
        else
        {
            value.resize(8);
        }
        value[0].fe = buffer_[offset];
        value[1].fe = buffer_[offset + 1];
        value[2].fe = buffer_[offset + 2];
        value[3].fe = buffer_[offset + 3];
        value[4].fe = buffer_[offset + 4];
        value[5].fe = buffer_[offset + 5];
        value[6].fe = buffer_[offset + 6];
        value[7].fe = buffer_[offset + 7];
        if (isleaf)
        {
            value[8].fe = buffer_[offset + 8];
            value[9].fe = buffer_[offset + 9];
            value[10].fe = buffer_[offset + 10];
            value[11].fe = buffer_[offset + 11];
        }
        return true;
    }
    return false;
}

// TODO:
// 5.understand when we use 8 or 12 values
// 6.repassar tamany caches
// 7.full understanding of atomics...
#endif
