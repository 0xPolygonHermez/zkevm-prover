#include "database_associative_cache_64.hpp"
#include <vector>
#include "goldilocks_base_field.hpp"
#include <nlohmann/json.hpp>
#include <mutex>
#include "zklog.hpp"
#include "zkmax.hpp"
#include "exit_process.hpp"
#include "scalar.hpp"




DatabaseMTAssociativeCache64::DatabaseMTAssociativeCache64()
{
    nKeyBits = 0;
    indexesSize = 0;
    log2CacheSize = 0;
    cacheSize = 0;
    indexes = NULL;
    keys = NULL;
    values = NULL;
    isLeaf = NULL;
    currentCacheIndex = 0;
    attempts = 0;
    hits = 0;
    name = "";
};

DatabaseMTAssociativeCache64::DatabaseMTAssociativeCache64(int nKeyBits_, int cacheSize_, string name_)
{
    postConstruct(nKeyBits_, cacheSize_, name_);
};

DatabaseMTAssociativeCache64::~DatabaseMTAssociativeCache64()
{
    if (indexes != NULL)
        delete[] indexes;
    if (keys != NULL)
        delete[] keys;
    if (values != NULL)
        delete[] values;
    if (isLeaf != NULL)
        delete[] isLeaf;
};

void DatabaseMTAssociativeCache64::postConstruct(int nKeyBits_, int log2CacheSize_, string name_)
{
    nKeyBits = nKeyBits_;
    if (nKeyBits_ > 32)
    {
        zklog.error("DatabaseMTAssociativeCache64::DatabaseMTAssociativeCache64() nKeyBits_ > 32");
        exitProcess();
    }
    indexesSize = 1 << nKeyBits;

    log2CacheSize = log2CacheSize_;
    if (log2CacheSize_ > 32)
    {
        zklog.error("DatabaseMTAssociativeCache64::DatabaseMTAssociativeCache64() log2CacheSize_ > 32");
        exitProcess();
    }
    cacheSize = 1 << log2CacheSize_;

    if(indexes != NULL) delete[] indexes;
    indexes = new uint32_t[indexesSize];
    uint32_t initValue = UINT32_MAX-cacheSize-(uint32_t)1;
    #pragma omp parallel for schedule(static) num_threads(4)
    for (size_t i = 0; i < indexesSize; i++)
    {
        indexes[i] = initValue;
    }
    if(keys != NULL) delete[] keys;
    keys = new Goldilocks::Element[4 * cacheSize];

    if(values != NULL) delete[] values;
    values = new Goldilocks::Element[12 * cacheSize];
    if(isLeaf != NULL) delete[] isLeaf;
    isLeaf = new bool[cacheSize];

    currentCacheIndex = 0;
    attempts = 0;
    hits = 0;
    name = name_;

    cacheMask = 0;
    for (int i = 0; i < log2CacheSize; i++)
    {
        cacheMask = cacheMask << 1;
        cacheMask += 1;
    }
    assert(cacheMask == cacheSize - 1);

    indexesMask = 0;
    for (int i = 0; i < nKeyBits; i++)
    {
        indexesMask = indexesMask << 1;
        indexesMask += 1;
    }
    assert(indexesMask == indexesSize - 1);
};

void DatabaseMTAssociativeCache64::addKeyValue(Goldilocks::Element (&key)[4], const vector<Goldilocks::Element> &value, bool update)
{

    //
    //  Statistics
    //
    if (attempts%1000000 == 0)
    {
        zklog.info("DatabaseMTAssociativeCache64::addKeyValue() name=" + name + " indexesSize=" + to_string(indexesSize) + " cacheSize=" + to_string(cacheSize) + " attempts=" + to_string(attempts) + " hits=" + to_string(hits) + " hit ratio=" + to_string(double(hits) * 100.0 / double(zkmax(attempts, 1))) + "%");
    }

    //
    // Try to add in one of my 4 slots
    //
    for (int i = 0; i < 4; ++i)
    {
        uint32_t tableIndex = (uint32_t)(key[i].fe & indexesMask);
        uint32_t cacheIndexRaw = indexes[tableIndex];
        uint32_t cacheIndex = cacheIndexRaw & cacheMask;
        uint32_t cacheIndexKey, cacheIndexValue;
        bool write = false;

        if ((currentCacheIndex >= cacheIndexRaw &&  currentCacheIndex - cacheIndexRaw > cacheSize) ||
            (currentCacheIndex < cacheIndexRaw && UINT32_MAX - cacheIndexRaw + currentCacheIndex > cacheSize))
        {
            write = true;
            indexes[tableIndex] = currentCacheIndex;
            cacheIndex = currentCacheIndex & cacheMask;

            cacheIndexKey = cacheIndex * 4;
            cacheIndexValue = cacheIndex * 12;
            currentCacheIndex = (currentCacheIndex == UINT32_MAX) ? 0 : (currentCacheIndex + 1); 
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
            if (isLeaf[cacheIndex])
            {
                values[cacheIndexValue + 8] = value[8];
                values[cacheIndexValue + 9] = value[9];
                values[cacheIndexValue + 10] = value[10];
                values[cacheIndexValue + 11] = value[11];
            }else{
                values[cacheIndexValue + 8] = Goldilocks::zero();
                values[cacheIndexValue + 9] = Goldilocks::zero();
                values[cacheIndexValue + 10] = Goldilocks::zero();
                values[cacheIndexValue + 11] = Goldilocks::zero();
            }
            return;
        }else{
            return;
        }
    }
    //
    // forced entry insertion
    //
    uint32_t cacheIndex = (uint32_t)(currentCacheIndex & cacheMask);
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
    uint32_t rawCacheIndexes[10];
    rawCacheIndexes[0] = currentCacheIndex-1;
    forcedInsertion(rawCacheIndexes, iters);

}

void DatabaseMTAssociativeCache64::forcedInsertion(uint32_t (&rawCacheIndexes)[10], int &iters)
{
    uint32_t rawCacheIndex = rawCacheIndexes[iters];
    //
    // avoid infinite loop
    //
    iters++;
    if (iters > 9)
    {
        zklog.error("forcedInsertion() more than 10 iterations required. Index: " + to_string(rawCacheIndex));
        exitProcess();
    }    

    //
    // find a slot into my indexes
    //
    uint32_t cacheIndex = (uint32_t)(rawCacheIndex & cacheMask);
    Goldilocks::Element *key = &keys[cacheIndex * 4];
    uint32_t minRawCacheIndex = UINT32_MAX;
    int pos = -1;

    for (int i = 0; i < 4; ++i)
    {
        uint32_t tableIndex_ = (uint32_t)(key[i].fe & indexesMask);
        uint32_t rawCacheIndex_ = (uint32_t)(indexes[tableIndex_]);
        if ((currentCacheIndex >= rawCacheIndex_ &&  currentCacheIndex - rawCacheIndex_ > cacheSize) ||
            (currentCacheIndex < rawCacheIndex_ && UINT32_MAX - rawCacheIndex_ + currentCacheIndex > cacheSize))
        {
            indexes[tableIndex_] = rawCacheIndex;
            return;
        }
        else
        {
            bool used = false;
            for(int k=0; k<iters; k++){
                if(rawCacheIndexes[k] == rawCacheIndex_){
                    used = true;
                    break;
                }
            }
            if (!used && rawCacheIndex_ < minRawCacheIndex)
            {
                minRawCacheIndex = rawCacheIndex_;
                pos = i;
            }
        }
    }

    if (pos < 0)
    {
        zklog.error("forcedInsertion() could not continue the recursion: " + to_string(rawCacheIndex));
        exitProcess();
    }  
    indexes[(uint32_t)(key[pos].fe & indexesMask)] = rawCacheIndex;
    rawCacheIndexes[iters] = minRawCacheIndex;
    forcedInsertion(rawCacheIndexes, iters);
    
}

bool DatabaseMTAssociativeCache64::findKey(Goldilocks::Element (&key)[4], vector<Goldilocks::Element> &value)
{
    attempts++; 
    for (int i = 0; i < 4; i++)
    {
        uint32_t tableIndex = (uint32_t)(key[i].fe & indexesMask);
        uint32_t cacheIndexRaw = (uint32_t)(indexes[tableIndex]);
        uint32_t cacheIndex = cacheIndexRaw  & cacheMask;
        if ((currentCacheIndex >= cacheIndexRaw &&  currentCacheIndex - cacheIndexRaw > cacheSize) ||
            (currentCacheIndex < cacheIndexRaw && UINT32_MAX - cacheIndexRaw + currentCacheIndex > cacheSize))
            continue;
            
        uint32_t cacheIndexKey = cacheIndex * 4;

        if (keys[cacheIndexKey + 0].fe == key[0].fe &&
            keys[cacheIndexKey + 1].fe == key[1].fe &&
            keys[cacheIndexKey + 2].fe == key[2].fe &&
            keys[cacheIndexKey + 3].fe == key[3].fe)
        {
            uint32_t cacheIndexValue = cacheIndex * 12;
            ++hits; // must be atomic operation!! makes is sence?
            value.resize(12);
            value[0] = values[cacheIndexValue];
            value[1] = values[cacheIndexValue + 1];
            value[2] = values[cacheIndexValue + 2];
            value[3] = values[cacheIndexValue + 3];
            value[4] = values[cacheIndexValue + 4];
            value[5] = values[cacheIndexValue + 5];
            value[6] = values[cacheIndexValue + 6];
            value[7] = values[cacheIndexValue + 7];

            value[8] = values[cacheIndexValue + 8];
            value[9] = values[cacheIndexValue + 9];
            value[10] = values[cacheIndexValue + 10];
            value[11] = values[cacheIndexValue + 11];
            return true;
        }
    }
    return false;
}
