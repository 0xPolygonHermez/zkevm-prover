#include "database_associative_cache.hpp"
#include <vector>
#include "goldilocks_base_field.hpp"
#include <nlohmann/json.hpp>
#include <mutex>
#include "zklog.hpp"
#include "zkmax.hpp"
#include "exit_process.hpp"
#include "scalar.hpp"




DatabaseMTAssociativeCache::DatabaseMTAssociativeCache()
{
    log2IndexesSize = 0;
    indexesSize = 0;
    log2CacheSize = 0;
    cacheSize = 0;
    indexes = NULL;
    keys = NULL;
    values = NULL;
    currentCacheIndex = 0;
    attempts = 0;
    hits = 0;
    name = "";
    auxBufferKeysValues.clear();
};

DatabaseMTAssociativeCache::DatabaseMTAssociativeCache(uint32_t log2IndexesSize_, uint32_t cacheSize_, string name_) :
    indexes(NULL), keys(NULL), values(NULL), currentCacheIndex(0), attempts(0), hits(0), name(name_)
{
    postConstruct(log2IndexesSize_, cacheSize_, name_);
};

DatabaseMTAssociativeCache::~DatabaseMTAssociativeCache()
{
    if (indexes != NULL)
        delete[] indexes;
    if (keys != NULL)
        delete[] keys;
    if (values != NULL)
        delete[] values;
    auxBufferKeysValues.clear();

};

void DatabaseMTAssociativeCache::postConstruct(uint32_t log2IndexesSize_, uint32_t log2CacheSize_, string name_)
{
    lock_guard<recursive_mutex> guard(mlock);
    log2IndexesSize = log2IndexesSize_;
    if (log2IndexesSize_ > 31)
    {
        // if log2IndexesSize_ >=32, we would need to use uint64_t values to refere to positions in the indexes array
        zklog.error("DatabaseMTAssociativeCache::DatabaseMTAssociativeCache() log2IndexesSize_ > 31");
        exitProcess();
    }
    indexesSize = 1 << log2IndexesSize;

    log2CacheSize = log2CacheSize_;
    if (log2CacheSize_ > 28)
    {
        // if log2CacheSize_ > 28, we would need to use uint64_t values to refere to positions in the keys and values arrays
        // note thar for each cacheIndex we have 4 keys and 12 values
        zklog.error("DatabaseMTAssociativeCache::DatabaseMTAssociativeCache() log2CacheSize_ > 31");
        exitProcess();
    }
    cacheSize = 1 << log2CacheSize;

    if ( indexesSize / cacheSize < 8)
    {
        // This forces that the maximum occupancy of the table of indexes is 12.5%
        zklog.error("DatabaseMTAssociativeCache::DatabaseMTAssociativeCache() indexesSize/ cacheSize < 8");
        exitProcess();
    }

    if(indexes != NULL) delete[] indexes;
    indexes = new uint32_t[indexesSize];
    //initialization of indexes array with UINT32_MAX-cacheSize: the value that is at distance
    //cacheSize+1 from the initial currentCacheIndex, i.e. 0
    uint32_t initValue = UINT32_MAX-cacheSize;
    #pragma omp parallel for schedule(static) num_threads(4)
    for (size_t i = 0; i < indexesSize; i++)
    {
        indexes[i] = initValue;
    }

    if(keys != NULL) delete[] keys;
    keys = new Goldilocks::Element[4 * cacheSize];

    if(values != NULL) delete[] values;
    values = new Goldilocks::Element[12 * cacheSize];

    currentCacheIndex = 0;
    attempts = 0;
    hits = 0;
    name = name_;
    
    //masks for fast module, note cache size and indexes size must be power of 2
    cacheMask = cacheSize - 1;
    indexesMask = indexesSize - 1;

    auxBufferKeysValues.clear();
};

void DatabaseMTAssociativeCache::addKeyValue(Goldilocks::Element (&key)[4], const vector<Goldilocks::Element> &value, bool update)
{
    lock_guard<recursive_mutex> guard(mlock);
    bool emptySlot = false;
    bool present = false;
    uint32_t cacheIndex;
    uint32_t tableIndexEmpty=0;

    //
    // Check if present in one of the four slots or there is an empty slot
    //
    for (int i = 0; i < 4; ++i)
    {
        uint32_t tableIndex = key[i].fe & indexesMask;
        uint32_t cacheIndexRaw = indexes[tableIndex];

        if (!emptyCacheSlot(cacheIndexRaw)){
            cacheIndex = cacheIndexRaw & cacheMask;
            uint32_t cacheIndexKey = cacheIndex * 4;
            if( keys[cacheIndexKey + 0].fe == key[0].fe &&
                keys[cacheIndexKey + 1].fe == key[1].fe &&
                keys[cacheIndexKey + 2].fe == key[2].fe &&
                keys[cacheIndexKey + 3].fe == key[3].fe){
                    if(update == false) return;
                    present = true;
                    break;
            }
        }else if (emptySlot == false){
            emptySlot = true;
            tableIndexEmpty = tableIndex;
        }
    }

    //
    // Evaluate cacheIndexKey 
    //
    if(!present){
        if(emptySlot == true){
            indexes[tableIndexEmpty] = currentCacheIndex;
        }
        cacheIndex = currentCacheIndex & cacheMask;
        currentCacheIndex = (currentCacheIndex == UINT32_MAX) ? 0 : (currentCacheIndex + 1);
    }
    uint64_t cacheIndexKey, cacheIndexValue;
    cacheIndexKey = cacheIndex * 4;
    cacheIndexValue = cacheIndex * 12;
    
    //
    // Add value
    //
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
    if (value.size() > 8)
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
            
    //
    // Forced index insertion
    //
    if(!present && !emptySlot){
        uint32_t iters = 0;
        uint32_t usedRawCacheIndexes[20];
        usedRawCacheIndexes[0] = (currentCacheIndex == 0) ? UINT32_MAX : currentCacheIndex-1;
        forcedInsertion(usedRawCacheIndexes, iters, update);
    }
    //
    // Clear empty auxBufferKeysValues slots (the probability of auxBufferKeysValues.size() > 0 is almost negligible)
    //
    auto it = auxBufferKeysValues.begin();
    while (it < auxBufferKeysValues.end()) {
        if (emptyCacheSlot(static_cast<uint32_t>(it->fe))) {
            auto next_it = (std::distance(it, auxBufferKeysValues.end()) >= 17) ? it + 17 : auxBufferKeysValues.end();
            it = auxBufferKeysValues.erase(it, next_it);
        } else {
            if (std::distance(it, auxBufferKeysValues.end()) >= 17) {
                it += 17;
            } else {
                it = auxBufferKeysValues.end();
            }
        }
    }
}

// This function is called recursively a maximum of 20 times. The probability of not finding a free slot at
// each call is multiplied by roughly 1/2**9 (three new keys are checks and I assume the ratio cacheSize/indexesSize 
// is 1/8). With this rough estimation the probablilty of requiring 20 iterations would be 1/2**180. If, after 20 
// iterations is not possible to find a free slot, the entry is added to the auxBufferKeysValues vector.
void DatabaseMTAssociativeCache::forcedInsertion(uint32_t (&usedRawCacheIndexes)[20], uint32_t &iters, bool update)
{
    
    uint32_t inputRawCacheIndex = usedRawCacheIndexes[iters];
    iters++;

    //
    // find a slot into my indexes
    //
    Goldilocks::Element *inputKey = &keys[(inputRawCacheIndex & cacheMask) * 4];
    uint32_t minRawCacheIndex = UINT32_MAX;
    int pos = -1;

    for (int i = 0; i < 4; ++i)
    {
        uint32_t tableIndex_ = inputKey[i].fe & indexesMask;
        uint32_t rawCacheIndex_ = indexes[tableIndex_];
        if (emptyCacheSlot(rawCacheIndex_))
        {
            indexes[tableIndex_] = inputRawCacheIndex;
            return;
        }
        else
        {
            //consider minimum not used rawCacheIndex_
            bool used = false;
            for(uint32_t k=0; k<iters; k++){
                // remember that with vergy high probability iters < 3
                if(usedRawCacheIndexes[k] == rawCacheIndex_){
                    used = true;
                    break;
                }
            }
            if (!used && rawCacheIndex_ <= minRawCacheIndex)
            {
                minRawCacheIndex = rawCacheIndex_;
                pos = i;
            }
        }
    }
    
    //
    // avoid infinite loop, only 20 iterations allowed, pox < 0 means that there is no unused slot to continue iterating
    //
    if (iters >=20 || pos < 0)
    {
        zklog.warning("forcedInsertion() maxForcedInsertionIterations reached");
        Goldilocks::Element *buffKey = &keys[(inputRawCacheIndex & cacheMask) * 4];
        Goldilocks::Element *buffValue = &values[(inputRawCacheIndex & cacheMask) * 12];
        
        // check first if there is an slot in the vector if the key is already present
        int pos = -1;
        for(size_t i=0; i<auxBufferKeysValues.size(); i+=17){
            if( emptyCacheSlot(((uint32_t)(auxBufferKeysValues[i].fe)))){
                if(pos < 0) pos = i;
            }else{
                if( auxBufferKeysValues[i+1].fe == buffKey[0].fe &&
                    auxBufferKeysValues[i+2].fe == buffKey[1].fe &&
                    auxBufferKeysValues[i+3].fe == buffKey[2].fe &&
                    auxBufferKeysValues[i+4].fe == buffKey[3].fe){
                    if(update == false) return;
                    pos = i;
                }
            }
        }
        if(pos > 0){
            auxBufferKeysValues[pos] = Goldilocks::fromU64((uint64_t)(inputRawCacheIndex));
            auxBufferKeysValues[pos + 1] = buffKey[0];
            auxBufferKeysValues[pos + 2] = buffKey[1];
            auxBufferKeysValues[pos + 3] = buffKey[2];
            auxBufferKeysValues[pos + 4] = buffKey[3];    
            auxBufferKeysValues[pos + 5] = buffValue[0];
            auxBufferKeysValues[pos + 6] = buffValue[1];
            auxBufferKeysValues[pos + 7] = buffValue[2];
            auxBufferKeysValues[pos + 8] = buffValue[3];
            auxBufferKeysValues[pos + 9] = buffValue[4];
            auxBufferKeysValues[pos + 10] = buffValue[5];
            auxBufferKeysValues[pos + 11] = buffValue[6];
            auxBufferKeysValues[pos + 12] = buffValue[7];
            auxBufferKeysValues[pos + 13] = buffValue[8];
            auxBufferKeysValues[pos + 14] = buffValue[9];
            auxBufferKeysValues[pos + 15] = buffValue[10];
            auxBufferKeysValues[pos + 16] = buffValue[11];

        }else{
            auxBufferKeysValues.push_back(Goldilocks::fromU64((uint64_t)(inputRawCacheIndex)));
            auxBufferKeysValues.push_back(buffKey[0]);
            auxBufferKeysValues.push_back(buffKey[1]);
            auxBufferKeysValues.push_back(buffKey[2]);
            auxBufferKeysValues.push_back(buffKey[3]);
            auxBufferKeysValues.push_back(buffValue[0]);
            auxBufferKeysValues.push_back(buffValue[1]);
            auxBufferKeysValues.push_back(buffValue[2]);
            auxBufferKeysValues.push_back(buffValue[3]);
            auxBufferKeysValues.push_back(buffValue[4]);
            auxBufferKeysValues.push_back(buffValue[5]);
            auxBufferKeysValues.push_back(buffValue[6]);
            auxBufferKeysValues.push_back(buffValue[7]);
            auxBufferKeysValues.push_back(buffValue[8]);
            auxBufferKeysValues.push_back(buffValue[9]);
            auxBufferKeysValues.push_back(buffValue[10]);
            auxBufferKeysValues.push_back(buffValue[11]);
        }
        return;
    }else{
        indexes[(uint32_t)(inputKey[pos].fe & indexesMask)] = inputRawCacheIndex;
        usedRawCacheIndexes[iters] = minRawCacheIndex; //new cache element to add in the indexes table
        forcedInsertion(usedRawCacheIndexes, iters, update);
    }
}

bool DatabaseMTAssociativeCache::findKey(const Goldilocks::Element (&key)[4], vector<Goldilocks::Element> &value)
{
    lock_guard<recursive_mutex> guard(mlock);
    attempts++; 
    //
    //  Statistics
    //
    if (attempts<<34 == 0)
    {
        zklog.info("DatabaseMTAssociativeCache::findKey() name=" + name + " indexesSize=" + to_string(indexesSize) + " cacheSize=" + to_string(cacheSize) + " attempts=" + to_string(attempts) + " hits=" + to_string(hits) + " hit ratio=" + to_string(double(hits) * 100.0 / double(zkmax(attempts, 1))) + "%");
        if(auxBufferKeysValues.size()>0){
            zklog.warning("DatabaseMTAssociativeCache using auxBufferKeysValues" + to_string(auxBufferKeysValues.size()));
        }
    }
    //
    // Find the value
    //
    for (int i = 0; i < 4; i++)
    {
        uint32_t cacheIndexRaw = indexes[key[i].fe & indexesMask];
        if (emptyCacheSlot(cacheIndexRaw)) continue;
        
        uint32_t cacheIndex = cacheIndexRaw  & cacheMask;
        uint32_t cacheIndexKey = cacheIndex * 4;

        if (keys[cacheIndexKey + 0].fe == key[0].fe &&
            keys[cacheIndexKey + 1].fe == key[1].fe &&
            keys[cacheIndexKey + 2].fe == key[2].fe &&
            keys[cacheIndexKey + 3].fe == key[3].fe)
        {
            uint32_t cacheIndexValue = cacheIndex * 12;
            ++hits;
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
    //
    // look at the auxBufferKeysValues (chances that this buffer has any entry are almost negligible),
    // for this reason this search is not optimized at all
    //
    for(size_t i=0; i<auxBufferKeysValues.size(); i+=17){

        if( !emptyCacheSlot(((uint32_t)(auxBufferKeysValues[i].fe))) &&
            auxBufferKeysValues[i+1].fe == key[0].fe &&
            auxBufferKeysValues[i+2].fe == key[1].fe &&
            auxBufferKeysValues[i+3].fe == key[2].fe &&
            auxBufferKeysValues[i+4].fe == key[3].fe){
            ++hits;
            value.resize(12);
            value[0] = auxBufferKeysValues[i+5];
            value[1] = auxBufferKeysValues[i+6];
            value[2] = auxBufferKeysValues[i+7];
            value[3] = auxBufferKeysValues[i+8];
            value[4] = auxBufferKeysValues[i+9];
            value[5] = auxBufferKeysValues[i+10];
            value[6] = auxBufferKeysValues[i+11];
            value[7] = auxBufferKeysValues[i+12];
            value[8] = auxBufferKeysValues[i+13];
            value[9] = auxBufferKeysValues[i+14];
            value[10] = auxBufferKeysValues[i+15];
            value[11] = auxBufferKeysValues[i+16];
            return true;
        }
    } 
    
    return false;
}
