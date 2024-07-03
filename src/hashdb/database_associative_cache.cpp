#include "database_associative_cache.hpp"
#include <vector>
#include "goldilocks_base_field.hpp"
#include <nlohmann/json.hpp>
#include <mutex>
#include "zklog.hpp"
#include "zkmax.hpp"
#include "exit_process.hpp"
#include "scalar.hpp"
#include "zkassert.hpp"

#ifdef ENABLE_EXPERIMENTAL_CODE

DatabaseMTAssociativeCache::DatabaseMTAssociativeCache()
{
    log2IndexesSize = 0;
    indexesSize = 0;
    log2CacheSize = 0;
    cacheSize = 0;
    indexes = NULL;
    keys = NULL;
    isValidKey = NULL;
    values = NULL;
    currentCacheIndex = 0;
#ifdef LOG_ASSOCIATIVE_CACHE
    attempts = 0;
    hits = 0;
#endif
    name = "";
    auxBufferKeysValues.clear();
};

DatabaseMTAssociativeCache::DatabaseMTAssociativeCache(uint32_t log2IndexesSize_, uint32_t log2CacheSize_, string name_) :
    indexes(NULL), keys(NULL), isValidKey(NULL), values(NULL), currentCacheIndex(0), name(name_)
{
    postConstruct(log2IndexesSize_, log2CacheSize_, name_);
};

DatabaseMTAssociativeCache::DatabaseMTAssociativeCache(uint64_t cacheBytes_,string name_) :
indexes(NULL), keys(NULL), isValidKey(NULL), values(NULL), currentCacheIndex(0), name(name_)
{
    postConstruct(cacheBytes_, name_);
}

DatabaseMTAssociativeCache::~DatabaseMTAssociativeCache()
{
    if (indexes != NULL)
        delete[] indexes;
    if (keys != NULL)
        delete[] keys;
    if (values != NULL)
        delete[] values;
    if(isValidKey != NULL)
        delete[] isValidKey;
    auxBufferKeysValues.clear();

};

void DatabaseMTAssociativeCache::postConstruct(uint32_t log2IndexesSize_, uint32_t log2CacheSize_, string name_)
{
    unique_lock<shared_mutex> guard(mlock);
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
        zklog.error("DatabaseMTAssociativeCache::DatabaseMTAssociativeCache() log2CacheSize_ > 28");
        exitProcess();
    }
    cacheSize = 1 << log2CacheSize;
    cacheSizeDiv2 = cacheSize / 2;

    if ( indexesSize / cacheSize < 8)
    {
        // This forces that the maximum occupancy of the table of indexes is 12.5%
        zklog.error("DatabaseMTAssociativeCache::DatabaseMTAssociativeCache() indexesSize/ cacheSize < 8");
        exitProcess();
    }

    if(indexes == NULL){
        indexes = new uint32_t[indexesSize];
    }else{
        delete[] indexes;
        indexes = new uint32_t[indexesSize];
    }
    // initialization of indexes array with UINT32_MAX-cacheSize: the value
    // that is at distance cacheSize+1 from the initial currentCacheIndex, i.e. 0
    uint32_t initValue = UINT32_MAX-cacheSize;
    #pragma omp parallel for schedule(static) num_threads(4)
    for (size_t i = 0; i < indexesSize; i++)
    {
        indexes[i] = initValue;
    }
    if(keys == NULL){
        keys = new Goldilocks::Element[4 * cacheSize];
    }else{
        delete[] keys;
        keys = new Goldilocks::Element[4 * cacheSize];
    }

    if(isValidKey == NULL){
        isValidKey = new bool[cacheSize];
    }else{
        delete[] isValidKey;
        isValidKey = new bool[cacheSize];
    }
    #pragma omp parallel for schedule(static) num_threads(4)
    for(uint32_t i=0; i<cacheSize; i++){
        isValidKey[i] = false;
    }

    if(values == NULL){ 
        values = new Goldilocks::Element[12 * cacheSize];
    }else{
        delete[] values;
        values = new Goldilocks::Element[12 * cacheSize];
    }

    currentCacheIndex = 0;
#ifdef LOG_ASSOCIATIVE_CACHE
    attempts = 0;
    hits = 0;
#endif
    name = name_;
    
    //masks for fast module, note cache size and indexes size must be power of 2
    cacheMask = cacheSize - 1;
    indexesMask = indexesSize - 1;

    auxBufferKeysValues.clear();

};

void DatabaseMTAssociativeCache::postConstruct(uint64_t cacheBytes_, string name_){
    // Each cache entrie requires 16 Goldiclosk::Elements, 4 for the key and 12 for the value
    // This is 16*8 = 128 bytes
    uint32_t log2IndexesSize_ = 0;
    uint32_t log2CacheSize_ = 0;
    for(uint64_t i=28; i>=1; --i){
        uint64_t bytes = uint64_t(1<<i)*uint64_t(128)+uint64_t(1<<(i+3))*uint64_t(4);
        if(bytes <= cacheBytes_){

            int cacheMB = cacheBytes_/1024/1024;
            int usedMB = bytes/1024/1024;
            zklog.info( to_string(usedMB) + " MB of " + to_string(cacheMB) + " MB used for " + name_ + " associative cache. " + to_string(2*usedMB) + " MB should be exposed to efectively increase the cache size.");
            log2IndexesSize_ = i+3;
            log2CacheSize_ = i;
            break;
        }
    }
    postConstruct(log2IndexesSize_, log2CacheSize_, name_);

}

void DatabaseMTAssociativeCache::clear(){
    unique_lock<shared_mutex> guard(mlock);
    #pragma omp parallel for schedule(static) num_threads(4)
    for(uint32_t i=0; i<cacheSize; i++){
        isValidKey[i] = false;
    }    
    uint32_t initValue = UINT32_MAX-cacheSize;
    #pragma omp parallel for schedule(static) num_threads(4)
    for (size_t i = 0; i < indexesSize; i++)
    {
        indexes[i] = initValue;
    }
    auxBufferKeysValues.clear();
    currentCacheIndex = 0;
}

bool DatabaseMTAssociativeCache::extractKeyValueFromAuxBuffer_(const Goldilocks::Element (&key)[4], vector<Goldilocks::Element> &value){
    //
    // look at the auxBufferKeysValues (chances that this buffer has any entry are almost negligible),
    // for this reason this search is not optimized at all
    //
    if(auxBufferKeysValues.size() > 0){ 
        if(auxBufferKeysValues.size() % 17!= 0) {
            zklog.error("DatabaseMTAssociativeCache::cleanExpiredAuxBufferKeysValues_() auxBufferKeysValues.size() % 17!= 0");
            exitProcess();

        }
        for(size_t i=0; i<auxBufferKeysValues.size(); i+=17){

            if( !hasExpiredInBuffer_((uint32_t)(auxBufferKeysValues[i].fe)) &&
                auxBufferKeysValues[i+1].fe == key[0].fe &&
                auxBufferKeysValues[i+2].fe == key[1].fe &&
                auxBufferKeysValues[i+3].fe == key[2].fe &&
                auxBufferKeysValues[i+4].fe == key[3].fe){
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
                //erase the value
                auxBufferKeysValues.erase(auxBufferKeysValues.begin() + i, auxBufferKeysValues.begin() + i + 17);   
                return true;
            }
        } 
    }
    return false;
}

void DatabaseMTAssociativeCache::addKeyValue_(const Goldilocks::Element (&key)[4], const vector<Goldilocks::Element> &value, bool update)
{
    if(auxBufferKeysValues.size() > 0){
        vector<Goldilocks::Element> value_;
        extractKeyValueFromAuxBuffer_(key,value_);
        // We assume that if update was false is beacause value and value_ are equal
        // so we reincert value
    }
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

        if (!hasExpired_(cacheIndexRaw)){
            cacheIndex = cacheIndexRaw & cacheMask;
            uint32_t cacheIndexKey = cacheIndex * 4;
            if( keys[cacheIndexKey + 0].fe == key[0].fe &&
                keys[cacheIndexKey + 1].fe == key[1].fe &&
                keys[cacheIndexKey + 2].fe == key[2].fe &&
                keys[cacheIndexKey + 3].fe == key[3].fe){
                    if(distanceFromCurrentCacheIndex_(cacheIndexRaw) > cacheSizeDiv2){
                        // It is present but it is far from the currentCacheIndex, so we need to reinsert it
                        // We assume that if update was false is beacause value and value_ are equal
                        // so we reincert value
                        isValidKey[cacheIndex]=false;
                        if(emptySlot == false){
                            emptySlot = true;
                            tableIndexEmpty = tableIndex;
                        }
                    }else{
                        if(update == false) return;
                        present = true;
                        break;
                    }
            }
        }else if (emptySlot == false){
            emptySlot = true;
            tableIndexEmpty = tableIndex;
            // I can not break because I need to check if the key is present in the other slots
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
        //incerment the currentCacheIndex
        currentCacheIndex = (currentCacheIndex == UINT32_MAX) ? 0 : (currentCacheIndex + 1);
    }
    uint64_t cacheIndexKey, cacheIndexValue;
    cacheIndexKey = cacheIndex * 4;
    cacheIndexValue = cacheIndex * 12;
    
    //
    // Add value
    //
    if(!isValidKey[cacheIndex]) isValidKey[cacheIndex] = true; //avoid modify if uneccessary (cache effects)
    keys[cacheIndexKey + 0].fe = key[0].fe;
    keys[cacheIndexKey + 1].fe = key[1].fe;
    keys[cacheIndexKey + 2].fe = key[2].fe;
    keys[cacheIndexKey + 3].fe = key[3].fe;
    zkassert(value.size() == 12 || value.size() == 8);
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
        uint32_t iter = 0;
        uint32_t usedRawCacheIndexes[20]; // we will do at maximum 20 force transaction iterations
        usedRawCacheIndexes[0] = (currentCacheIndex == 0) ? UINT32_MAX : currentCacheIndex-1;
        forcedInsertion_(usedRawCacheIndexes, iter, update);
    }
    //
    // Clear empty auxBufferKeysValues slots (the probability of auxBufferKeysValues.size() > 0 is almost negligible)
    //
    if(auxBufferKeysValues.size() > 0){
        cleanExpiredAuxBufferKeysValues_();
    }
}

// This function is called recursively a maximum of 20 times. The probability of not finding a free slot at
// each call is multiplied by roughly 1/2**9 (three new keys are checks and I assume the ratio cacheSize/indexesSize 
// is 1/8). With this rough estimation the probablilty of requiring 20 iterations would be 1/2**180. If, after 20 
// iterations is not possible to find a free slot, the entry is added to the auxBufferKeysValues vector.
void DatabaseMTAssociativeCache::forcedInsertion_(uint32_t (&usedRawCacheIndexes)[20], uint32_t &iters, bool update)
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
        if (hasExpired_(rawCacheIndex_))
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
    if (iters >=20 || pos == -1)
    {

        //We can do a push_bach, we know the key is not in the buffer because it was found in the keys table
        zklog.warning("forcedInsertion_() maxforcedInsertion_Iterations reached");
        Goldilocks::Element *buffKey = &keys[(inputRawCacheIndex & cacheMask) * 4];
        Goldilocks::Element *buffValue = &values[(inputRawCacheIndex & cacheMask) * 12];
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
        return;
    }else{
        indexes[(uint32_t)(inputKey[pos].fe & indexesMask)] = inputRawCacheIndex;
        usedRawCacheIndexes[iters] = minRawCacheIndex; //new cache element to add in the indexes table
        forcedInsertion_(usedRawCacheIndexes, iters, update);
    }
}

bool DatabaseMTAssociativeCache::findKey_(const Goldilocks::Element (&key)[4], vector<Goldilocks::Element> &value, bool &reinsert)
{
    reinsert = false;
    //
    // look at the auxBufferKeysValues (chances that this buffer has any entry are almost negligible),
    // for this reason this search is not optimized at all
    //
    if(auxBufferKeysValues.size() > 0){ 
        if(auxBufferKeysValues.size() % 17!= 0) {
            zklog.error("DatabaseMTAssociativeCache::cleanExpiredAuxBufferKeysValues_() auxBufferKeysValues.size() % 17!= 0");
            exitProcess();

        }
        for(size_t i=0; i<auxBufferKeysValues.size(); i+=17){
            
            if( !hasExpiredInBuffer_((uint32_t)(auxBufferKeysValues[i].fe)) &&
                auxBufferKeysValues[i+1].fe == key[0].fe &&
                auxBufferKeysValues[i+2].fe == key[1].fe &&
                auxBufferKeysValues[i+3].fe == key[2].fe &&
                auxBufferKeysValues[i+4].fe == key[3].fe){
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
                if(distanceFromCurrentCacheIndex_((uint32_t)(auxBufferKeysValues[i].fe)) > cacheSizeDiv2){
                    reinsert = true;
                }
                return true;
            }
        } 
    }
    //
    // Look at the circulant buffer
    //
    for (int i = 0; i < 4; i++)
    {
        uint32_t cacheIndexRaw = indexes[key[i].fe & indexesMask];
        if (hasExpired_(cacheIndexRaw)) continue;
        
        uint32_t cacheIndex = cacheIndexRaw  & cacheMask;
        uint32_t cacheIndexKey = cacheIndex * 4;

        if (keys[cacheIndexKey + 0].fe == key[0].fe &&
            keys[cacheIndexKey + 1].fe == key[1].fe &&
            keys[cacheIndexKey + 2].fe == key[2].fe &&
            keys[cacheIndexKey + 3].fe == key[3].fe)
        {
            uint32_t cacheIndexValue = cacheIndex * 12;
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
            if(distanceFromCurrentCacheIndex_(cacheIndexRaw) > cacheSizeDiv2){
                reinsert = true;
            }
            return true;
        }
    }
    return false;
}
#endif
