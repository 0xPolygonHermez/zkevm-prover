#include "database_kv_associative_cache.hpp"
#include <vector>
#include "goldilocks_base_field.hpp"
#include <nlohmann/json.hpp>
#include <mutex>
#include "zklog.hpp"
#include "zkmax.hpp"
#include "exit_process.hpp"
#include "scalar.hpp"

DatabaseKVAssociativeCache::DatabaseKVAssociativeCache()
{
    log2IndexesSize = 0;
    indexesSize = 0;
    log2CacheSize = 0;
    cacheSize = 0;
    maxVersions = 0; 
    indexes = NULL;
    keys = NULL;
    values = NULL;
    versions = NULL;
    currentCacheIndex = 0;
    attempts = 0;
    hits = 0;
    name = "";
};

DatabaseKVAssociativeCache::DatabaseKVAssociativeCache(int log2IndexesSize_, int cacheSize_, string name_)
{
    postConstruct(log2IndexesSize_, cacheSize_, name_);
};

DatabaseKVAssociativeCache::~DatabaseKVAssociativeCache()
{
    if (indexes != NULL)
        delete[] indexes;
    if (keys != NULL)
        delete[] keys;
    if (values != NULL)
        delete[] values;
    if (versions != NULL)
        delete[] versions;

};

void DatabaseKVAssociativeCache::postConstruct(int log2IndexesSize_, int log2CacheSize_, string name_)
{
    lock_guard<recursive_mutex> guard(mlock);
    log2IndexesSize = log2IndexesSize_;
    if (log2IndexesSize_ > 32)
    {
        zklog.error("DatabaseKVAssociativeCache::DatabaseKVAssociativeCache() log2IndexesSize_ > 32");
        exitProcess();
    }
    indexesSize = 1 << log2IndexesSize;

    log2CacheSize = log2CacheSize_;
    if (log2CacheSize_ > 32)
    {
        zklog.error("DatabaseKVAssociativeCache::DatabaseKVAssociativeCache() log2CacheSize_ > 32");
        exitProcess();
    }
    cacheSize = 1 << log2CacheSize_;
    maxVersions = cacheSize;

    if(indexes != NULL) delete[] indexes;
    indexes = new uint32_t[indexesSize];
    //initialization of indexes array
    uint32_t initValue = UINT32_MAX-cacheSize-(uint32_t)1;
    #pragma omp parallel for schedule(static) num_threads(4)
    for (size_t i = 0; i < indexesSize; i++)
    {
        indexes[i] = initValue;
    }
    if(keys != NULL) delete[] keys;
    keys = new Goldilocks::Element[4 * cacheSize];

    if(values != NULL) delete[] values;
    values = new mpz_class[cacheSize];

    if(versions != NULL) delete[] versions;
    versions = new uint64_t[2 * cacheSize];
    #pragma omp parallel for schedule(static) num_threads(4)
    for (size_t i = 0; i < cacheSize; i++)
    {
        versions[i*2+1] = UINT64_MAX;
    }

    currentCacheIndex = 0;
    attempts = 0;
    hits = 0;
    name = name_;
    
    //masks for fast module, note cache size and indexes size must be power of 2
    cacheMask = cacheSize - 1;
    indexesMask = indexesSize - 1;
};

void DatabaseKVAssociativeCache::addKeyValueVersion(const uint64_t version, const Goldilocks::Element (&key)[4], const mpz_class &value){
    
    lock_guard<recursive_mutex> guard(mlock);
    bool emptySlot = false;
    bool present = false;
    uint32_t cacheIndex;
    uint32_t tableIndexUse=0;
    uint64_t cacheIndexPrev=UINT64_MAX;

    //
    // Check if present in one of the four slots
    //
    Goldilocks::Element key_hashed[4];
    hashKey(key_hashed, key);
    for (int i = 0; i < 4; ++i)
    {
        uint32_t tableIndex = (uint32_t)(key_hashed[i].fe & indexesMask);
        uint32_t cacheIndexRaw = indexes[tableIndex];
        cacheIndex = cacheIndexRaw & cacheMask;
        uint32_t cacheIndexKey = cacheIndex * 4;
        uint32_t cacheIndexVersions = cacheIndex * 2;
        uint32_t cacheIndexValue = cacheIndex;

        if (!emptyCacheSlot(cacheIndexRaw)){
            if( keys[cacheIndexKey + 0].fe == key[0].fe &&
                keys[cacheIndexKey + 1].fe == key[1].fe &&
                keys[cacheIndexKey + 2].fe == key[2].fe &&
                keys[cacheIndexKey + 3].fe == key[3].fe){
                    present = true;
                    if(versions[cacheIndexVersions] == version){
                        return; //no update
                    } else if (version < versions[cacheIndexVersions]){
                        zklog.error("DatabaseKVAssociativeCache::addKeyValueVersion() adding lower version than the current one");
                        exitProcess();
                    } else {
                        if(value==values[cacheIndexValue]){
                            return;
                        }
                    }

                    tableIndexUse = tableIndex;
                    cacheIndexPrev = cacheIndex;
                    break;
            }
        }else if (emptySlot == false){
            emptySlot = true;
            tableIndexUse = tableIndex;
        }
    }

    //
    // Evaluate cacheIndexKey and 
    //
    if(emptySlot == true || present){
        indexes[tableIndexUse] = currentCacheIndex;
    }
    cacheIndex = (uint32_t)(currentCacheIndex & cacheMask);
    currentCacheIndex = (currentCacheIndex == UINT32_MAX) ? 0 : (currentCacheIndex + 1);
    
    uint64_t cacheIndexKey, cacheIndexValue, cacheIndexVersions;
    cacheIndexKey = cacheIndex * 4;
    cacheIndexValue = cacheIndex;
    cacheIndexVersions = cacheIndex * 2;
    
    //
    // Add value
    //
    keys[cacheIndexKey + 0].fe = key[0].fe;
    keys[cacheIndexKey + 1].fe = key[1].fe;
    keys[cacheIndexKey + 2].fe = key[2].fe;
    keys[cacheIndexKey + 3].fe = key[3].fe;
    values[cacheIndexValue] = value;
    versions[cacheIndexVersions] = version;
    if(present){
        versions[cacheIndexVersions+1] = cacheIndexPrev;
    }else{
        versions[cacheIndexVersions+1] = UINT64_MAX;
    }
    //
    // Forced index insertion
    //
    if(!present && !emptySlot){
        int iters = 0;
        uint32_t usedRawCacheIndexes[10];
        usedRawCacheIndexes[0] = currentCacheIndex-1;
        forcedInsertion(usedRawCacheIndexes, iters);
    }
}

void DatabaseKVAssociativeCache::downstreamAddKeyZeroVersion(const uint64_t version, const Goldilocks::Element (&key)[4]){

    lock_guard<recursive_mutex> guard(mlock);
    bool presentSameVersion = false;
    uint32_t cacheIndexZero = 0;
    //
    // Check if present in one of the four slots
    //
    Goldilocks::Element key_hashed[4];
    hashKey(key_hashed, key);
    bool breakOuterLoop = false;
    for (int i = 0; i < 4; ++i)
    {
        uint32_t tableIndex = (uint32_t)(key_hashed[i].fe & indexesMask);
        uint32_t cacheIndexRaw = indexes[tableIndex];
        uint32_t cacheIndex = cacheIndexRaw & cacheMask;
        uint32_t cacheIndexKey = cacheIndex * 4;
        uint32_t cacheIndexVersions = cacheIndex * 2;

        if (!emptyCacheSlot(cacheIndexRaw)){
            for(int j=0; j<maxVersions; j++){
                if (keys[cacheIndexKey + 0].fe == key[0].fe &&
                    keys[cacheIndexKey + 1].fe == key[1].fe &&
                    keys[cacheIndexKey + 2].fe == key[2].fe &&
                    keys[cacheIndexKey + 3].fe == key[3].fe){ 
                    
                        if( versions[cacheIndexVersions] == version &&
                            versions[cacheIndexVersions+1] == UINT64_MAX){ //if linked zero can not be added){

                            presentSameVersion = true;
                            cacheIndexZero = currentCacheIndex & cacheMask;
                            currentCacheIndex = (currentCacheIndex == UINT32_MAX) ? 0 : (currentCacheIndex + 1);
                            versions[cacheIndexVersions+1] = cacheIndexZero;
                            breakOuterLoop = true;
                            break;
                        }
                        if(versions[cacheIndexVersions+1]==UINT64_MAX) return; //No more versions
                        cacheIndex = versions[cacheIndexVersions+1] & cacheMask;
                        cacheIndexKey = cacheIndex * 4;
                        cacheIndexVersions = cacheIndex * 2;
                    
                }else{
                    if(j>0){
                        return;
                    }else{
                        break;
                    }
                }
            }
        }
        if(breakOuterLoop) break;
    }

    //
    // Evaluate cacheIndexKey and 
    //
    if(presentSameVersion){
    
        uint64_t cacheIndexKey = cacheIndexZero * 4;
        uint64_t cacheIndexValue = cacheIndexZero;
        uint64_t cacheIndexVersions = cacheIndexZero * 2;
        
        //
        // Add value
        //
        keys[cacheIndexKey + 0].fe = key[0].fe;
        keys[cacheIndexKey + 1].fe = key[1].fe;
        keys[cacheIndexKey + 2].fe = key[2].fe;
        keys[cacheIndexKey + 3].fe = key[3].fe;
        values[cacheIndexValue] = 0;
        versions[cacheIndexVersions] = 0;
        versions[cacheIndexVersions+1] = UINT64_MAX;
    }
    return;

}

void DatabaseKVAssociativeCache::uploadKeyValueVersions(const Goldilocks::Element (&key)[4], vector<VersionValue> &versionsValues){
    
    lock_guard<recursive_mutex> guard(mlock);
    //Get last version for the key
    vector<uint64_t> versions;
    getLastCachedVersions(key, versions, versionsValues.size());
    if(versions.size()==0){
        for (std::vector<VersionValue>::reverse_iterator it = versionsValues.rbegin(); it != versionsValues.rend(); ++it) {
            addKeyValueVersion(it->version, key, it->value);
        }
    }else{
        vector<VersionValue>::const_iterator it = versionsValues.begin();
        for(vector<uint64_t>::const_iterator it2 = versions.begin(); it2 != versions.end(); ++it2, ++it){
            if(*it2 != it->version){
                zklog.error("DatabaseKVAssociativeCache::uploadKeyValueVersions() versions mismatch between cache and db");
                exitProcess();
            }
        }
        //If vector.zize() < versionsValues.size() we could add some backward versions in the cache but not supported yet
    }
}

void DatabaseKVAssociativeCache::getLastCachedVersions(const Goldilocks::Element (&key)[4], vector<uint64_t> &versions, const int maxVersionsOut){
    
    versions.clear();
    lock_guard<recursive_mutex> guard(mlock);
    //
    // Find the value
    //
    Goldilocks::Element key_hashed[4];
    hashKey(key_hashed, key);
    for (int i = 0; i < 4; i++)
    {
        uint32_t cacheIndexRaw = indexes[key_hashed[i].fe & indexesMask];
        if (emptyCacheSlot(cacheIndexRaw)) continue;
        
        uint32_t cacheIndex = cacheIndexRaw  & cacheMask;
        uint32_t cacheIndexKey = cacheIndex * 4;
        uint32_t cacheIndexVersions = cacheIndex * 2;

        for(int j=0; j<maxVersionsOut; j++){
            if (keys[cacheIndexKey + 0].fe == key[0].fe &&
                keys[cacheIndexKey + 1].fe == key[1].fe &&
                keys[cacheIndexKey + 2].fe == key[2].fe &&
                keys[cacheIndexKey + 3].fe == key[3].fe){

                versions.push_back(versions[cacheIndexVersions]);
                if(versions[cacheIndexVersions+1]==UINT64_MAX) return; //No more versions
                cacheIndex = versions[cacheIndexVersions+1] & cacheMask;
                cacheIndexKey = cacheIndex * 4;
                cacheIndexVersions = cacheIndex * 2;
                
            }else{
                if(j>0){
                    return;
                }else{
                    break;
                }
            }
        }
    }
    return;
    
}

void DatabaseKVAssociativeCache::forcedInsertion(uint32_t (&usedRawCacheIndexes)[10], int &iters)
{
    uint32_t inputRawCacheIndex = usedRawCacheIndexes[iters];
    //
    // avoid infinite loop
    //
    iters++;
    if (iters > 9)
    {
        zklog.error("forcedInsertion() more than 10 iterations required. Index: " + to_string(inputRawCacheIndex));
        exitProcess();
    }    
    //
    // find a slot into my indexes
    //
    Goldilocks::Element key_hashed[4];
    hashKey(key_hashed, (Goldilocks::Element(&)[4])keys[(inputRawCacheIndex & cacheMask) * 4]);
    Goldilocks::Element *inputKey = &key_hashed[0];
    uint32_t minRawCacheIndex = UINT32_MAX;
    int pos = -1;

    for (int i = 0; i < 4; ++i)
    {
        uint32_t tableIndex_ = (uint32_t)(inputKey[i].fe & indexesMask);
        uint32_t rawCacheIndex_ = (uint32_t)(indexes[tableIndex_]);
        if (emptyCacheSlot(rawCacheIndex_))
        {
            indexes[tableIndex_] = inputRawCacheIndex;
            return;
        }
        else
        {
            //consider minimum not used rawCacheIndex_
            bool used = false;
            for(int k=0; k<iters; k++){
                if(usedRawCacheIndexes[k] == rawCacheIndex_){
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
        zklog.error("forcedInsertion() could not continue the recursion: " + to_string(inputRawCacheIndex));
        exitProcess();
    } 
    indexes[(uint32_t)(inputKey[pos].fe & indexesMask)] = inputRawCacheIndex;
    usedRawCacheIndexes[iters] = minRawCacheIndex; //new cache element to add in the indexes table
    forcedInsertion(usedRawCacheIndexes, iters);
    
}

bool DatabaseKVAssociativeCache::findKey( const uint64_t version, const Goldilocks::Element (&key)[4], mpz_class &value)
{
    lock_guard<recursive_mutex> guard(mlock);
    attempts++; 
    //
    //  Statistics
    //
    if (attempts<<40 == 0)
    {
        zklog.info("DatabaseKVAssociativeCache::findKey() name=" + name + " indexesSize=" + to_string(indexesSize) + " cacheSize=" + to_string(cacheSize) + " attempts=" + to_string(attempts) + " hits=" + to_string(hits) + " hit ratio=" + to_string(double(hits) * 100.0 / double(zkmax(attempts, 1))) + "%");
    }
    //
    // Find the value
    //
    Goldilocks::Element key_hashed[4];
    hashKey(key_hashed, key);
    for (int i = 0; i < 4; i++)
    {
        uint32_t cacheIndexRaw = indexes[key_hashed[i].fe & indexesMask];
        if (emptyCacheSlot(cacheIndexRaw)) continue;
        
        uint32_t cacheIndex = cacheIndexRaw  & cacheMask;
        uint32_t cacheIndexKey = cacheIndex * 4;
        uint32_t cacheIndexVersions = cacheIndex * 2;

        for(int j=0; j<maxVersions; j++){
            if (keys[cacheIndexKey + 0].fe == key[0].fe &&
                keys[cacheIndexKey + 1].fe == key[1].fe &&
                keys[cacheIndexKey + 2].fe == key[2].fe &&
                keys[cacheIndexKey + 3].fe == key[3].fe){

                if( version >= versions[cacheIndexVersions] ){
                    uint32_t cacheIndexValue = cacheIndex;
                    ++hits;
                    value = values[cacheIndexValue];
                    return true;
                }
                if(versions[cacheIndexVersions+1]==UINT64_MAX) return false; //No more versions
                cacheIndex = versions[cacheIndexVersions+1] & cacheMask;
                cacheIndexKey = cacheIndex * 4;
                cacheIndexVersions = cacheIndex * 2;
                
            }else{
                if(j>0){
                    return false;
                }else{
                    break;
                }
            }
        }
    }
    return false;
}
