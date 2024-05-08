#ifndef DATABASE_ASSOCIATIVE_CACHE_HPP
#define DATABASE_ASSOCIATIVE_CACHE_HPP
#include <vector>
#include "goldilocks_base_field.hpp"
#include <nlohmann/json.hpp>
#include <mutex>
#include <shared_mutex>
#include "zklog.hpp"
#include "zkmax.hpp"

using namespace std;
class DatabaseMTAssociativeCache
{
    private:
        shared_mutex mlock;
        
        uint32_t log2IndexesSize;
        uint32_t indexesSize;
        uint32_t log2CacheSize;
        uint32_t cacheSize;
        uint32_t cacheSizeDiv2;

        uint32_t *indexes;
        Goldilocks::Element *keys;
        bool *isValidKey;
        Goldilocks::Element *values;
        uint32_t currentCacheIndex; //next index to be used (acts as a clock for the cache)

#ifdef LOG_ASSOCIATIVE_CACHE
        uint64_t attempts;
        uint64_t hits;
#endif
        string name;

        uint32_t indexesMask;
        uint32_t cacheMask;

        vector<Goldilocks::Element> auxBufferKeysValues;
        // The buffer uses 17 slots for each key-value pair:
        // 1 for the raw cache index, 4 for the key, 12 for the value

    public:

        DatabaseMTAssociativeCache();
        DatabaseMTAssociativeCache(uint32_t log2IndexesSize_, uint32_t log2CacheSize_, string name_ = "associative_cache");
        DatabaseMTAssociativeCache(uint64_t cacheBytes_, string name_ = "associative_cache");
        ~DatabaseMTAssociativeCache();
        void postConstruct(uint32_t log2IndexesSize_, uint32_t log2CacheSize_, string name_= "associative_cache");
        void postConstruct(uint64_t cacheBytes_, string name_ = "associative_cache");
        void clear();
        
        inline void addKeyValue(const Goldilocks::Element (&key)[4], const vector<Goldilocks::Element> &value, bool update);
        inline bool findKey(const Goldilocks::Element (&key)[4], vector<Goldilocks::Element> &value);

        inline bool enabled() const { return (log2IndexesSize > 0); };
        inline uint32_t getCacheSize()  const { return cacheSize; };
        inline uint32_t getIndexesSize() const { return indexesSize; };
        inline uint32_t getAuxBufferKeysValuesSize() const { return auxBufferKeysValues.size(); };

    private:
        void addKeyValue_(const Goldilocks::Element (&key)[4], const vector<Goldilocks::Element> &value, bool update);
        bool findKey_(const Goldilocks::Element (&key)[4], vector<Goldilocks::Element> &value, bool &reinsert);
        bool extractKeyValue_(const Goldilocks::Element (&key)[4], vector<Goldilocks::Element> &value);
        bool extractKeyValueFromAuxBuffer_(const Goldilocks::Element (&key)[4], vector<Goldilocks::Element> &value);
        
         inline bool hasExpired_(uint32_t cacheIndexRaw) const { 
            return  !isValidKey[cacheIndexRaw & cacheMask] || distanceFromCurrentCacheIndex_(cacheIndexRaw) > cacheSize ;
         };
        inline uint32_t distanceFromCurrentCacheIndex_(uint32_t cacheIndexRaw) const {
            //note: currentCacheIndex is the next index to be used (not used yet)
            if(currentCacheIndex == cacheIndexRaw) return UINT32_MAX; //it should be UINT32_MAX + 1 but is out of range
            return (currentCacheIndex > cacheIndexRaw) ? currentCacheIndex - cacheIndexRaw : UINT32_MAX - cacheIndexRaw + currentCacheIndex + 1;
         };
        void forcedInsertion_(uint32_t (&usedRawCacheIndexes)[20], uint32_t &iters, bool update);
        inline void cleanExpiredAuxBufferKeysValues_();

};

void DatabaseMTAssociativeCache::addKeyValue(const Goldilocks::Element (&key)[4], const vector<Goldilocks::Element> &value, bool update){
    // This wrapper is used to avoid the necessity of using lock_guard within the addKeyValue_ function
    mlock.lock();
    addKeyValue_(key, value, update);
    mlock.unlock();
}
bool DatabaseMTAssociativeCache::findKey(const Goldilocks::Element (&key)[4], vector<Goldilocks::Element> &value){
    bool reinsert=false;
    bool found=false;
    
    mlock.lock_shared();
    found = findKey_(key, value, reinsert);
    mlock.unlock_shared();

if(reinsert){
        mlock.lock();
        vector<Goldilocks::Element> values_;
        bool foundAgain = extractKeyValue_(key, values_);
        // Retrieved the values again (values_) to prevent potential modifications 
        // by another thread between the findKey_ and the addKeyValue_ operations
        // update = true of false is the same since it will not be found...
        if(foundAgain) addKeyValue_(key, values_, true);
        mlock.unlock();

    }

#ifdef LOG_ASSOCIATIVE_CACHE
    mlock.lock();
    attempts++; 
    if(found){
        hits++;
    }
    mlock.unlock();
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
#endif

    return found;
}
void DatabaseMTAssociativeCache::cleanExpiredAuxBufferKeysValues_() {
    if(auxBufferKeysValues.size() == 0) return;
    if(auxBufferKeysValues.size() % 17!= 0) {
        zklog.error("DatabaseMTAssociativeCache::cleanExpiredAuxBufferKeysValues_() auxBufferKeysValues.size() % 17!= 0");
        return;
    }
    auto it = auxBufferKeysValues.begin();
    while (it != auxBufferKeysValues.end()) {
        uint32_t cacheIndexRaw = static_cast<uint32_t>(it->fe);
        if (distanceFromCurrentCacheIndex_(cacheIndexRaw) > cacheSize) {
            it = auxBufferKeysValues.erase(it, it + 17);
        } else {
            std::advance(it, 17);
        }
    }
}
#endif