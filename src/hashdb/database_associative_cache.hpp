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
        uint32_t currentCacheIndex; 

        uint64_t attempts;
        uint64_t hits;
        string name;

        uint32_t indexesMask;
        uint32_t cacheMask;

        vector<Goldilocks::Element> auxBufferKeysValues;

    public:

        DatabaseMTAssociativeCache();
        DatabaseMTAssociativeCache(uint32_t log2IndexesSize_, uint32_t log2CacheSize_, string name_);
        ~DatabaseMTAssociativeCache();
        void postConstruct(uint32_t log2IndexesSize_, uint32_t log2CacheSize_, string name_);
        
        inline void addKeyValue(const Goldilocks::Element (&key)[4], const vector<Goldilocks::Element> &value, bool update);
        inline bool findKey(const Goldilocks::Element (&key)[4], vector<Goldilocks::Element> &value);

        inline bool enabled() const { return (log2IndexesSize > 0); };
        inline uint32_t getCacheSize()  const { return cacheSize; };
        inline uint32_t getIndexesSize() const { return indexesSize; };
        inline uint32_t getAuxBufferKeysValuesSize() const { return auxBufferKeysValues.size(); };
        inline void clear();

    private:
        bool findKey_(const Goldilocks::Element (&key)[4], vector<Goldilocks::Element> &value, bool &reinsert);
        void addKeyValue_(const Goldilocks::Element (&key)[4], const vector<Goldilocks::Element> &value, bool update);
        bool extractKeyValue_(const Goldilocks::Element (&key)[4], vector<Goldilocks::Element> &value);
        
        inline bool isEmptySlot(uint32_t cacheIndexRaw) const { 
            return  !isValidKey[cacheIndexRaw & cacheMask] || distanceToCurrentCacheIndex(cacheIndexRaw) > cacheSize ;
         };
        inline uint32_t distanceToCurrentCacheIndex(uint32_t cacheIndexRaw) const {
            if(currentCacheIndex == cacheIndexRaw) return UINT32_MAX; //it should be UINT32_MAX + 1 but is out of range
            return (currentCacheIndex > cacheIndexRaw) ? currentCacheIndex - cacheIndexRaw : UINT32_MAX - cacheIndexRaw + currentCacheIndex + 1;
         };
        void forcedInsertion(uint32_t (&usedRawCacheIndexes)[20], uint32_t &iters, bool update);
        inline void cleanAuxBufferKeysValues();
};

void DatabaseMTAssociativeCache::addKeyValue(const Goldilocks::Element (&key)[4], const vector<Goldilocks::Element> &value, bool update){
    //This wrapper is used to aviod using the lock_guard inside the addKeyValue_ function
    shared_lock<shared_mutex> lock(mlock);
    addKeyValue_(key, value, update);
}
bool DatabaseMTAssociativeCache::findKey(const Goldilocks::Element (&key)[4], vector<Goldilocks::Element> &value){
    bool reinsert=false;
    bool found=false;
    {
        shared_lock<shared_mutex> lock(mlock);
        found = findKey_(key, value, reinsert);
    }
    if(reinsert){
        unique_lock<shared_mutex> lock(mlock);
        vector<Goldilocks::Element> values_;
        bool found = extractKeyValue_(key, values_);
        //I retrive the values againg (values_) to prevent situations in which the values could have been modified by another thread betxeen the findKey_ and the addKeyValue_
        if(found) addKeyValue_(key, values_, false);
    }

#ifdef LOG_CACHE
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
void DatabaseMTAssociativeCache::clear(){
    memset(isValidKey, 0, cacheSize * sizeof(bool));
    auxBufferKeysValues.clear();
    currentCacheIndex = 0;
}
void DatabaseMTAssociativeCache::cleanAuxBufferKeysValues(){
            auto it = auxBufferKeysValues.begin();
            while (it < auxBufferKeysValues.end()) {
                if (isEmptySlot(static_cast<uint32_t>(it->fe))) {
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
        };
#endif