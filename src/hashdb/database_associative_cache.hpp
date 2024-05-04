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

        uint32_t *indexes;
        Goldilocks::Element *keys;
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
        void addKeyValue(Goldilocks::Element (&key)[4], const vector<Goldilocks::Element> &value, bool update);
        bool findKey(const Goldilocks::Element (&key)[4], vector<Goldilocks::Element> &value);
        inline bool enabled() const { return (log2IndexesSize > 0); };
        inline uint32_t getCacheSize()  const { return cacheSize; };
        inline uint32_t getIndexesSize() const { return indexesSize; };
        inline uint32_t getAuxBufferKeysValuesSize() const { return auxBufferKeysValues.size(); };
        inline void clear(){
            if(enabled()){
                postConstruct(log2IndexesSize, log2CacheSize, name);                
            }
        }

    private:
        inline bool emptyCacheSlot(uint32_t cacheIndexRaw) const { 
            return (currentCacheIndex > cacheIndexRaw &&  currentCacheIndex - cacheIndexRaw > cacheSize) ||
            (currentCacheIndex <= cacheIndexRaw && UINT32_MAX - cacheIndexRaw + currentCacheIndex > cacheSize - 1);
         };
        void forcedInsertion(uint32_t (&usedRawCacheIndexes)[20], uint32_t &iters, bool update);
        inline void cleanAuxBufferKeysValues(){
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
        };
};
#endif