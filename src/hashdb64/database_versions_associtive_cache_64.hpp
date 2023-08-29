#ifndef DATABASE_VERSIONS_ASSOCIATIVE_CACHE_HPP
#define DATABASE_VERSIONS_ASSOCIATIVE_CACHE_HPP
#include <vector>
#include "goldilocks_base_field.hpp"
#include <nlohmann/json.hpp>
#include <mutex>
#include "zklog.hpp"
#include "zkmax.hpp"

using namespace std;
class DatabaseVersionsAssociativeCache
{
    private:
        recursive_mutex mlock;

        int log2IndexesSize;
        uint32_t indexesSize;
        int log2CacheSize;
        uint32_t cacheSize;

        uint32_t *indexes;
        Goldilocks::Element *keys;
        uint64_t *versions;
        uint32_t currentCacheIndex; 

        uint64_t attempts;
        uint64_t hits;
        string name;

        uint64_t indexesMask;
        uint64_t cacheMask;


    public:

        DatabaseVersionsAssociativeCache();
        DatabaseVersionsAssociativeCache(int log2IndexesSize_, int log2CacheSize_, string name_);
        ~DatabaseVersionsAssociativeCache();
        void postConstruct(int log2IndexesSize_, int log2CacheSize_, string name_);

        void addKeyVersion(Goldilocks::Element (&key)[4], const uint64_t version, const bool update);
        bool findKey(const Goldilocks::Element (&key)[4], uint64_t &version);
        
        inline bool enabled() const { return (log2IndexesSize > 0); };
        inline uint32_t getCacheSize()  const { return cacheSize; };
        inline uint32_t getIndexesSize() const { return indexesSize; };

    private:
        
        inline bool emptyCacheSlot(uint32_t cacheIndexRaw) const { 
            return (currentCacheIndex >= cacheIndexRaw &&  currentCacheIndex - cacheIndexRaw > cacheSize) ||
            (currentCacheIndex < cacheIndexRaw && UINT32_MAX - cacheIndexRaw + currentCacheIndex > cacheSize);
         };
        void forcedInsertion(uint32_t (&usedRawCacheIndexes)[10], int &iters);
};
#endif

