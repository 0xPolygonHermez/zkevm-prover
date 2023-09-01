#ifndef DATABASE_KV_ASSOCIATIVE_CACHE_HPP
#define DATABASE_KV_ASSOCIATIVE_CACHE_HPP
#include <vector>
#include "goldilocks_base_field.hpp"
#include <nlohmann/json.hpp>
#include <mutex>
#include "zklog.hpp"
#include "zkmax.hpp"

using namespace std;
class DatabaseKVAssociativeCache
{
    private:
        recursive_mutex mlock;

        int log2IndexesSize;
        uint32_t indexesSize;
        int log2CacheSize;
        uint32_t cacheSize;
        int maxVersions;

        uint32_t *indexes;
        Goldilocks::Element *keys;
        uint64_t *versions;
        mpz_class *values; //rick: this I do not like
        uint32_t currentCacheIndex; 

        uint64_t attempts;
        uint64_t hits;
        string name;

        uint64_t indexesMask;
        uint64_t cacheMask;


    public:

        DatabaseKVAssociativeCache();
        DatabaseKVAssociativeCache(int log2IndexesSize_, int log2CacheSize_, string name_);
        ~DatabaseKVAssociativeCache();
        void postConstruct(int log2IndexesSize_, int log2CacheSize_, string name_);

        void addKeyValueVersion(const uint64_t version, const Goldilocks::Element (&key)[4], const mpz_class &value, bool update);
        bool findKey( const uint64_t version, const Goldilocks::Element (&key)[4], mpz_class &value);

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