#ifndef DATABASE_ASSOCIATIVE_CACHE_HPP
#define DATABASE_ASSOCIATIVE_CACHE_HPP
#include <vector>
#include "goldilocks_base_field.hpp"
#include <nlohmann/json.hpp>
#include <mutex>
#include "zklog.hpp"
#include "zkmax.hpp"

using namespace std;
using json = nlohmann::json;
class DatabaseMTAssociativeCache
{
    private:
        recursive_mutex mlock;

        int nKeyBits;
        uint32_t indexesSize;
        int log2CacheSize;
        uint32_t cacheSize;

        uint32_t *indexes;
        Goldilocks::Element *keys;
        Goldilocks::Element *values;
        bool *isLeaf;
        uint32_t currentCacheIndex; 

        uint64_t attempts;
        uint64_t hits;
        string name;

        uint64_t indexesMask;
        uint64_t cacheMask;


    public:

        DatabaseMTAssociativeCache();
        DatabaseMTAssociativeCache(int nKeyBits_, int log2CacheSize_, string name_);
        ~DatabaseMTAssociativeCache();

        void postConstruct(int nKeyBits_, int log2CacheSize_, string name_);
        void addKeyValue(Goldilocks::Element (&key)[4], const vector<Goldilocks::Element> &value, bool update);
        bool findKey(Goldilocks::Element (&key)[4], vector<Goldilocks::Element> &value);
        inline bool enabled() const { return (nKeyBits > 0); };
        inline uint32_t getCacheSize()  const { return cacheSize; };
        inline uint32_t getIndexesSize() const { return indexesSize; };

    private:
        void forcedInsertion(uint32_t (&rawCacheIndexes)[10], int &iters);
};
#endif

// TODO:
// 5. Put attempts and hits in a higher level
