#ifndef DATABASE_ASSOCIATIVE_CACHE_2_HPP
#define DATABASE_ASSOCIATIVE_CACHE_2_HPP
#include <vector>
#include "goldilocks_base_field.hpp"
#include <nlohmann/json.hpp>
#include <mutex>
#include "zklog.hpp"
#include "zkmax.hpp"

using namespace std;
using json = nlohmann::json;
class DatabaseMTAssociativeCache2
{
    private:
        recursive_mutex mlock;

        int nKeyBits;
        int indicesSize;
        uint32_t cacheSize;

        uint32_t *indices;
        Goldilocks::Element *keys;
        Goldilocks::Element *values;
        bool *isLeaf;
        uint32_t currentCacheIndex; 

        uint64_t attempts; // punt in a higher level!
        uint64_t hits;
        string name;

        uint64_t indicesMask;

    public:

        DatabaseMTAssociativeCache2();
        DatabaseMTAssociativeCache2(int nKeyBits_, int cacheSize_, string name_);
        ~DatabaseMTAssociativeCache2();

        void postConstruct(int nKeyBits_, int cacheSize_, string name_);
        void addKeyValue(Goldilocks::Element (&key)[4], const vector<Goldilocks::Element> &value, bool update = true);
        bool findKey(Goldilocks::Element (&key)[4], vector<Goldilocks::Element> &value);
        inline bool enabled() { return (nKeyBits > 0); };

    private:
        void forcedInsertion(uint32_t cacheIndex, int &iters);
};
#endif

// TODO:
// 5. vull carregar-me attempts y hits
