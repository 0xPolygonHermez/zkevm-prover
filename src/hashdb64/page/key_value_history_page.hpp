#ifndef KEY_VALUE_HISTORY_PAGE_HPP
#define KEY_VALUE_HISTORY_PAGE_HPP

#include <unistd.h>
#include "zkresult.hpp"
#include "zkassert.hpp"
#include "scalar.hpp"
#include "hash_value_gl.hpp"
#include "key_value.hpp"
#include "tree_chunk.hpp"
#include "page_context.hpp"


struct KeyValueHistoryStruct
{
    uint64_t historyOffset; // Offset of the next history entry that is free, or maxHistoryCounter
    uint64_t previousPage; // Number of the previous page
    uint64_t keyValueEntry[64][3];
        // [0] = control (4bits) + previousVersionOffset (12bits) + version (6B)
        // [1] = if leaf node, location of key(32B) + value(32B) = rawDataOffset (2B) + rawDataPage (6B)
        // [1] = if intermediate node, page with level+1 = nextPageNumber (6B)
        // [2] = if leaf or intermediate node, location of hash(32B) = rawDataOffset (2B) + rawDataPage (6B), or 0 if hash needs to be calculated (e.g. key was moved to a deeper level page)
        // If control == 0, then this entry is empty, i.e. value = 0
        // If control == 1, then this entry contains a version (leaf node)
        // If control == 2, then this entry contains the page number of the next level page (intermediate node)
    uint64_t historyEntry[106][3]; // history entries, copied from keyValueEntry when a new version is written in this page
};

class KeyValueHistoryCounters
{
public:
    uint64_t intermediateNodes;
    uint64_t leafNodes;
    uint64_t maxLevel;
    uint64_t intermediateHashes;
    uint64_t leafHashes;
    KeyValueHistoryCounters() : intermediateNodes(0), leafNodes(0), maxLevel(0), intermediateHashes(0), leafHashes(0) {};
};

class KeyValueHistoryPage
{
public:
    static const uint64_t entrySize = 3*8; // 24B
    static const uint64_t minHistoryOffset = 8 + 8 + 64*3*8; // 1552
    static const uint64_t maxHistoryOffset = 8 + 8 + 64*3*8 + 106*3*8; // 4096
private:
    static zkresult Read          (PageContext &ctx, const uint64_t pageNumber,  const string &key, const string &keyBits, const uint64_t version,       mpz_class &value, const uint64_t level, uint64_t &keyLevel);
    static zkresult ReadLevel     (PageContext &ctx, const uint64_t pageNumber,  const string &key, const string &keyBits,                                                 const uint64_t level, uint64_t &keyLevel);
    static zkresult ReadTree      (PageContext &ctx, const uint64_t pageNumber,  const string &key, const string &keyBits, const uint64_t version,       mpz_class &value, vector<HashValueGL> *hashValues, const uint64_t level, unordered_map<uint64_t, TreeChunk> &treeChunkMap);
    static zkresult Write         (PageContext &ctx,       uint64_t &pageNumber, const string &key, const string &keyBits, const uint64_t version, const mpz_class &value, const uint64_t level, uint64_t &headerPageNumber);
public:
    static zkresult InitEmptyPage (PageContext &ctx, const uint64_t pageNumber);
    static zkresult Read          (PageContext &ctx, const uint64_t pageNumber,  const string &key, const uint64_t version,       mpz_class &value, uint64_t &keyLevel);
    static zkresult ReadLevel     (PageContext &ctx, const uint64_t pageNumber,  const string &key,                                                 uint64_t &keyLevel);
    static zkresult ReadTree      (PageContext &ctx, const uint64_t pageNumber,  const uint64_t version,  vector<KeyValue> &keyValues, vector<HashValueGL> *hashValues);
    static zkresult Write         (PageContext &ctx,       uint64_t &pageNumber, const string &key, const uint64_t version, const mpz_class &value, uint64_t &headerPageNumber);
    
    static zkresult calculateHash             (PageContext &ctx, uint64_t &pageNumber, Goldilocks::Element (&hash)[4], uint64_t &headerPageNumber);
private:
    static zkresult calculatePageHash         (PageContext &ctx, uint64_t &pageNumber, const uint64_t level, Goldilocks::Element (&hash)[4], uint64_t &headerPageNumber);
public:
    static void Print (PageContext &ctx, const uint64_t pageNumber, bool details, const string &prefix, const uint64_t level, KeyValueHistoryCounters &counters);
    static void Print (PageContext &ctx, const uint64_t pageNumber, bool details, const string &prefix);

};

#endif