#ifndef KEY_VALUE_HISTORY_PAGE_HPP
#define KEY_VALUE_HISTORY_PAGE_HPP

#include <unistd.h>
#include "zkresult.hpp"
#include "zkassert.hpp"
#include "scalar.hpp"
#include "hash_value_gl.hpp"

struct KeyValueHistoryStruct
{
    uint64_t historyOffset; // Offset of the next history entry that is free, or maxHistoryCounter
    uint64_t previousPage; // Number of the previous page
    uint64_t keyValueEntry[64][3];
        // [0] = control (4bits) + previousVersionOffset (12bits) + version (6B)
        // [1] = if leaf node, location of key(32B) + value(32B) = rawDataOffset (2B) + rawDataPage (6B)
        // [1] = if intermediate node, page with level+1 = nextPageNumber (6B)
        // [2] = if leaf or intermediate node, location of hash(32B) = rawDataOffset (2B) + rawDataPage (6B), or 0 if has needs to be calculated
        // If control == 0, then this entry is empty, i.e. value = 0
        // If control == 1, then this entry contains a version (leaf node)
        // If control == 2, then this entry contains the page number of the next level page (intermediate node)
    uint64_t historyEntry[106][3]; // history entries, copied from keyValueEntry when a new version is written in this page
};

class KeyValueHistoryPage
{
private:
    static const uint64_t entrySize = 3*8; // 24B
    static const uint64_t minHistoryOffset = 8 + 8 + 64*3*8; // 1552
    static const uint64_t maxHistoryOffset = 8 + 8 + 64*3*8 + 106*3*8; // 4096
private:
    static zkresult Read          (const uint64_t pageNumber,  const string &key, const string &keyBits, const uint64_t version,       mpz_class &value, const uint64_t level, uint64_t &keyLevel);
    static zkresult ReadLevel     (const uint64_t pageNumber,  const string &key, const string &keyBits,                                                 const uint64_t level, uint64_t &keyLevel);
    static zkresult Write         (      uint64_t &pageNumber, const string &key, const string &keyBits, const uint64_t version, const mpz_class &value, const uint64_t level, uint64_t &headerPageNumber);
public:
    static zkresult InitEmptyPage (const uint64_t pageNumber);
    static zkresult Read          (const uint64_t pageNumber,  const string &key, const uint64_t version,       mpz_class &value, uint64_t &keyLevel);
    static zkresult ReadLevel     (const uint64_t pageNumber,  const string &key,                                                 uint64_t &keyLevel);
    static zkresult Write         (      uint64_t &pageNumber, const string &key, const uint64_t version, const mpz_class &value, uint64_t &headerPageNumber);
    
    static zkresult calculateHash             (const uint64_t pageNumber, Goldilocks::Element (&hash)[4], uint64_t &headerPageNumber);
private:
    static zkresult calculatePageHash         (const uint64_t pageNumber, const uint64_t level, Goldilocks::Element (&hash)[4], uint64_t &headerPageNumber);
    //static void calculateLeafHash         (const Goldilocks::Element (&key)[4], const uint64_t level, const mpz_class &value, Goldilocks::Element (&hash)[4], vector<HashValueGL> *hashValues);
    //static void calculateIntermediateHash (const Goldilocks::Element (&leftHash)[4], const Goldilocks::Element (&rightHash)[4], Goldilocks::Element (&hash)[4], vector<HashValueGL> *hashValues);
public:
    static void Print (const uint64_t pageNumber, bool details, const string &prefix);
};

#endif