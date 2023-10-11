#ifndef KEY_VALUE_HISTORY_PAGE_HPP
#define KEY_VALUE_HISTORY_PAGE_HPP

#include <unistd.h>
#include "zkresult.hpp"
#include "zkassert.hpp"
#include "scalar.hpp"
#include "hash_value_gl.hpp"

struct KeyValueHistoryStruct
{
    uint64_t hashPage1AndHistoryCounter; // historyCounter (2B) + hashPage1 (6B)
    uint64_t hashPage2AndPadding; // padding (2B) + hashPage2 (6B)
    uint64_t keyValueEntry[64][2];
        // [0] = rawDataOffset (2B) + rawDataPage (6B) (if leaf node, location of key(32B) + value(32B)), or nextPageNumber (6B) (if intermediate node, page with level+1)
        // [1] = control (4bits) + previousVersionOffset (12bits) + version (6B)
        // If control == 0, then this entry is empty, i.e. value = 0
        // If control == 1, then this entry contains a version (leaf node)
        // If control == 2, then this entry contains the page number of the next level page (intermediate node)
    uint64_t historyEntry[191][2]; // history entries    
};

class KeyValueHistoryPage
{
    static const uint64_t minVersionOffset = 8 + 8 + 64*16;
    static const uint64_t maxVersionOffset = 4096;
private:
    static zkresult Read          (const uint64_t pageNumber,  const string &key, const string &keyBits, const uint64_t version,       mpz_class &value, const uint64_t level);
    static zkresult Write         (      uint64_t &pageNumber, const string &key, const string &keyBits, const uint64_t version, const mpz_class &value, const uint64_t level, const uint64_t headerPageNumber);
public:

    static zkresult InitEmptyPage (const uint64_t pageNumber);
    static zkresult Read          (const uint64_t pageNumber,  const string &key, const uint64_t version,       mpz_class &value);
    static zkresult Write         (      uint64_t &pageNumber, const string &key, const uint64_t version, const mpz_class &value, const uint64_t headerPageNumber);
    
    static void calculateLeafHash         (const Goldilocks::Element (&key)[4], const uint64_t level, const mpz_class &value, Goldilocks::Element (&hash)[4], vector<HashValueGL> *hashValues);
    static void calculateIntermediateHash (const Goldilocks::Element (&leftHash)[4], const Goldilocks::Element (&rightHash)[4], Goldilocks::Element (&hash)[4], vector<HashValueGL> *hashValues);

    static void Print (const uint64_t pageNumber, bool details);
};

#endif