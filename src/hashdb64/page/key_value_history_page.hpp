#ifndef KEY_VALUE_HISTORY_PAGE_HPP
#define KEY_VALUE_HISTORY_PAGE_HPP

#include <unistd.h>
#include "zkresult.hpp"
#include "zkassert.hpp"
#include "scalar.hpp"

struct KeyValueHistoryStruct
{
    uint64_t hashPage1AndHistoryCounter; // historyCounter (2B) + hashPage1 (6B)
    uint64_t hashPage2AndPadding; // padding (2B) + hashPage2 (6B)
    uint8_t keyValueEntry[64][16]; // pageNumber (6B) + control (2B) + version (6B) + previousVersionOffset (2B) of the 64 key-values corresponding to this level
                                   // If control == 0, then this position is empty, i.e. value = 0
                                   // If control == 1, then this is a version (leaf node)
                                   // If control == 2, then this is the page number of the next level page (intermediate node)
    uint8_t historyEntry[191][16]; // history entries    
};

class KeyValueHistoryPage
{
private:
    static zkresult Read          (const uint64_t pageNumber,  const string &key, const string &keyBits,       mpz_class &value, const uint64_t level);
    static zkresult Write         (      uint64_t &pageNumber, const string &key, const string &keyBits, const mpz_class &value, const uint64_t level);
public:

    static zkresult InitEmptyPage (const uint64_t pageNumber);
    static zkresult Read          (const uint64_t pageNumber,  const string &key,       mpz_class &value);
    static zkresult Write         (      uint64_t &pageNumber, const string &key, const mpz_class &value);
    
    static void Print (const uint64_t pageNumber, bool details);
};

#endif