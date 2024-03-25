#ifndef KEY_VALUE_PAGE_HPP
#define KEY_VALUE_PAGE_HPP

#include <unistd.h>
#include <vector>
#include <stdint.h>
#include "zkresult.hpp"
#include "zkassert.hpp"

using namespace std;

struct KeyValueStruct
{
    uint64_t key[512]; // 512 entries, each of which is 8B long, being the first 4 bits of the entry the control
    // if control == 0 --> empty slot
    // if control == 1 --> leaf node = control (4b) + rawPageOffset (12b) + rawPageNumber (6B)
    // if control == 2 --> intermediate node = control (4b) + reserved (12b) + nextKeyValuePage (6B)
    // Raw data contains: length (4B) + key (xB) + value (yB)
    // For the same KeyValuePage, the length of the key must always be the same, e.g.: 32B for a root-version, 8B for a version-versionData, 32B for a program page, etc.
};

class KeyValuePage
{
private:

    static zkresult Read          (const uint64_t  pageNumber, const string &key, const vector<uint64_t> &keyBits,       string &value, const uint64_t level);
    static zkresult Write         (      uint64_t &pageNumber, const string &key, const vector<uint64_t> &keyBits, const string &value, const uint64_t level, const uint64_t headerPageNumber);

public:

    static zkresult InitEmptyPage (const uint64_t  pageNumber);
    static zkresult Read          (const uint64_t  pageNumber, const string &key,       string &value);
    static zkresult Write         (      uint64_t &pageNumber, const string &key, const string &value, const uint64_t headerPageNumber);
    
    static void     Print         (const uint64_t pageNumber, bool details, const string &prefix, const uint64_t keySize);
};

#endif