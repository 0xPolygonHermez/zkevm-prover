#ifndef KEY_VALUE_PAGE_HPP
#define KEY_VALUE_PAGE_HPP

#include <unistd.h>
#include "zkresult.hpp"
#include "zkassert.hpp"

struct KeyValueStruct
{
    uint64_t key[256][2]; // 256 entries, each of which is 16B long, being the first 2 bytes of the entry the control
    // if control == 0 --> empty slot
    // if control == 1 --> leaf node = control (2B) + length (6B) + rawPageOffset (2B) + rawPageNumber (6B)
    // if control == 2 --> intermediate node = control (2B) + nextKeyValuePage (6B) + reserved (8B)
    // Raw data contains: key (xB) + value (yB)
    // For the same KeyValuePage, the length of the key must always be the same, e.g.: 32B for a root-version, 8B for a version-versionData, 32B for a program page, etc.
};

class KeyValuePage
{
private:

    static zkresult Read          (const uint64_t  pageNumber, const string &key,       string &value, const uint64_t level);
    static zkresult Write         (      uint64_t &pageNumber, const string &key, const string &value, const uint64_t level, const uint64_t headerPageNumber);

public:

    static zkresult InitEmptyPage (const uint64_t  pageNumber);
    static zkresult Read          (const uint64_t  pageNumber, const string &key,       string &value);
    static zkresult Write         (      uint64_t &pageNumber, const string &key, const string &value, const uint64_t headerPageNumber);
    
    static void     Print         (const uint64_t pageNumber, bool details, const string &prefix);
};

#endif