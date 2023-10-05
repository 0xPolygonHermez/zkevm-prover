#ifndef KEY_VALUE_PAGE_HPP
#define KEY_VALUE_PAGE_HPP

#include <unistd.h>
#include "zkresult.hpp"
#include "zkassert.hpp"

struct KeyValueStruct
{
    uint64_t key[256][2]; // 256 hashes of 16B each, being the first 2B the control
    // if control == 0 --> empty slot
    // if control == 1 --> leaf node = control (2B) + length (6B) + rawPageOffset (2B) + rawPageNumber (6B)
    // if control == 2 --> intermediate node = control (2B) + nextKeyValuePage (6B) + reserved (8B)
    // Raw data contains: key (32B) + program (xB)
};

class KeyValuePage
{
private:

    static zkresult Read          (const uint64_t pageNumber, const string &key,       string &value, const uint64_t level);
    static zkresult Write         (const uint64_t pageNumber, const string &key, const string &value, const uint64_t level, const uint64_t headerPageNumber);

public:

    static zkresult InitEmptyPage (const uint64_t pageNumber);
    static zkresult Read          (const uint64_t pageNumber, const string &key,       string &value);
    static zkresult Write         (const uint64_t pageNumber, const string &key, const string &value, const uint64_t headerPageNumber);
    
    static void     Print         (const uint64_t pageNumber, bool details);
};

#endif