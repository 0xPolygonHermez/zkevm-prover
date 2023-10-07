#ifndef HASH_PAGE_HPP
#define HASH_PAGE_HPP

#include <unistd.h>
#include "zkresult.hpp"
#include "zkassert.hpp"

struct HashStruct
{
    uint8_t hash[128][32]; // 128 hashes of 32B each
};

class HashPage
{
public:

    static zkresult InitEmptyPage (const uint64_t pageNumber);
    static zkresult Read          (const uint64_t pageNumber, const uint64_t position,       string &hash);
    static zkresult Write         (const uint64_t pageNumber, const uint64_t position, const string &hash);
    
    static void Print (const uint64_t pageNumber, bool details);
};

#endif