#ifndef ROOT_VERSION_PAGE_HPP
#define ROOT_VERSION_PAGE_HPP

#include <unistd.h>
#include "zkresult.hpp"
#include "zkassert.hpp"

struct RootVersion
{
    char root[32];
    uint64_t version;
};

struct RootVersionStruct
{
    uint64_t nextPage; // page number of the next page in the list
    uint64_t size; // offset to current page next root version; size=4096 if the page is full
};

class RootVersionPage
{
public:
    
    static const uint64_t headerSize = 2*sizeof(uint64_t);
    static const uint64_t entrySize = sizeof(RootVersion);
    static const uint64_t maxEntries = (4096 - headerSize)/entrySize;
    static const uint64_t maxSize = headerSize + maxEntries*entrySize;

    static zkresult InitEmptyPage (const uint64_t pageNumber);
    static zkresult Read          (const uint64_t pageNumber,  const string &root, uint64_t &version);
    static zkresult Write         (      uint64_t &pageNumber, const string &root, uint64_t version);
};

#endif