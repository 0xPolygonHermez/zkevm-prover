#ifndef RAW_DATA_PAGE_HPP
#define RAW_DATA_PAGE_HPP

#include <unistd.h>
#include "zkresult.hpp"
#include "zkassert.hpp"

struct RawDataStruct
{
    uint64_t previousPageNumber;
    uint64_t nextPageNumber;
};

class RawDataPage
{
public:

    static const uint64_t minPosition = 16; // This means that the page is completely empty
    static const uint64_t maxPosition = 4096; // This means that the page is completely full

    static zkresult InitEmptyPage (const uint64_t  pageNumber);
    static zkresult Read          (const uint64_t  pageNumber, const uint64_t  position, const uint64_t length,      string &data);
    static zkresult Write         (      uint64_t &pageNumber,       uint64_t &position,                       const string &data);
    
    static void Print (const uint64_t pageNumber, bool details);
};

#endif