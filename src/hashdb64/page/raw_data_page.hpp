#ifndef RAW_DATA_PAGE_HPP
#define RAW_DATA_PAGE_HPP

#include <unistd.h>
#include <stdint.h>
#include "zkresult.hpp"
#include "zkassert.hpp"

/*
    Example with 3 raw pages:

    First page:
        previousPageNumber = 0;
        nextPageNumber = second page

    Second page:
        previousPageNumber = first page
        nextPageNumber = third page

    Third page: (current page)
        previousPageNumber = second page
        nextPageNumber = 0;

    Header contains:
        first page (0)
        current page (3)

    You can append data to the raw data list, but you cannot modify data that has been previously appended
*/

struct RawDataStruct
{
    uint64_t previousPageNumber; // reserved (2B) + previousPageNumber (6B)
    uint64_t nextPageNumberAndOffset; // offset (2B) + nextPageNumber (6B)
};

class RawDataPage
{
public:

    static const uint64_t minOffset= 16; // This means that the page is completely empty
    static const uint64_t maxOffset = 4096; // This means that the page is completely full
    static const uint64_t allignment = 8; // The data will be written in an address alligned with these many bytes

    static zkresult InitEmptyPage (const uint64_t  pageNumber);
    static zkresult Read          (const uint64_t  pageNumber, const uint64_t offset, const uint64_t length,       string &data);
    static zkresult Write         (      uint64_t &pageNumber,                                               const string &data);
    
    static uint64_t GetOffset     (const uint64_t  pageNumber);
    
    static void Print (const uint64_t pageNumber, bool details, const string &prefix);
};

#endif