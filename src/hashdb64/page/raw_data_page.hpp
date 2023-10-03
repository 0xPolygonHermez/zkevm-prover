#ifndef RAW_DATA_PAGE_HPP
#define RAW_DATA_PAGE_HPP

#include <unistd.h>
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
        current offset (n)

    You can append data to the raw data list, but you cannot modify data that has been previously appended
*/

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