#ifndef PAGE_LIST_PAGE_HPP
#define PAGE_LIST_PAGE_HPP

#include <unistd.h>
#include <stdint.h>
#include <vector>
#include "zkresult.hpp"
#include "zkassert.hpp"

using namespace std;

/*
    Example with 3 page list pages:

    First page:
        previousPageNumber = 0;
        nextPageNumber = second page

    Second page:
        previousPageNumber = first page
        nextPageNumber = third page

    Third page: (current page)
        previousPageNumber = second page
        nextPageNumber = 0;

    You can insert a page to the page list, creating a new pages if the current one is full
    You can remove a page from the page list, feeing the current pages when empty
*/

struct PageListStruct
{
    uint64_t previousPageNumber; // reserved (2B) + previousPageNumber (6B)
    uint64_t nextPageNumberAndOffset; // offset (2B) + nextPageNumber (6B)
};

class PageListPage
{
public:

    static const uint64_t minOffset= 16; // This means that the page is completely empty
    static const uint64_t maxOffset = 4096; // This means that the page is completely full

    static zkresult InitEmptyPage (const uint64_t  pageNumber);
    static zkresult InsertPage    (      uint64_t &pageNumber, const uint64_t pageNumberToInsert);
    static zkresult ExtractPage   (      uint64_t &pageNumber,       uint64_t &extractedPageNumber);

    static zkresult GetPages      (const uint64_t  pageNumber,                                vector<uint64_t> (&containerPages), vector<uint64_t> (&containedPages));
    static zkresult CreatePages   (      uint64_t &pageNumber, vector<uint64_t> (&freePages), vector<uint64_t> (&containerPages), vector<uint64_t> (&containedPages));

    static void Print (const uint64_t pageNumber, bool details, const string &prefix);
};

#endif