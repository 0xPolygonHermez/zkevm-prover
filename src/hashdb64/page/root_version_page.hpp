#ifndef ROOT_VERSION_PAGE_HPP
#define ROOT_VERSION_PAGE_HPP

#include <unistd.h>
#include "zkresult.hpp"
#include "zkassert.hpp"

struct RootVersionStruct
{
    uint64_t nextPage; // page number of the next page in the list
    uint64_t versionAndControl[128]; // version (6B) + control (2B) of the 128 state roots corresponding to this level
                                     // If control == 0, then this position is empty
                                     // If control == 1, then this is a version (leaf node)
                                     // If control == 2, then this is the page number of the next level page (intermediate node)
};

class RootVersionPage
{
private:
    static zkresult Read          (const uint64_t pageNumber,  const string &root,       uint64_t &version, const uint64_t level);
    static zkresult Write         (      uint64_t &pageNumber, const string &root, const uint64_t  version, const uint64_t level);
public:

    static zkresult InitEmptyPage (const uint64_t pageNumber);
    static zkresult Read          (const uint64_t pageNumber,  const string &root,       uint64_t &version);
    static zkresult Write         (      uint64_t &pageNumber, const string &root, const uint64_t  version);

    static void Print (const uint64_t pageNumber, bool details);
};

#endif