#ifndef HEADER_PAGE_HPP
#define HEADER_PAGE_HPP

#include <unistd.h>
#include "zkresult.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"
#include "zkassert.hpp"

struct HeaderStruct
{
    // Raw data
    uint64_t firstRawDataPage;
    uint64_t rawDataPage;

    // Root -> version
    uint64_t lastVersion;
    uint64_t rootVersionPage;

    // Version -> version data
    uint64_t versionDataPage;

    // Key -> value history
    uint64_t keyValueHistoryPage;

    // Program page list
    uint64_t programPage;

    // Free pages list
    uint64_t freePages;
};

class HeaderPage
{
public:
    // Header-only methods
    static zkresult InitEmptyPage  (const uint64_t  headerPageNumber);
    static uint64_t GetLastVersion (const uint64_t  headerPageNumber);
    static void     SetLastVersion (      uint64_t &headerPageNumber, const uint64_t lastVersion);

    // Free pages list methods
    static zkresult GetFreePages    (const uint64_t  headerPageNumber,                                      vector<uint64_t> (&containerPages), vector<uint64_t> (&containedPages));
    static zkresult CreateFreePages (      uint64_t &headerPageNumber, const vector<uint64_t> (&freePages), vector<uint64_t> (&containerPages), vector<uint64_t> (&containedPages));

    static void Print (const uint64_t headerPageNumber, bool details);
};

#endif