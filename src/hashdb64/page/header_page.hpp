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
};

class HeaderPage
{
public:
    // Header-only methods
    static zkresult InitEmptyPage  (const uint64_t pageNumber);
    static uint64_t GetLastVersion (const uint64_t pageNumber);
    static void     SetLastVersion (uint64_t &pageNumber, const uint64_t lastVersion);



    static void Print (const uint64_t pageNumber, bool details);
};

#endif