#ifndef VERSION_DATA_ENTRY_HPP
#define VERSION_DATA_ENTRY_HPP

#include <cstdint>

struct VersionDataEntry
{
    uint64_t key;
    uint8_t  root[32];
    uint64_t keyValueHistoryPage;
    uint64_t controlAndFreePagesList; // control (2B) + free pages list (6B), or control (2B) + nextVersionDataPage (6B)
    uint64_t createdPagesList;
    //uint64_t modifiedPagesList;
    //uint64_t rawDataPage;
    //uint64_t rawDataOffset;
};

#endif