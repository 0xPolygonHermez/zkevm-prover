#ifndef VERSION_DATA_PAGE_HPP
#define VERSION_DATA_PAGE_NPP

#include <string>
#include <cstdint>

using namespace std;

/* Version data page is built on top of a generic KeyValuePage,
   in which the key is a U64 version converted to/from a string,
   and the value is a version data struct */

struct VersionDataStruct
{
    uint8_t  root[32];
    uint64_t keyValueHistoryPage;
    uint64_t freePagesList;
    uint64_t createdPagesList;
    uint64_t modifiedPagesList;
    uint64_t rawDataPage;
    uint64_t rawDataOffset;
};

string version2key (const uint64_t version);

string versionData2value (const VersionDataStruct &versionData);

void value2versionData (VersionDataStruct &versionData, const string &value);

#endif