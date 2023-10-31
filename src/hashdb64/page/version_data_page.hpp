#ifndef VERSION_DATA_PAGE_HPP
#define VERSION_DATA_PAGE_NPP

#include <string>
#include <cstdint>
#include "page_context.hpp"
#include "version_data_entry.hpp"

using namespace std;

/* Version data page is a tree of 6 bits per level, 64 entries per page, each of them 64B-long */

struct VersionDataStruct
{
    VersionDataEntry versionDataEntry[64];
};

class VersionDataCounters
{
public:
    uint64_t intermediateNodes;
    uint64_t leafNodes;
    uint64_t maxLevel;
    VersionDataCounters() : intermediateNodes(0), leafNodes(0), maxLevel(0) {};
};

class VersionDataPage
{
private:
    static zkresult Read          (PageContext &ctx, const uint64_t pageNumber,  const uint64_t key, const string &keyBits,       VersionDataEntry &value, const uint64_t level);
    static zkresult Write         (PageContext &ctx,       uint64_t &pageNumber, const uint64_t key, const string &keyBits, const VersionDataEntry &value, const uint64_t level, uint64_t &headerPageNumber);
public:
    static zkresult InitEmptyPage (PageContext &ctx, const uint64_t pageNumber);
    static zkresult Read          (PageContext &ctx, const uint64_t pageNumber,  const uint64_t key,       VersionDataEntry &value);
    static zkresult Write         (PageContext &ctx,       uint64_t &pageNumber, const uint64_t key, const VersionDataEntry &value, uint64_t &headerPageNumber);
    
    static void Print (PageContext &ctx, const uint64_t pageNumber, bool details, const string &prefix, const uint64_t level, VersionDataCounters &counters);
    static void Print (PageContext &ctx, const uint64_t pageNumber, bool details, const string &prefix);
};

#endif