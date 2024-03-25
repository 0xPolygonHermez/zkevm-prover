#ifndef HEADER_PAGE_HPP
#define HEADER_PAGE_HPP

#include <unistd.h>
#include "zkresult.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"
#include "zkassert.hpp"
#include "version_data_page.hpp"
#include "scalar.hpp"

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
    static zkresult GetFreePages    (const uint64_t  headerPageNumber,                                vector<uint64_t> (&containerPages), vector<uint64_t> (&containedPages));
    static zkresult CreateFreePages (      uint64_t &headerPageNumber, vector<uint64_t> (&freePages), vector<uint64_t> (&containerPages), vector<uint64_t> (&containedPages));

    // Root version methods
    static zkresult ReadRootVersion  (const uint64_t  headerPageNumber, const string &root,       uint64_t &version);
    static zkresult WriteRootVersion (      uint64_t &headerPageNumber, const string &root, const uint64_t &version);

    // Version data methods
    static zkresult ReadVersionData  (const uint64_t  headerPageNumber, const uint64_t &version,       VersionDataStruct &versionData);
    static zkresult WriteVersionData (      uint64_t &headerPageNumber, const uint64_t &version, const VersionDataStruct &versionData);

    // Key-Value-History methods
    static zkresult KeyValueHistoryRead          (const uint64_t  keyValueHistoryPage, const string &key, const uint64_t version,       mpz_class &value, uint64_t &keyLevel);
    static zkresult KeyValueHistoryReadLevel     (      uint64_t &headerPageNumber,    const string &key, uint64_t &keyLevel);
    static zkresult KeyValueHistoryWrite         (      uint64_t &headerPageNumber,    const string &key, const uint64_t version, const mpz_class &value);
    static zkresult KeyValueHistoryCalculateHash (uint64_t &headerPageNumber, Goldilocks::Element (&hash)[4]);

    // Program page methods
    static zkresult ReadProgram  (const uint64_t  headerPageNumber, const string &key,       string &value);
    static zkresult WriteProgram (      uint64_t &headerPageNumber, const string &key, const string &value);


    static void Print (const uint64_t headerPageNumber, bool details);
};

#endif