#ifndef HEADER_PAGE_HPP
#define HEADER_PAGE_HPP

#include <unistd.h>
#include "zkresult.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"
#include "zkassert.hpp"
#include "version_data_entry.hpp"
#include "scalar.hpp"
#include "key_value.hpp"
#include "hash_value_gl.hpp"
#include "page_context.hpp"

struct HeaderStruct
{
    // UUID
    uint8_t uuid[32];

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
    uint64_t firstUnusedPage;
};

class HeaderPage
{
public:
    
    // Header uuid
    static zkresult Check (PageContext &ctx, const uint64_t headerPageNumber);

    // Header-only methods
    static zkresult InitEmptyPage  (PageContext &ctx, const uint64_t  headerPageNumber);
    static uint64_t GetLastVersion (PageContext &ctx, const uint64_t  headerPageNumber);
    static void     SetLastVersion (PageContext &ctx,       uint64_t &headerPageNumber, const uint64_t lastVersion);

    // Free pages list methods
    static zkresult GetFreePagesContainer (PageContext &ctx, const uint64_t  headerPageNumber, vector<uint64_t> (&containerPages));
    static zkresult GetFreePages          (PageContext &ctx, const uint64_t  headerPageNumber, vector<uint64_t> (&freePages));
    static zkresult CreateFreePages       (PageContext &ctx,       uint64_t &headerPageNumber, vector<uint64_t> (&freePages), vector<uint64_t> (&containerPages));
    static zkresult GetFirstUnusedPage    (PageContext &ctx,  const uint64_t  headerPageNumber, uint64_t &firstUnusedPage);
    static zkresult SetFirstUnusedPage    (PageContext &ctx,       uint64_t &headerPageNumber, const uint64_t firstUnusedPage);

    // Root version methods
    static zkresult GetLatestStateRoot (PageContext &ctx, const uint64_t  headerPageNumber, Goldilocks::Element (&root)[4]);
    static zkresult ReadRootVersion    (PageContext &ctx, const uint64_t  headerPageNumber, const string &root,       uint64_t &version);
    static zkresult WriteRootVersion   (PageContext &ctx,       uint64_t &headerPageNumber, const string &root, const uint64_t &version);

    // Version data methods
    static zkresult ReadVersionData  (PageContext &ctx, const uint64_t  headerPageNumber, const uint64_t &version,       VersionDataEntry &versionData);
    static zkresult WriteVersionData (PageContext &ctx,       uint64_t &headerPageNumber, const uint64_t &version, const VersionDataEntry &versionData);

    // Key-Value-History methods
    static zkresult KeyValueHistoryRead          (PageContext &ctx, const uint64_t  keyValueHistoryPage, const string &key, const uint64_t version,       mpz_class &value, uint64_t &keyLevel);
    static zkresult KeyValueHistoryReadLevel     (PageContext &ctx, const uint64_t &headerPageNumber,    const string &key, uint64_t &keyLevel);
    static zkresult KeyValueHistoryReadTree      (PageContext &ctx, const uint64_t  keyValueHistoryPage, const uint64_t version,    vector<KeyValue> &keyValues, vector<HashValueGL> *hashValues);
    static zkresult KeyValueHistoryWrite         (PageContext &ctx,       uint64_t &headerPageNumber,    const string &key, const uint64_t version, const mpz_class &value);
    static zkresult KeyValueHistoryCalculateHash (PageContext &ctx,       uint64_t &headerPageNumber,    Goldilocks::Element (&hash)[4]);
    static zkresult KeyValueHistoryPrint         (PageContext &ctx, const uint64_t  headerPageNumber,    const string &root);

    // Program page methods
    static zkresult ReadProgram  (PageContext &ctx, const uint64_t  headerPageNumber, const string &key,       string &value);
    static zkresult WriteProgram (PageContext &ctx,       uint64_t &headerPageNumber, const string &key, const string &value);


    static void Print (PageContext &ctx, const uint64_t headerPageNumber, bool details);
};

#endif