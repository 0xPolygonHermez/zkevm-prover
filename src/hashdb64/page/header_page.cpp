#include "header_page.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"
#include "zkassert.hpp"
#include "page_manager.hpp"
#include "key_value_history_page.hpp"
#include "raw_data_page.hpp"
#include "key_value_page.hpp"
#include "page_list_page.hpp"
#include "root_version_page.hpp"

zkresult HeaderPage::InitEmptyPage (const uint64_t headerPageNumber)
{
    zkresult zkr;

    // Get the header page
    HeaderStruct * page = (HeaderStruct *)pageManager.getPageAddress(headerPageNumber);
    memset((void *)page, 0, 4096);

    // Create the raw data page, and init it
    page->firstRawDataPage = pageManager.getFreePage();
    zkr = RawDataPage::InitEmptyPage(page->firstRawDataPage);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("HeaderPage::InitEmptyPage() failed calling RawDataPage::InitEmptyPage() result=" + zkresult2string(zkr));
        return zkr;
    }
    page->rawDataPage = page->firstRawDataPage;

    // Create the root version page, and init it
    page->rootVersionPage = pageManager.getFreePage();
    zkr = KeyValuePage::InitEmptyPage(page->rootVersionPage);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("HeaderPage::InitEmptyPage() failed calling KeyValuePage::InitEmptyPage(rootVersionPage) result=" + zkresult2string(zkr));
        return zkr;
    }

    // Create the version data page, and init it
    page->versionDataPage = pageManager.getFreePage();
    zkr = KeyValuePage::InitEmptyPage(page->versionDataPage);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("HeaderPage::InitEmptyPage() failed calling KeyValuePage::InitEmptyPage(versionDataPage) result=" + zkresult2string(zkr));
        return zkr;
    }

    // Create the key value page, and init it
    page->keyValueHistoryPage = pageManager.getFreePage();
    zkr = KeyValueHistoryPage::InitEmptyPage(page->keyValueHistoryPage);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("HeaderPage::InitEmptyPage() failed calling KeyValueHistoryPage::InitEmptyPage() result=" + zkresult2string(zkr));
        return zkr;
    }

    // Create the program page, and init it
    page->programPage = pageManager.getFreePage();
    zkr = KeyValuePage::InitEmptyPage(page->programPage);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("HeaderPage::InitEmptyPage() failed calling KeyValuePage::InitEmptyPage() result=" + zkresult2string(zkr));
        return zkr;
    }

    // Create the free pages page, and init it
    page->freePages = pageManager.getFreePage();
    zkr = PageListPage::InitEmptyPage(page->freePages);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("HeaderPage::InitEmptyPage() failed calling PageListPage::InitEmptyPage(freePages) result=" + zkresult2string(zkr));
        return zkr;
    }
    
    return ZKR_SUCCESS;
}

uint64_t HeaderPage::GetLastVersion (const uint64_t headerPageNumber)
{
    // Get the header page
    HeaderStruct * page = (HeaderStruct *)pageManager.getPageAddress(headerPageNumber);

    return page->lastVersion;
}

void HeaderPage::SetLastVersion (uint64_t &headerPageNumber, const uint64_t lastVersion)
{
    // Get an editable page
    headerPageNumber = pageManager.editPage(headerPageNumber);

    // Get the header page
    HeaderStruct * page = (HeaderStruct *)pageManager.getPageAddress(headerPageNumber);

    // Check that version is incrementing
    if (lastVersion != page->lastVersion + 1)
    {
        zklog.error("HeaderPage::SetLastVersion() found new lastVersion=" + to_string(lastVersion) + " != page->lastVersion=" + to_string(page->lastVersion) + " + 1");
        exitProcess();
    }

    // Save the last version
    page->lastVersion = lastVersion;
}

zkresult HeaderPage::GetFreePages (const uint64_t headerPageNumber, vector<uint64_t> (&containerPages), vector<uint64_t> (&containedPages))
{
    // Get header page
    HeaderStruct * page = (HeaderStruct *)pageManager.getPageAddress(headerPageNumber);

    // Call the specific method
    return PageListPage::GetPages(page->freePages, containerPages, containedPages);
}

zkresult HeaderPage::CreateFreePages (uint64_t &headerPageNumber, vector<uint64_t> (&freePages), vector<uint64_t> (&containerPages), vector<uint64_t> (&containedPages))
{
    // Get header page
    HeaderStruct * page = (HeaderStruct *)pageManager.getPageAddress(headerPageNumber);

    // Call the specific method
    return PageListPage::CreatePages(page->freePages, freePages, containerPages, containedPages);
}

zkresult HeaderPage::ReadRootVersion (const uint64_t headerPageNumber, const string &root, uint64_t &version)
{
    // Get header page
    HeaderStruct * page = (HeaderStruct *)pageManager.getPageAddress(headerPageNumber);

    // Call the specific method
    string value;
    zkresult zkr = KeyValuePage::Read(page->rootVersionPage, root, value);
    if (zkr == ZKR_SUCCESS)
    {
        version = value2version(value);
    }
    else if (zkr == ZKR_DB_KEY_NOT_FOUND)
    {
        version = 0;
    }

    return zkr;
}

zkresult HeaderPage::WriteRootVersion (uint64_t &headerPageNumber, const string &root, const uint64_t &version)
{
    // Get an editable page
    headerPageNumber = pageManager.editPage(headerPageNumber);
    
    // Get header page
    HeaderStruct * headerPage = (HeaderStruct *)pageManager.getPageAddress(headerPageNumber);

    // Call the specific method
    return KeyValuePage::Write(headerPage->rootVersionPage, root, version2value(version), headerPageNumber);
}

zkresult HeaderPage::ReadVersionData (const uint64_t headerPageNumber, const uint64_t &version, VersionDataStruct &versionData)
{
    // Get header page
    HeaderStruct * headerPage = (HeaderStruct *)pageManager.getPageAddress(headerPageNumber);

    // Call the specific method
    string value;
    zkresult zkr = KeyValuePage::Read(headerPage->versionDataPage, version2key(version), value);
    if (zkr == ZKR_SUCCESS)
    {
        value2versionData(versionData, value);
    }

    return zkr;
}

zkresult HeaderPage::WriteVersionData (uint64_t &headerPageNumber, const uint64_t &version, const VersionDataStruct &versionData)
{
    // Get an editable page
    headerPageNumber = pageManager.editPage(headerPageNumber);
    
    // Get header page
    HeaderStruct * headerPage = (HeaderStruct *)pageManager.getPageAddress(headerPageNumber);

    // Call the specific method
    return KeyValuePage::Write(headerPage->versionDataPage, version2key(version), versionData2value(versionData), headerPageNumber);
}

zkresult HeaderPage::KeyValueHistoryRead (const uint64_t keyValueHistoryPage, const string &key, const uint64_t version, mpz_class &value, uint64_t &keyLevel)
{
    // Call the specific method
    return KeyValueHistoryPage::Read(keyValueHistoryPage, key, version, value, keyLevel);
}

zkresult HeaderPage::KeyValueHistoryReadLevel (uint64_t &headerPageNumber, const string &key, uint64_t &keyLevel)
{
    // Get header page
    HeaderStruct * headerPage = (HeaderStruct *)pageManager.getPageAddress(headerPageNumber);

    // Call the specific method
    return KeyValueHistoryPage::ReadLevel(headerPage->keyValueHistoryPage, key, keyLevel);
}

zkresult HeaderPage::KeyValueHistoryWrite (uint64_t &headerPageNumber, const string &key, const uint64_t version, const mpz_class &value)
{
    // Get an editable page
    headerPageNumber = pageManager.editPage(headerPageNumber);
    
    // Get header page
    HeaderStruct * headerPage = (HeaderStruct *)pageManager.getPageAddress(headerPageNumber);

    // Call the specific method
    return KeyValueHistoryPage::Write(headerPage->keyValueHistoryPage, key, version, value, headerPageNumber);
}

zkresult HeaderPage::KeyValueHistoryCalculateHash (uint64_t &headerPageNumber, Goldilocks::Element (&hash)[4])
{
    // Get an editable page
    headerPageNumber = pageManager.editPage(headerPageNumber);
    
    // Get header page
    HeaderStruct * headerPage = (HeaderStruct *)pageManager.getPageAddress(headerPageNumber);

    // Call the specific method
    return KeyValueHistoryPage::calculateHash(headerPage->keyValueHistoryPage, hash, headerPageNumber);
}

zkresult HeaderPage::ReadProgram (const uint64_t headerPageNumber, const string &key, string &value)
{
    // Get header page
    HeaderStruct * headerPage = (HeaderStruct *)pageManager.getPageAddress(headerPageNumber);

    // Call the specific method
    return KeyValuePage::Read(headerPage->programPage, key, value);
}

zkresult HeaderPage::WriteProgram (uint64_t &headerPageNumber, const string &key, const string &value)
{
    // Get an editable page
    headerPageNumber = pageManager.editPage(headerPageNumber);
    
    // Get header page
    HeaderStruct * headerPage = (HeaderStruct *)pageManager.getPageAddress(headerPageNumber);

    // Call the specific method
    return KeyValuePage::Write(headerPage->programPage, key, value, headerPageNumber);
}

void HeaderPage::Print (const uint64_t headerPageNumber, bool details)
{
    // Get page
    HeaderStruct * page = (HeaderStruct *)pageManager.getPageAddress(headerPageNumber);
    zklog.info("");
    zklog.info("HeaderPage::Print() headerPageNumber=" + to_string(headerPageNumber));

    // Print raw data
    zklog.info("");
    zklog.info("firstRawDataPage=" + to_string(page->firstRawDataPage));
    zklog.info("rawDataPage=" + to_string(page->rawDataPage));
    RawDataPage::Print(page->rawDataPage, details, " ");

    // Print last version
    zklog.info("");
    zklog.info("lastVersion=" + to_string(page->lastVersion));

    // Print root-version page list
    zklog.info("");
    zklog.info("rootVersionPage=" + to_string(page->rootVersionPage));
    KeyValuePage::Print(page->rootVersionPage, details, " ", 32);

    // Print version-versionData page list
    zklog.info("");
    zklog.info("versionDataPage=" + to_string(page->versionDataPage));
    KeyValuePage::Print(page->versionDataPage, details, " ", 8);

    // Print key-value page list
    zklog.info("");
    zklog.info("keyValueHistoryPage=" + to_string(page->keyValueHistoryPage));
    KeyValueHistoryPage::Print(page->keyValueHistoryPage, details, " ");

    // Program page
    zklog.info("");
    zklog.info("programPage=" + to_string(page->programPage));
    KeyValuePage::Print(page->programPage, details, " ", 32);

    // Free pages
    zklog.info("");
    zklog.info("freePages=" + to_string(page->freePages));
    PageListPage::Print(page->freePages, details, " ");

    zklog.info("");
}