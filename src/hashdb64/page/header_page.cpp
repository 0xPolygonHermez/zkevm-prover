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
#include "version_data_page.hpp"


zkresult HeaderPage::Check (PageContext &ctx, const uint64_t headerPageNumber)
{
    // Get the header page
    HeaderStruct * page = (HeaderStruct *)ctx.pageManager.getPageAddress(headerPageNumber);

    // Check the uuid
    for(int i=0; i<32; ++i){
        if (page->uuid[i] != ctx.uuid[i])
        {
            zklog.error("HeaderPage::Check() found uuid[" + to_string(i) + "]=" + to_string(page->uuid[i]) + " != config.hashDBUUID[" + to_string(i) + "]=" + to_string(ctx.uuid[i]));
            exitProcess();
            return ZKR_DB_ERROR;
        }
    }
    return ZKR_SUCCESS;

}
zkresult HeaderPage::InitEmptyPage (PageContext &ctx, const uint64_t headerPageNumber)
{
    zkresult zkr;

    // Get the header page
    HeaderStruct * page = (HeaderStruct *)ctx.pageManager.getPageAddress(headerPageNumber);
    memset((void *)page, 0, 4096);

    // Create the raw data page, and init it
    page->firstRawDataPage = ctx.pageManager.getFreePage();
    zkr = RawDataPage::InitEmptyPage(ctx, page->firstRawDataPage);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("HeaderPage::InitEmptyPage() failed calling RawDataPage::InitEmptyPage() result=" + zkresult2string(zkr));
        return zkr;
    }
    page->rawDataPage = page->firstRawDataPage;

    // Create the root version page, and init it
    page->rootVersionPage = ctx.pageManager.getFreePage();
    zkr = KeyValuePage::InitEmptyPage(ctx, page->rootVersionPage);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("HeaderPage::InitEmptyPage() failed calling KeyValuePage::InitEmptyPage(rootVersionPage) result=" + zkresult2string(zkr));
        return zkr;
    }

    // Create the version data page, and init it
    page->versionDataPage = ctx.pageManager.getFreePage();
    zkr = KeyValuePage::InitEmptyPage(ctx, page->versionDataPage);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("HeaderPage::InitEmptyPage() failed calling KeyValuePage::InitEmptyPage(versionDataPage) result=" + zkresult2string(zkr));
        return zkr;
    }

    // Create the key value page, and init it
    page->keyValueHistoryPage = ctx.pageManager.getFreePage();
    zkr = KeyValueHistoryPage::InitEmptyPage(ctx, page->keyValueHistoryPage);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("HeaderPage::InitEmptyPage() failed calling KeyValueHistoryPage::InitEmptyPage() result=" + zkresult2string(zkr));
        return zkr;
    }

    // Create the program page, and init it
    page->programPage = ctx.pageManager.getFreePage();
    zkr = KeyValuePage::InitEmptyPage(ctx, page->programPage);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("HeaderPage::InitEmptyPage() failed calling KeyValuePage::InitEmptyPage() result=" + zkresult2string(zkr));
        return zkr;
    }

    // Create the free pages page, and init it
    page->freePages = ctx.pageManager.getFreePage();
    zkr = PageListPage::InitEmptyPage(ctx, page->freePages);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("HeaderPage::InitEmptyPage() failed calling PageListPage::InitEmptyPage(freePages) result=" + zkresult2string(zkr));
        return zkr;
    }
    page->firstUnusedPage = ctx.pageManager.getFirstUnusedPage();
    // UUID
    for(int i=0; i<32; ++i){
        page->uuid[i] = ctx.uuid[i];
    }
    
    return ZKR_SUCCESS;
}

uint64_t HeaderPage::GetLastVersion (PageContext &ctx, const uint64_t headerPageNumber)
{
    // Get the header page
    HeaderStruct * page = (HeaderStruct *)ctx.pageManager.getPageAddress(headerPageNumber);

    return page->lastVersion;
}

void HeaderPage::SetLastVersion (PageContext &ctx, uint64_t &headerPageNumber, const uint64_t lastVersion)
{
    // Get an editable page
    headerPageNumber = ctx.pageManager.editPage(headerPageNumber);

    // Get the header page
    HeaderStruct * page = (HeaderStruct *)ctx.pageManager.getPageAddress(headerPageNumber);

    // Check that version is incrementing
    if (lastVersion != page->lastVersion + 1)
    {
        zklog.error("HeaderPage::SetLastVersion() found new lastVersion=" + to_string(lastVersion) + " != page->lastVersion=" + to_string(page->lastVersion) + " + 1");
        exitProcess();
    }

    // Save the last version
    page->lastVersion = lastVersion;
}

zkresult HeaderPage::GetFreePagesContainer (PageContext &ctx, const uint64_t headerPageNumber, vector<uint64_t> (&containerPages))
{
    // Get header page
    HeaderStruct * page = (HeaderStruct *)ctx.pageManager.getPageAddress(headerPageNumber);

    // Call the specific method
    return PageListPage::GetContainerPages(ctx, page->freePages, containerPages);
}

zkresult HeaderPage::GetFreePages (PageContext &ctx, const uint64_t headerPageNumber, vector<uint64_t> (&freePages))
{
    // Get header page
    HeaderStruct * page = (HeaderStruct *)ctx.pageManager.getPageAddress(headerPageNumber);

    // Call the specific method
    freePages.clear();
    return PageListPage::GetPages(ctx, page->freePages, freePages);
}

zkresult HeaderPage::CreateFreePages (PageContext &ctx, uint64_t &headerPageNumber, vector<uint64_t> (&freePages), vector<uint64_t> (&containerPages))
{

    // Get an editable page
    headerPageNumber = ctx.pageManager.editPage(headerPageNumber);

    // Get header page
    HeaderStruct * page = (HeaderStruct *)ctx.pageManager.getPageAddress(headerPageNumber);

    // Call the specific method    
    return PageListPage::CreatePages(ctx, page->freePages, freePages, containerPages);

}

zkresult HeaderPage::GetFirstUnusedPage (PageContext &ctx, const uint64_t headerPageNumber, uint64_t &firstUnusedPage)
{
    // Get header page
    HeaderStruct * page = (HeaderStruct *)ctx.pageManager.getPageAddress(headerPageNumber);

    // Call the specific method
    firstUnusedPage = page->firstUnusedPage;

    return ZKR_SUCCESS;
}

zkresult HeaderPage::SetFirstUnusedPage (PageContext &ctx, uint64_t &headerPageNumber, const uint64_t firstUnusedPage)
{
    // Get an editable page
    headerPageNumber = ctx.pageManager.editPage(headerPageNumber);

    // Get header page
    HeaderStruct * page = (HeaderStruct *)ctx.pageManager.getPageAddress(headerPageNumber);

    // Call the specific method
    page->firstUnusedPage = firstUnusedPage;

    return ZKR_SUCCESS;
}
zkresult HeaderPage::GetLatestStateRoot (PageContext &ctx, const uint64_t  headerPageNumber, Goldilocks::Element (&root)[4]){
    
    zkresult zkr;

    // Get header page
    HeaderStruct * page = (HeaderStruct *)ctx.pageManager.getPageAddress(headerPageNumber);

    // Call the specific method
    VersionDataEntry versionData;
    zkr = VersionDataPage::Read(ctx, page->versionDataPage, page->lastVersion, versionData);

    if(zkr==ZKR_DB_KEY_NOT_FOUND){
        if(page->lastVersion==0){
            // If the last version is 0, then the root is 0
            root[0].fe = 0;
            root[1].fe = 0;
            root[2].fe = 0;
            root[3].fe = 0;
            zkr = ZKR_SUCCESS;
        }
    }else{
        // Convert the root from uint8_t[32] to Goldilocks::Element[4]
        root[3].fe = (((uint64_t)versionData.root[0])<<56) + (((uint64_t)versionData.root[1])<<48) + (((uint64_t)versionData.root[2])<<40) + (((uint64_t)versionData.root[3])<<32) + (((uint64_t)versionData.root[4])<<24) + (((uint64_t)versionData.root[5])<<16) + (((uint64_t)versionData.root[6])<<8) + ((uint64_t)versionData.root[7]);
        root[2].fe = (((uint64_t)versionData.root[8])<<56) + (((uint64_t)versionData.root[9])<<48) + (((uint64_t)versionData.root[10])<<40) + (((uint64_t)versionData.root[11])<<32) + (((uint64_t)versionData.root[12])<<24) + (((uint64_t)versionData.root[13])<<16) + (((uint64_t)versionData.root[14])<<8) + ((uint64_t)versionData.root[15]);
        root[1].fe = (((uint64_t)versionData.root[16])<<56) + (((uint64_t)versionData.root[17])<<48) + (((uint64_t)versionData.root[18])<<40) + (((uint64_t)versionData.root[19])<<32) + (((uint64_t)versionData.root[20])<<24) + (((uint64_t)versionData.root[21])<<16) + (((uint64_t)versionData.root[22])<<8) + ((uint64_t)versionData.root[23]);
        root[0].fe = (((uint64_t)versionData.root[24])<<56) + (((uint64_t)versionData.root[25])<<48) + (((uint64_t)versionData.root[26])<<40) + (((uint64_t)versionData.root[27])<<32) + (((uint64_t)versionData.root[28])<<24) + (((uint64_t)versionData.root[29])<<16) + (((uint64_t)versionData.root[30])<<8) + ((uint64_t)versionData.root[31]);
    }

    return zkr;
}

zkresult HeaderPage::ReadRootVersion (PageContext &ctx, const uint64_t headerPageNumber, const string &root, uint64_t &version)
{
    // Get header page
    HeaderStruct * page = (HeaderStruct *)ctx.pageManager.getPageAddress(headerPageNumber);

    // Call the specific method
    string value;
    zkresult zkr = KeyValuePage::Read(ctx, page->rootVersionPage, root, value);
    if (zkr == ZKR_SUCCESS)
    {
        version = value2version(ctx, value);
    }
    else if (zkr == ZKR_DB_KEY_NOT_FOUND)
    {
        version = 0;
    }

    return zkr;
}

zkresult HeaderPage::WriteRootVersion (PageContext &ctx, uint64_t &headerPageNumber, const string &root, const uint64_t &version)
{
    // Get an editable page
    headerPageNumber = ctx.pageManager.editPage(headerPageNumber);
    
    // Get header page
    HeaderStruct * headerPage = (HeaderStruct *)ctx.pageManager.getPageAddress(headerPageNumber);

    // Call the specific method
    return KeyValuePage::Write(ctx, headerPage->rootVersionPage, root, version2value(ctx, version), headerPageNumber);
}

zkresult HeaderPage::ReadVersionData (PageContext &ctx, const uint64_t headerPageNumber, const uint64_t &version, VersionDataEntry &versionData)
{
    // Get header page
    HeaderStruct * headerPage = (HeaderStruct *)ctx.pageManager.getPageAddress(headerPageNumber);

    // Call the specific method
    return VersionDataPage::Read(ctx, headerPage->versionDataPage, version, versionData);
}

zkresult HeaderPage::WriteVersionData (PageContext &ctx, uint64_t &headerPageNumber, const uint64_t &version, const VersionDataEntry &versionData)
{
    // Get an editable page
    headerPageNumber = ctx.pageManager.editPage(headerPageNumber);
    
    // Get header page
    HeaderStruct * headerPage = (HeaderStruct *)ctx.pageManager.getPageAddress(headerPageNumber);

    // Call the specific method
    return VersionDataPage::Write(ctx, headerPage->versionDataPage, version, versionData, headerPageNumber);
}

zkresult HeaderPage::KeyValueHistoryRead (PageContext &ctx, const uint64_t keyValueHistoryPage, const string &key, const uint64_t version, mpz_class &value, uint64_t &keyLevel)
{
    // Call the specific method
    return KeyValueHistoryPage::Read(ctx, keyValueHistoryPage, key, version, value, keyLevel);
}

zkresult HeaderPage::KeyValueHistoryReadLevel (PageContext &ctx, const uint64_t &headerPageNumber, const string &key, uint64_t &keyLevel)
{
    // Get header page
    HeaderStruct * headerPage = (HeaderStruct *)ctx.pageManager.getPageAddress(headerPageNumber);

    // Call the specific method
    return KeyValueHistoryPage::ReadLevel(ctx, headerPage->keyValueHistoryPage, key, keyLevel);
}

zkresult HeaderPage::KeyValueHistoryReadTree (PageContext &ctx, const uint64_t keyValueHistoryPage, const uint64_t version, vector<KeyValue> &keyValues, vector<HashValueGL> *hashValues)
{
    // Call the specific method
    return KeyValueHistoryPage::ReadTree(ctx, keyValueHistoryPage, version, keyValues, hashValues);
}

zkresult HeaderPage::KeyValueHistoryWrite (PageContext &ctx, uint64_t &headerPageNumber, const string &key, const uint64_t version, const mpz_class &value)
{
    // Get an editable page
    headerPageNumber = ctx.pageManager.editPage(headerPageNumber);
    
    // Get header page
    HeaderStruct * headerPage = (HeaderStruct *)ctx.pageManager.getPageAddress(headerPageNumber);

    // Call the specific method
    return KeyValueHistoryPage::Write(ctx, headerPage->keyValueHistoryPage, key, version, value, headerPageNumber);
}

zkresult HeaderPage::KeyValueHistoryCalculateHash (PageContext &ctx, uint64_t &headerPageNumber, Goldilocks::Element (&hash)[4])
{
    // Get an editable page
    headerPageNumber = ctx.pageManager.editPage(headerPageNumber);
    
    // Get header page
    HeaderStruct * headerPage = (HeaderStruct *)ctx.pageManager.getPageAddress(headerPageNumber);

    // Call the specific method
    return KeyValueHistoryPage::calculateHash(ctx, headerPage->keyValueHistoryPage, hash, headerPageNumber);
}

zkresult HeaderPage::KeyValueHistoryPrint (PageContext &ctx, const uint64_t headerPageNumber, const string &root)
{
    // Get header page
    HeaderStruct * headerPage = (HeaderStruct *)ctx.pageManager.getPageAddress(headerPageNumber);

    // If root is empty, print the last version
    if (root == "")
    {
        KeyValueHistoryPage::Print(ctx, headerPage->keyValueHistoryPage, false, "");
        return ZKR_SUCCESS;
    }

    // Convert root to a byte array
    string rootString = NormalizeToNFormat(root, 64);
    string rootBa =  string2ba(rootString);

    // Get the version associated to this root
    uint64_t version;
    zkresult zkr = HeaderPage::ReadRootVersion(ctx, headerPageNumber, rootBa, version);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("HeaderPage::KeyValueHistoryPrint() failed calling HeaderPage::ReadRootVersion() result=" + zkresult2string(zkr) + " root=" + rootString);
        return zkr;
    }

    // Get the version data
    VersionDataEntry versionData;
    zkr = HeaderPage::ReadVersionData(ctx, headerPageNumber, version, versionData);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("HeaderPage::KeyValueHistoryPrint() faile calling HeaderPage::ReadVersionData() result=" + zkresult2string(zkr) + " root=" + rootString);
        return zkr;
    }

    KeyValueHistoryPage::Print(ctx, versionData.keyValueHistoryPage, false, "");

    return ZKR_SUCCESS;
}

zkresult HeaderPage::ReadProgram (PageContext &ctx, const uint64_t headerPageNumber, const string &key, string &value)
{
    // Get header page
    HeaderStruct * headerPage = (HeaderStruct *)ctx.pageManager.getPageAddress(headerPageNumber);

    // Call the specific method
    return KeyValuePage::Read(ctx, headerPage->programPage, key, value);
}

zkresult HeaderPage::WriteProgram (PageContext &ctx, uint64_t &headerPageNumber, const string &key, const string &value)
{
    // Get an editable page
    headerPageNumber = ctx.pageManager.editPage(headerPageNumber);
    
    // Get header page
    HeaderStruct * headerPage = (HeaderStruct *)ctx.pageManager.getPageAddress(headerPageNumber);

    // Call the specific method
    return KeyValuePage::Write(ctx, headerPage->programPage, key, value, headerPageNumber);
}

void HeaderPage::Print (PageContext &ctx, const uint64_t headerPageNumber, bool details)
{
    // Get page
    HeaderStruct * page = (HeaderStruct *)ctx.pageManager.getPageAddress(headerPageNumber);
    zklog.info("");
    zklog.info("HeaderPage::Print() headerPageNumber=" + to_string(headerPageNumber));

    // Print raw data
    zklog.info("");
    zklog.info("firstRawDataPage=" + to_string(page->firstRawDataPage));
    zklog.info("rawDataPage=" + to_string(page->rawDataPage));
    RawDataPage::Print(ctx, page->rawDataPage, details, " ");

    // Print last version
    zklog.info("");
    zklog.info("lastVersion=" + to_string(page->lastVersion));

    // Print root-version page list
    zklog.info("");
    zklog.info("rootVersionPage=" + to_string(page->rootVersionPage));
    KeyValuePage::Print(ctx, page->rootVersionPage, details, " ", 32);

    // Print version-versionData page list
    zklog.info("");
    zklog.info("versionDataPage=" + to_string(page->versionDataPage));
    KeyValuePage::Print(ctx, page->versionDataPage, details, " ", 8);

    // Print key-value page list
    zklog.info("");
    zklog.info("keyValueHistoryPage=" + to_string(page->keyValueHistoryPage));
    KeyValueHistoryPage::Print(ctx, page->keyValueHistoryPage, details, " ");

    // Program page
    zklog.info("");
    zklog.info("programPage=" + to_string(page->programPage));
    KeyValuePage::Print(ctx, page->programPage, details, " ", 32);

    // Free pages
    zklog.info("");
    zklog.info("freePages=" + to_string(page->freePages));
    PageListPage::Print(ctx, page->freePages, details, " ");

    zklog.info("");
}