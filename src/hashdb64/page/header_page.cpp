#include "header_page.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"
#include "zkassert.hpp"
#include "page_manager.hpp"
#include "key_value_history_page.hpp"
#include "raw_data_page.hpp"
#include "key_value_page.hpp"

zkresult HeaderPage::InitEmptyPage (const uint64_t pageNumber)
{
    zkresult zkr;

    // Get the header page
    HeaderStruct * page = (HeaderStruct *)pageManager.getPageAddress(pageNumber);
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
    
    return ZKR_SUCCESS;
}

uint64_t HeaderPage::GetLastVersion (const uint64_t pageNumber)
{
    // Get the header page
    HeaderStruct * page = (HeaderStruct *)pageManager.getPageAddress(pageNumber);

    return page->lastVersion;
}

void HeaderPage::SetLastVersion (uint64_t &pageNumber, const uint64_t lastVersion)
{
    // Get an editable page
    pageNumber = pageManager.editPage(pageNumber);

    // Get the header page
    HeaderStruct * page = (HeaderStruct *)pageManager.getPageAddress(pageNumber);

    // Check that version is incrementing
    if (lastVersion != page->lastVersion + 1)
    {
        zklog.error("HeaderPage::SetLastVersion() found new lastVersion=" + to_string(lastVersion) + " != page->lastVersion=" + to_string(page->lastVersion) + " + 1");
        exitProcess();
    }

    // Save the last version
    page->lastVersion = lastVersion;
}

void HeaderPage::Print (const uint64_t pageNumber, bool details)
{
    // Get page
    HeaderStruct * page = (HeaderStruct *)pageManager.getPageAddress(pageNumber);
    zklog.info("HeaderPage::Print() pageNumber=" + to_string(pageNumber));

    // Print raw data
    zklog.info("  firstRawDataPage=" + to_string(page->firstRawDataPage) + "=" + to_string((uint64_t)pageManager.getPageAddress(page->firstRawDataPage)));
    zklog.info("  rawDataPage=" + to_string(page->rawDataPage) + "=" + to_string((uint64_t)pageManager.getPageAddress(page->rawDataPage)));
    RawDataPage::Print(page->rawDataPage, details);

    // Print last version
    zklog.info("  lastVersion=" + to_string(page->lastVersion));

    // Print root-version page list
    zklog.info("  rootVersionPage=" + to_string(page->rootVersionPage) + "=" + to_string((uint64_t)pageManager.getPageAddress(page->rootVersionPage)));
    KeyValuePage::Print(page->rootVersionPage, details);

    // Print version-versionData page list
    zklog.info("  versionDataPage=" + to_string(page->versionDataPage) + "=" + to_string((uint64_t)pageManager.getPageAddress(page->versionDataPage)));
    KeyValuePage::Print(page->versionDataPage, details);

    // Print key-value page list
    zklog.info("  keyValueHistoryPage=" + to_string(page->keyValueHistoryPage) + "=" + to_string((uint64_t)pageManager.getPageAddress(page->keyValueHistoryPage)));
    KeyValueHistoryPage::Print(page->keyValueHistoryPage, details);

    // Program page
    zklog.info("  programPage=" + to_string(page->programPage) + "=" + to_string((uint64_t)pageManager.getPageAddress(page->programPage)));
    KeyValuePage::Print(page->programPage, details);
}