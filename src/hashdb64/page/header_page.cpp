#include "header_page.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"
#include "zkassert.hpp"
#include "page_manager.hpp"
#include "root_version_page.hpp"
#include "key_value_page.hpp"
#include "raw_data_page.hpp"
#include "program_page.hpp"

zkresult HeaderPage::InitEmptyPage (const uint64_t pageNumber)
{
    zkresult zkr;

    // Get the header page
    HeaderStruct * page = (HeaderStruct *)pageManager.getPage(pageNumber);
    memset((void *)page, 0, 4096);

    // Create the root version page, and init it
    page->rootVersionPage = pageManager.getFreePage();
    zkr = RootVersionPage::InitEmptyPage(page->rootVersionPage);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("HeaderPage::InitEmptyPage() failed calling RootVersionPage::InitEmptyPage() result=" + zkresult2string(zkr));
        return zkr;
    }

    // Create the key value page, and init it
    page->keyValuePage = pageManager.getFreePage();
    zkr = KeyValuePage::InitEmptyPage(page->keyValuePage);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("HeaderPage::InitEmptyPage() failed calling KeyValuePage::InitEmptyPage() result=" + zkresult2string(zkr));
        return zkr;
    }

    // Create the raw data page, and init it
    page->firstRawDataPage = pageManager.getFreePage();
    zkr = RawDataPage::InitEmptyPage(page->firstRawDataPage);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("HeaderPage::InitEmptyPage() failed calling RawDataPage::InitEmptyPage() result=" + zkresult2string(zkr));
        return zkr;
    }
    page->rawDataPage = page->firstRawDataPage;
    page->rawDataOffset = RawDataPage::minPosition;

    // Create the program page, and init it
    page->programPage = pageManager.getFreePage();
    zkr = ProgramPage::InitEmptyPage(page->programPage);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("HeaderPage::InitEmptyPage() failed calling ProgramPage::InitEmptyPage() result=" + zkresult2string(zkr));
        return zkr;
    }
    
    return ZKR_SUCCESS;
}

uint64_t HeaderPage::GetLastVersion (const uint64_t pageNumber)
{
    // Get the header page
    HeaderStruct * page = (HeaderStruct *)pageManager.getPage(pageNumber);

    return page->lastVersion;
}

void HeaderPage::SetLastVersion (const uint64_t pageNumber, const uint64_t lastVersion)
{
    // Get the header page
    HeaderStruct * page = (HeaderStruct *)pageManager.getPage(pageNumber);

    zkassert(lastVersion == page->lastVersion + 1);

    page->lastVersion = lastVersion;
}

void HeaderPage::Print (const uint64_t pageNumber, bool details)
{
    // Get page
    HeaderStruct * page = (HeaderStruct *)pageManager.getPage(pageNumber);
    zklog.info("HeaderPage::Print() pageNumber=" + to_string(pageNumber));

    // Print last version
    zklog.info("  lastVersion=" + to_string(page->lastVersion));

    // Print root-version page list
    zklog.info("  rootVersionPage=" + to_string(page->rootVersionPage) + "=" + to_string((uint64_t)pageManager.getPage(page->rootVersionPage)));
    RootVersionPage::Print(page->rootVersionPage, details);

    // Print key-value page list
    zklog.info("  keyValuePage=" + to_string(page->keyValuePage) + "=" + to_string((uint64_t)pageManager.getPage(page->keyValuePage)));
    KeyValuePage::Print(page->keyValuePage, details);

    // Print raw data
    zklog.info("  firstRawDataPage=" + to_string(page->firstRawDataPage) + "=" + to_string((uint64_t)pageManager.getPage(page->firstRawDataPage)));
    zklog.info("  rawDataPage=" + to_string(page->rawDataPage) + "=" + to_string((uint64_t)pageManager.getPage(page->rawDataPage)));
    zklog.info("  rawDataOffset=" + to_string(page->rawDataOffset));

    // Program page
    zklog.info("  programPage=" + to_string(page->programPage) + "=" + to_string((uint64_t)pageManager.getPage(page->programPage)));
    ProgramPage::Print(page->programPage, details);
}