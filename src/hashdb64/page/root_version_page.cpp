#include "root_version_page.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"
#include "zkassert.hpp"
#include "scalar.hpp"
#include "page_manager.hpp"

zkresult RootVersionPage::InitEmptyPage (const uint64_t pageNumber)
{
    RootVersionStruct * page = (RootVersionStruct *)(pageNumber*4096);
    page->nextPage = 0;
    page->size = headerSize;
    return ZKR_SUCCESS;
}

zkresult RootVersionPage::Read (const uint64_t pageNumber, const string &root, uint64_t &version)
{
    zkassert(root.size() == 32);
    RootVersionStruct * page = (RootVersionStruct *)(pageNumber*4096);

    // search in page->version from the last entry to the first entry, i.e. in reverse order
    for (int64_t i=(page->size-16)/entrySize; i > 0; i--)
    {
        RootVersion *rootVersion = (RootVersion *)((uint8_t *)page + headerSize + entrySize*i);
        if (root == rootVersion->root)
        {
            version = rootVersion->version;
            return ZKR_SUCCESS;
        }
    }

    // if not found, search in nextPage (older versions)
    if (page->nextPage != 0)
    {
        return Read(page->nextPage, root, version);
    }
    else
    {
        zklog.error("RootVersionPage::Read() could not find root=" + ba2string(root));
        return ZKR_DB_KEY_NOT_FOUND;
    }
}

zkresult RootVersionPage::Write (uint64_t &pageNumber, const string &root, uint64_t version)
{
    zkassert(root.size() == 32);
    RootVersionStruct * page = (RootVersionStruct *)(pageNumber*4096);

    // If there is room for a new entry in this page, we use it
    if (page->size < maxSize)
    {
        RootVersion *rootVersion = (RootVersion *)((uint8_t *)page + page->size);
        memcpy(rootVersion->root, root.c_str(), 32);
        rootVersion->version = version;
        page->size += entrySize;
        return ZKR_SUCCESS;
    }

    // Allocate a new page and link it to the current page
    uint64_t newPageNumber = pageManager.getFreeMemoryPage();
    RootVersionStruct * newPage = (RootVersionStruct *)(newPageNumber*4096);
    newPage->nextPage = (uint64_t)page/4096;

    // Create a new entry in the first position
    RootVersion *rootVersion = (RootVersion *)((uint8_t *)newPage + headerSize);
    memcpy(rootVersion->root, root.c_str(), 32);
    rootVersion->version = version;
    newPage->size = headerSize + entrySize;

    // Overwrite the page number with the new one
    pageNumber = newPageNumber;

    return ZKR_SUCCESS;
}