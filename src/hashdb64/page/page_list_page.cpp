#include <string.h>
#include "page_list_page.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"
#include "zkassert.hpp"
#include "page_manager.hpp"
#include "scalar.hpp"
#include "zkmax.hpp"

zkresult PageListPage::InitEmptyPage (const uint64_t pageNumber)
{
    PageListStruct * page = (PageListStruct *)pageManager.getPageAddress(pageNumber);
    memset((void *)page, 0, 4096);
    page->nextPageNumberAndOffset = minOffset << 48;
    return ZKR_SUCCESS;
}

zkresult PageListPage::InsertPage (uint64_t &pageNumber, const uint64_t pageNumberToInsert)
{
    // Get page
    PageListStruct * page = (PageListStruct *)pageManager.getPageAddress(pageNumber);

    // Check page attributes
    uint64_t offset = page->nextPageNumberAndOffset >> 48;
    zkassert(offset >= minOffset);
    zkassert(offset <= maxOffset);
    zkassert((offset & 0x7) == 0);
    uint64_t nextPageNumber = page->nextPageNumberAndOffset & 0xFFFFFF;
    zkassert(nextPageNumber == 0);

    // If page is full, create a new one
    if (offset == maxOffset)
    {
        uint64_t nextPageNumber = pageManager.getFreePage();
        PageListStruct * nextPage = (PageListStruct *)pageManager.getPageAddress(nextPageNumber);
        nextPage->previousPageNumber = pageNumber;
        page->nextPageNumberAndOffset = nextPageNumber | (maxOffset << 48);
        pageNumber = nextPageNumber;
        page = (PageListStruct *)pageManager.getPageAddress(pageNumber);
    }

    // Check page attributes
    offset = page->nextPageNumberAndOffset >> 48;
    zkassert(offset >= minOffset);
    zkassert(offset <= maxOffset);
    zkassert((offset & 0x7) == 0);
    nextPageNumber = page->nextPageNumberAndOffset & 0xFFFFFF;
    zkassert(nextPageNumber == 0);

    // Insert the page number
    *(uint64_t *)((uint8_t *)page + offset) = pageNumberToInsert;

    // Update the page offset
    page->nextPageNumberAndOffset = (offset + 8) << 48;

    return ZKR_SUCCESS;
}

zkresult PageListPage::ExtractPage (uint64_t &pageNumber, uint64_t &extractedPageNumber)
{
    // Get page
    PageListStruct * page = (PageListStruct *)pageManager.getPageAddress(pageNumber);

    // Check page attributes
    uint64_t offset = page->nextPageNumberAndOffset >> 48;
    zkassert(offset >= minOffset);
    zkassert(offset <= maxOffset);
    zkassert((offset & 0x7) == 0);
    uint64_t nextPageNumber = page->nextPageNumberAndOffset & 0xFFFFFF;
    zkassert(nextPageNumber == 0);

    // Release any empty page
    while (offset == minOffset)
    {
        // Release previous page
        uint64_t previousPageNumber = page->previousPageNumber;
        pageManager.releasePage(pageNumber);

        // If this is the last page, return
        if (previousPageNumber == 0)
        {
            pageNumber = 0;
            extractedPageNumber = 0;
            return ZKR_DB_KEY_NOT_FOUND;
        }

        // Replace page
        pageNumber = previousPageNumber;
        page = (PageListStruct *)pageManager.getPageAddress(pageNumber);
        offset = page->nextPageNumberAndOffset >> 48;
    }

    // Extract a page number
    zkassert((offset - minOffset) >= 8);
    offset -= 8;
    extractedPageNumber = *(uint64_t *)((uint8_t *)page + offset);
    page->nextPageNumberAndOffset = offset << 48;

    // Release any empty page
    while (offset == minOffset)
    {
        // Release previous page
        uint64_t previousPageNumber = page->previousPageNumber;
        pageManager.releasePage(pageNumber);

        // If this is the last page, return
        if (previousPageNumber == 0)
        {
            pageNumber = 0;
            break;
        }

        // Replace page
        pageNumber = previousPageNumber;
        page = (PageListStruct *)pageManager.getPageAddress(pageNumber);
        offset = page->nextPageNumberAndOffset >> 48;
    }

    return ZKR_SUCCESS;
}

void PageListPage::Print (const uint64_t pageNumber, bool details)
{
    zklog.info("PageListPage::Print() pageNumber=" + to_string(pageNumber));
    if (details)
    {
        PageListStruct * page = (PageListStruct *)pageManager.getPageAddress(pageNumber);
        zklog.info("  previousPageNumber=" + to_string(page->previousPageNumber) + "  nextPageNumber=" + to_string(page->nextPageNumberAndOffset & 0xFFFFFF) + " offset=" + to_string(page->nextPageNumberAndOffset >> 48));
    }
}