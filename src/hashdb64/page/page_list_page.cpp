#include <string.h>
#include "page_list_page.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"
#include "zkassert.hpp"
#include "page_manager.hpp"
#include "scalar.hpp"
#include "zkmax.hpp"
#include "constants.hpp"

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
    uint64_t nextPageNumber = page->nextPageNumberAndOffset & U64Mask48;
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
    nextPageNumber = page->nextPageNumberAndOffset & U64Mask48;
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
    uint64_t nextPageNumber = page->nextPageNumberAndOffset & U64Mask48;
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

zkresult PageListPage::GetPages (const uint64_t pageNumber, vector<uint64_t> (&containerPages), vector<uint64_t> (&containedPages))
{
    // Get page
    PageListStruct * page = (PageListStruct *)pageManager.getPageAddress(pageNumber);

    // Get the previous pages
    if (page->previousPageNumber != 0)
    {
        // Call GetPages of previous page number
        zkresult zkr;
        zkr = GetPages(page->previousPageNumber, containerPages, containedPages);
        if (zkr != ZKR_SUCCESS)
        {
            return zkr;
        }
    }

    // Get offset
    uint64_t offset = page->nextPageNumberAndOffset >> 48;
    zkassert(offset >= minOffset);
    zkassert(offset <= maxOffset);
    zkassert((offset & 0x7) == 0);

    // Add all pages contained in this page
    for (uint64_t i = minOffset; i < offset; i += 8)
    {
        containedPages.emplace_back(*(uint64_t *)((uint8_t *)page + offset));
    }

    // Add this page number to container pages
    containerPages.emplace_back(pageNumber);

    return ZKR_SUCCESS;
}

zkresult PageListPage::CreatePages (uint64_t &pageNumber, vector<uint64_t> (&freePages), vector<uint64_t> (&containerPages), vector<uint64_t> (&containedPages))
{
    if (freePages.size() == 0)
    {
        pageNumber = 0;
        return ZKR_SUCCESS;
    }

    // Create a new container page
    pageNumber = freePages[freePages.size() - 1];
    freePages.pop_back();
    containerPages.emplace_back(pageNumber);

    // Init the page
    zkresult zkr = PageListPage::InitEmptyPage(pageNumber);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("PageListPage::CreatePages() failed calling PageListPage::InitEmptyPage() result=" + zkresult2string(zkr) + " pageNumber=" + to_string(pageNumber));
        return zkr;
    }

    // Get page
    PageListStruct * page = (PageListStruct *)pageManager.getPageAddress(pageNumber);

    // Get offset
    uint64_t offset = page->nextPageNumberAndOffset >> 48;
    zkassert(offset >= minOffset);
    zkassert(offset <= maxOffset);
    zkassert((offset & 0x7) == 0);

    // Insert as many pages as possible
    while ((freePages.size() > 0) && (offset < maxOffset))
    {
        // Insert a new contained page
        uint64_t containedPageNumber = freePages[freePages.size() - 1];
        freePages.pop_back();
        containedPages.emplace_back(containedPageNumber);

        // Insert the page number
        *(uint64_t *)((uint8_t *)page + offset) = containedPageNumber;

        // Increase offset
        offset += 8;
    }

    // Update the page offset
    page->nextPageNumberAndOffset = (offset + 8) << 48;

    // If there are more pages left, fill the previous page
    if (freePages.size() > 0)
    {
        return CreatePages(page->previousPageNumber, freePages, containerPages, containedPages);
    }

    return ZKR_SUCCESS;
}

void PageListPage::Print (const uint64_t pageNumber, bool details, const string &prefix)
{
    zklog.info("PageListPage::Print() pageNumber=" + to_string(pageNumber));
    PageListStruct * page = (PageListStruct *)pageManager.getPageAddress(pageNumber);
    if (details)
    {
        zklog.info("  previousPageNumber=" + to_string(page->previousPageNumber) + "  nextPageNumber=" + to_string(page->nextPageNumberAndOffset & U64Mask48) + " offset=" + to_string(page->nextPageNumberAndOffset >> 48));
    }
    if (page->previousPageNumber != 0)
    {
        Print(page->previousPageNumber, details, prefix + " ");
    }
}