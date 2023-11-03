#include <string.h>
#include "raw_data_page.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"
#include "zkassert.hpp"
#include "page_manager.hpp"
#include "scalar.hpp"
#include "zkmax.hpp"
#include "constants.hpp"

zkresult RawDataPage::InitEmptyPage (PageContext &ctx, const uint64_t pageNumber)
{
    RawDataStruct * page = (RawDataStruct *)ctx.pageManager.getPageAddress(pageNumber);
    memset((void *)page, 0, 4096);
    page->nextPageNumberAndOffset = minOffset << 48;
    return ZKR_SUCCESS;
}

zkresult RawDataPage::Read (PageContext &ctx, const uint64_t _pageNumber, const uint64_t _offset, const uint64_t length, string &data)
{
    uint64_t pageNumber = _pageNumber;
    uint64_t offset = _offset;
    RawDataStruct * page = (RawDataStruct *)ctx.pageManager.getPageAddress(pageNumber);
    uint64_t pageOffset = page->nextPageNumberAndOffset >> 48;

    // Check offsets
    if (pageOffset < minOffset)
    {
        zklog.error("RawDataPage::Read() found too-small pageOffset=" + to_string(pageOffset) + " pageNumber=" + to_string(pageNumber) + " length=" + to_string(length));
        return ZKR_DB_ERROR;
    }
    if (pageOffset > maxOffset)
    {
        zklog.error("RawDataPage::Read() found too-big pageOffset=" + to_string(pageOffset) + " pageNumber=" + to_string(pageNumber) + " length=" + to_string(length));
        return ZKR_DB_ERROR;
    }
    if (offset < minOffset)
    {
        zklog.error("RawDataPage::Read() found too-small offset=" + to_string(offset) + " pageNumber=" + to_string(pageNumber) + " length=" + to_string(length));
        return ZKR_DB_ERROR;
    }
    if (offset > maxOffset)
    {
        zklog.error("RawDataPage::Read() found too-big offset=" + to_string(offset) + " pageNumber=" + to_string(pageNumber) + " length=" + to_string(length));
        return ZKR_DB_ERROR;
    }
    if (offset > pageOffset)
    {
        zklog.error("RawDataPage::Read() found offset=" + to_string(offset) + " > pageOffset=" + to_string(pageOffset) + " pageNumber=" + to_string(pageNumber) + " length=" + to_string(length));
        return ZKR_DB_ERROR;
    }

    uint64_t copiedBytes = 0;
    while (copiedBytes < length)
    {
        // If we run out of data in the current page, get the next one
        if (offset == maxOffset)
        { 
            RawDataStruct * page = (RawDataStruct *)ctx.pageManager.getPageAddress(pageNumber);
            uint64_t nextPageNumber = page->nextPageNumberAndOffset & U64Mask48;
            pageNumber = nextPageNumber;
            offset = minOffset;
        }

        // Get the pointer corresponding to the current page number
        RawDataStruct * page = (RawDataStruct *)ctx.pageManager.getPageAddress(pageNumber);

        // Calculate the amount of bytes to copy this time
        uint64_t pageRemainingBytes = maxOffset - offset;
        zkassert(pageRemainingBytes > 0);
        zkassert(copiedBytes < length);
        uint64_t bytesToCopy = zkmin(pageRemainingBytes, length - copiedBytes);

        // Copy data
        data.append((char *)page + offset, bytesToCopy);

        // Update counters
        offset += bytesToCopy;
        copiedBytes += bytesToCopy;
    }

    return ZKR_SUCCESS;
}

zkresult RawDataPage::Write (PageContext &ctx, uint64_t &pageNumber, const string &data)
{
    // Get the pointer corresponding to the current page number
    RawDataStruct *page = (RawDataStruct *)ctx.pageManager.getPageAddress(pageNumber);

    // Get page attributes
    uint64_t offset = page->nextPageNumberAndOffset >> 48;
    uint64_t nextPage = page->nextPageNumberAndOffset & U64Mask48;

    // Check attributes
    if (nextPage != 0)
    {
        zklog.error("RawDataPage::Write() 1 found non-zero nextPage=" + to_string(nextPage) + " pageNumber=" + to_string(pageNumber) + " length=" + to_string(data.size()));
        return ZKR_DB_ERROR;
    }
    if (offset < minOffset)
    {
        zklog.error("RawDataPage::Write() 1 found too-small offset=" + to_string(offset) + " pageNumber=" + to_string(pageNumber) + " length=" + to_string(data.size()));
        return ZKR_DB_ERROR;
    }
    if (offset > maxOffset)
    {
        zklog.error("RawDataPage::Write() 1 found too-big offset=" + to_string(offset) + " pageNumber=" + to_string(pageNumber) + " length=" + to_string(data.size()));
        return ZKR_DB_ERROR;
    }

    uint64_t copiedBytes = 0;
    while (copiedBytes < data.size())
    {
        // If we run out of space in the current page, get a new one
        if (offset == maxOffset)
        {
            uint64_t nextPageNumber = ctx.pageManager.getFreePage();
            InitEmptyPage(ctx, nextPageNumber);
            RawDataStruct * nextPage = (RawDataStruct *)ctx.pageManager.getPageAddress(nextPageNumber);
            page->nextPageNumberAndOffset = nextPageNumber | (maxOffset << 48);
            nextPage->previousPageNumber = pageNumber;
            pageNumber = nextPageNumber;
            page = nextPage;
            offset = minOffset;
        }

        // Check attributes
        if (nextPage != 0)
        {
            zklog.error("RawDataPage::Write() 2 found non-zero nextPage=" + to_string(nextPage) + " pageNumber=" + to_string(pageNumber) + " length=" + to_string(data.size()));
            return ZKR_DB_ERROR;
        }
        if (offset < minOffset)
        {
            zklog.error("RawDataPage::Write() 2 found too-small offset=" + to_string(offset) + " pageNumber=" + to_string(pageNumber) + " length=" + to_string(data.size()));
            return ZKR_DB_ERROR;
        }
        if (offset > maxOffset)
        {
            zklog.error("RawDataPage::Write() 2 found too-big offset=" + to_string(offset) + " pageNumber=" + to_string(pageNumber) + " length=" + to_string(data.size()));
            return ZKR_DB_ERROR;
        }

        // Calculate the amount of bytes to write this time
        uint64_t pageRemainingBytes = maxOffset - offset;
        zkassert(pageRemainingBytes > 0);
        zkassert(copiedBytes < data.size());
        uint64_t bytesToCopy = zkmin(pageRemainingBytes, data.size() - copiedBytes);

        // Copy data
        memcpy((char *)page + offset, data.c_str() + copiedBytes, bytesToCopy);
        //zklog.info("RawDataPage::Write() wrote bytes=" + to_string(bytesToCopy) + " page=" + to_string(pageNumber) + " offset=" + to_string(offset));

        // Update counters
        offset += bytesToCopy;
        copiedBytes += bytesToCopy;
        page->nextPageNumberAndOffset = offset << 48;
    }

    // If we run out of space in the current page, get a new one
    if (offset == maxOffset)
    {
        uint64_t nextPageNumber = ctx.pageManager.getFreePage();
        InitEmptyPage(ctx, nextPageNumber);
        RawDataStruct * nextPage = (RawDataStruct *)ctx.pageManager.getPageAddress(nextPageNumber);
        page->nextPageNumberAndOffset = nextPageNumber | (maxOffset << 48);
        nextPage->previousPageNumber = pageNumber;
        pageNumber = nextPageNumber;
        page = nextPage;
        offset = minOffset;
    }

    return ZKR_SUCCESS;
}

uint64_t RawDataPage::GetOffset (PageContext &ctx, const uint64_t pageNumber)
{
    RawDataStruct * page = (RawDataStruct *)ctx.pageManager.getPageAddress(pageNumber);
    uint64_t offset = page->nextPageNumberAndOffset >> 48;
    zkassert(offset >= minOffset);
    zkassert(offset <= maxOffset);
    return offset;
}

void RawDataPage::Print (PageContext &ctx, const uint64_t pageNumber, bool details, const string &prefix)
{
    zklog.info(prefix + "RawDataPage::Print() pageNumber=" + to_string(pageNumber));
    RawDataStruct * page = (RawDataStruct *)ctx.pageManager.getPageAddress(pageNumber);
    if (details)
    {
        zklog.info(prefix + "previousPageNumber=" + to_string(page->previousPageNumber) + " nextPageNumber=" + to_string(page->nextPageNumberAndOffset & U64Mask48) + " offset=" + to_string(page->nextPageNumberAndOffset >> 48));
    }
    if (page->previousPageNumber != 0)
    {
        Print(ctx, page->previousPageNumber, details, prefix + " ");
    }
}