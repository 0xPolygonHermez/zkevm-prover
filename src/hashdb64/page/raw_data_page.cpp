#include <string.h>
#include "raw_data_page.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"
#include "zkassert.hpp"
#include "page_manager.hpp"
#include "scalar.hpp"
#include "zkmax.hpp"

zkresult RawDataPage::InitEmptyPage (const uint64_t pageNumber)
{
    RawDataStruct * page = (RawDataStruct *)(pageNumber*4096);
    memset(page, 0, 4096);
    return ZKR_SUCCESS;
}

zkresult RawDataPage::Read (const uint64_t _pageNumber, const uint64_t _position, const uint64_t length, string &data)
{
    uint64_t pageNumber = _pageNumber;
    uint64_t position = _position;
    zkassert(position < 128);

    uint64_t copiedBytes = 0;
    while (copiedBytes < length)
    {
        // If we run out of data in the current page, get the next one
        if (position == maxPosition)
        {
            RawDataStruct * page = (RawDataStruct *)(pageNumber*4096);
            zkassert(page->nextPageNumber != 0);
            pageNumber = page->nextPageNumber;
            position = minPosition;
        }

        // Get the pointer corresponding to the current page number
        RawDataStruct * page = (RawDataStruct *)(pageNumber*4096);

        // Calculate the amount of bytes to copy this time
        uint64_t pageRemainingBytes = maxPosition - position;
        zkassert(pageRemainingBytes > 0);
        zkassert(copiedBytes < length);
        uint64_t bytesToCopy = zkmin(pageRemainingBytes, length - copiedBytes);

        // Copy data
        data.copy((char *)page + position, bytesToCopy, copiedBytes);

        // Update counters
        position += bytesToCopy;
        copiedBytes += bytesToCopy;
    }

    return ZKR_SUCCESS;
}

zkresult RawDataPage::Write (uint64_t &pageNumber, uint64_t &position, const string &data)
{
    zkassert(position >= minPosition);
    zkassert(position <= maxPosition);
    uint64_t copiedBytes = 0;
    while (copiedBytes < data.size())
    {
        // If we run out of space in the current page, get a new one
        if (position == maxPosition)
        {
            uint64_t nextPageNumber = pageManager.getFreePage();
            InitEmptyPage(nextPageNumber);
            RawDataStruct * nextPage = (RawDataStruct *)(nextPageNumber*4096);
            RawDataStruct * page = (RawDataStruct *)(pageNumber*4096);
            page->nextPageNumber = nextPageNumber;
            nextPage->previousPageNumber = pageNumber;
            pageNumber = nextPageNumber;
            position = minPosition;
        }

        // Get the pointer corresponding to the current page number
        RawDataStruct * page = (RawDataStruct *)(pageNumber*4096);

        // Calculate the amount of bytes to write this time
        uint64_t pageRemainingBytes = maxPosition - position;
        zkassert(pageRemainingBytes > 0);
        zkassert(copiedBytes < data.size());
        uint64_t bytesToCopy = zkmin(pageRemainingBytes, data.size() - copiedBytes);

        // Copy data
        memcpy((char *)page + position, data.c_str() + copiedBytes, bytesToCopy);

        // Update counters
        position += bytesToCopy;
        copiedBytes += bytesToCopy;
    }

    return ZKR_SUCCESS;
}

void RawDataPage::Print (const uint64_t pageNumber, bool details)
{
    zklog.info("RawDataPage::Print() pageNumber=" + to_string(pageNumber));
    if (details)
    {
        RawDataStruct * page = (RawDataStruct *)(pageNumber*4096);
        zklog.info("  previousPageNumber=" + to_string(page->previousPageNumber) + "  nextPageNumber=" + to_string(page->nextPageNumber));
    }
}