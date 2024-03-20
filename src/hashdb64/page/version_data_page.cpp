#include "version_data_page.hpp"
#include "scalar.hpp"
#include "zkassert.hpp"
#include "key_utils.hpp"
#include "constants.hpp"
#include "header_page.hpp"
#include "zkmax.hpp"
#include "zkresult.hpp"


zkresult VersionDataPage::InitEmptyPage (PageContext &ctx, const uint64_t pageNumber)
{
    VersionDataStruct * page = (VersionDataStruct *)ctx.pageManager.getPageAddress(pageNumber);
    memset((void *)page, 0, 4096);
    return ZKR_SUCCESS;
}

zkresult VersionDataPage::Read (PageContext &ctx, const uint64_t pageNumber, const uint64_t key, VersionDataEntry &value)
{
    // Get 64 key bits, in sets of 6 bits
    uint8_t keyBitsArray[11];
    splitKey6(fr, key, keyBitsArray);
    string keyBits;
    keyBits.append((char *)keyBitsArray, 11);

    return Read(ctx, pageNumber, key, keyBits, value, 0);
}

zkresult VersionDataPage::Read (PageContext &ctx, const uint64_t pageNumber, const uint64_t key, const string &keyBits, VersionDataEntry &value, const uint64_t level)
{
    zkassert(keyBits.size() == 11);
    zkassert(level < 11);

    // Get the data from this page
    VersionDataStruct * page = (VersionDataStruct *)ctx.pageManager.getPageAddress(pageNumber);
    uint64_t index = keyBits[level];
    uint64_t control = page->versionDataEntry[index].controlAndFreePagesList >> 48;

    // Check control
    switch (control)
    {
        // Empty slot
        case 0:
        {
            return ZKR_DB_KEY_NOT_FOUND;
        }

        // Leaf node
        case 1:
        {
            memcpy(&value, &(page->versionDataEntry[index]), sizeof(VersionDataEntry));
            value.controlAndFreePagesList &= U64Mask48;
            return ZKR_SUCCESS;
        }

        // Intermediate node
        case 2:
        {
            uint64_t nextPageNumber = page->versionDataEntry[index].controlAndFreePagesList & U64Mask48;
            return Read(ctx, nextPageNumber, key, keyBits, value, level + 1);
        }

        default:
        {
            zklog.error("VersionDataPage::Read() found invalid control=" + to_string(control) + " pageNumber=" + to_string(pageNumber));
            return ZKR_DB_ERROR;
        }
    }

    // Not found
    return ZKR_DB_KEY_NOT_FOUND;
}

zkresult VersionDataPage::Write (PageContext &ctx, uint64_t &pageNumber, const uint64_t key, const VersionDataEntry &value, uint64_t &headerPageNumber)
{
    // Get 64 key bits, in sets of 6 bits
    uint8_t keyBitsArray[11];
    splitKey6(fr, key, keyBitsArray);
    string keyBits;
    keyBits.append((char *)keyBitsArray, 11);

    // Start searching with level 0
    zkresult zkr;
    zkr = Write(ctx, pageNumber, key, keyBits, value, 0, headerPageNumber);

    // Edit the header version data page
    if (zkr == ZKR_SUCCESS)
    {
        headerPageNumber = ctx.pageManager.editPage(headerPageNumber);
        HeaderStruct *headerPage = (HeaderStruct *)ctx.pageManager.getPageAddress(headerPageNumber);
        headerPage->versionDataPage = pageNumber;
    }
    return zkr;
}

zkresult VersionDataPage::Write (PageContext &ctx, uint64_t &pageNumber, const uint64_t key, const string &keyBits,const VersionDataEntry &value, const uint64_t level, uint64_t &headerPageNumber)
{
    zkassert(keyBits.size() == 11);
    zkassert(level < 11);
    zkassert(key == value.key);
    zkassert((value.controlAndFreePagesList & U64Mask48) == value.controlAndFreePagesList);

    zkresult zkr;

    // Get the data from this page
    pageNumber = ctx.pageManager.editPage(pageNumber);
    VersionDataStruct * page = (VersionDataStruct *)ctx.pageManager.getPageAddress(pageNumber);
    uint64_t index = keyBits[level];
    uint64_t control = page->versionDataEntry[index].controlAndFreePagesList >> 48;

    // Check control
    switch (control)
    {
        // Empty slot
        case 0:
        {
            // Get an editable version of this page
            pageNumber = ctx.pageManager.editPage(pageNumber);
            VersionDataStruct * page = (VersionDataStruct *)ctx.pageManager.getPageAddress(pageNumber);

            memcpy(&(page->versionDataEntry[index]), &value, sizeof(VersionDataEntry));
            page->versionDataEntry[index].controlAndFreePagesList |= (uint64_t(1) << 48);

            return ZKR_SUCCESS;
        }

        // Leaf node
        case 1:
        {
            // If the key is the same, overwrite the value
            if (key == page->versionDataEntry[index].key)
            {
                memcpy(&(page->versionDataEntry[index]), &value, sizeof(VersionDataEntry));
                page->versionDataEntry[index].controlAndFreePagesList |= uint64_t(0x1) << 48;
                return ZKR_SUCCESS;
            }

            // If the key is different, move the key to a new KeyValuePage, and write the new key into the new page
            uint64_t newPageNumber = ctx.pageManager.getFreePage();
            VersionDataPage::InitEmptyPage(ctx, newPageNumber);
            VersionDataStruct *newPage = (VersionDataStruct *)ctx.pageManager.getPageAddress(newPageNumber);

            // Get 64 key bits, in sets of 6 bits
            uint8_t keyBitsArray[11];
            splitKey6(fr, page->versionDataEntry[index].key, keyBitsArray);
            string foundKeyBits;
            foundKeyBits.append((char *)keyBitsArray, 11);
            uint64_t newIndex = foundKeyBits[level+1];
            memcpy(&(newPage->versionDataEntry[newIndex]), &(page->versionDataEntry[index]), sizeof(VersionDataEntry));

            zkr = Write(ctx, newPageNumber, key, keyBits, value, level+1, headerPageNumber);
            if (zkr != ZKR_SUCCESS)
            {
                return zkr;
            }

            if (zkr == ZKR_SUCCESS)
            {
                memset(&(page->versionDataEntry[index]), 0, sizeof(VersionDataEntry));
                page->versionDataEntry[index].controlAndFreePagesList = (uint64_t(0x2) << 48) | newPageNumber;
            }

            return zkr;
        }

        // Intermediate node
        case 2:
        {
            // Call Write with the next page number, which can be modified in it runs out of history
            uint64_t oldNextPageNumber = page->versionDataEntry[index].controlAndFreePagesList & U64Mask48;
            uint64_t newNextPageNumber = oldNextPageNumber;
            zkr = Write(ctx, newNextPageNumber, key, keyBits, value, level + 1, headerPageNumber);
            // newNextPageNumber can be modified in the Write call
            page->versionDataEntry[index].controlAndFreePagesList = (uint64_t(0x2) << 48) | newNextPageNumber;

            return zkr;
        }

        default:
        {
            zklog.error("VersionDataPage::Write() found invalid control=" + to_string(control) + " pageNumber=" + to_string(pageNumber));
            return ZKR_DB_ERROR;
        }
    }
}

void VersionDataPage::Print (PageContext &ctx, const uint64_t pageNumber, bool details, const string &prefix, const uint64_t level, VersionDataCounters &counters)
{
    zklog.info(prefix + "VersionDataPage::Print() pageNumber=" + to_string(pageNumber));

    counters.maxLevel = zkmax(counters.maxLevel, (level * 6) + 5);

    vector<uint64_t> nextVersionDataPages;

    // Get the data from this page
    VersionDataStruct * page = (VersionDataStruct *)ctx.pageManager.getPageAddress(pageNumber);

    for (uint64_t i=0; i<64; i++)
    {
        uint64_t control = (page->versionDataEntry[i].controlAndFreePagesList) >> 48;

        // Check control
        switch (control)
        {
            // Empty slot
            case 0:
            {
                continue;
            }

            // Leaf node
            case 1:
            {
                counters.leafNodes++;

                if (details)
                {
                    string rootBa;
                    rootBa.append((const char *)page->versionDataEntry[i].root, 32);
                    zklog.info(prefix +
                        "i=" + to_string(i) +
                        " key=" + to_string(page->versionDataEntry[i].key) +
                        " root=" + ba2string(rootBa) +
                        " keyValueHistoryPage=" + to_string(page->versionDataEntry[i].keyValueHistoryPage) +
                        " freePagesList=" + to_string(page->versionDataEntry[i].controlAndFreePagesList & U64Mask48) +
                        " createdPagesList=" + to_string(page->versionDataEntry[i].createdPagesList));
                }
                continue;
            }

            // Intermediate node
            case 2:
            {
                counters.intermediateNodes++;

                uint64_t nextValueDataPage = page->versionDataEntry[i].controlAndFreePagesList & U64Mask48;
                nextVersionDataPages.emplace_back(nextValueDataPage);

                if (details)
                {
                    zklog.info(prefix + "i=" + to_string(i) + " nextValueDataPage=" + to_string(nextValueDataPage));
                }
                continue;
            }

            default:
            {
                zklog.error(prefix + "VersionDataPagePage::Print() found invalid control=" + to_string(control) + " pageNumber=" + to_string(pageNumber));
                exitProcess();
            }
        }
    }

    for (uint64_t i=0; i<nextVersionDataPages.size(); i++)
    {
        Print(ctx, nextVersionDataPages[i], details, prefix + " ", level + 1, counters);
    }
}

void VersionDataPage::Print (PageContext &ctx, const uint64_t pageNumber, bool details, const string &prefix)
{
    zklog.info(prefix + "VersionDataPage::Print()");
    VersionDataCounters counters;
    Print(ctx, pageNumber, details, prefix, 0, counters);
    zklog.info(prefix + "Counters: leafNodes=" + to_string(counters.leafNodes) + " intermediateNodes=" + to_string(counters.intermediateNodes) + " maxLevel=" + to_string(counters.maxLevel));
}