#include "root_version_page.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"
#include "zkassert.hpp"
#include "scalar.hpp"
#include "page_manager.hpp"

zkresult RootVersionPage::InitEmptyPage (const uint64_t pageNumber)
{
    RootVersionStruct * page = (RootVersionStruct *)(pageNumber*4096);
    memset(page, 0, 4096);
    return ZKR_SUCCESS;
}

zkresult RootVersionPage::Read (const uint64_t pageNumber, const string &root, uint64_t &version, const uint64_t level)
{
    zkassert(root.size() == 32);
    zkassert(level < 32);

    // Get the data from this page
    RootVersionStruct * page = (RootVersionStruct *)(pageNumber*4096);
    uint8_t levelBits = root[level];
    uint64_t versionAndControl = page->versionAndControl[levelBits];
    uint64_t foundVersion = versionAndControl & 0xFFFFFF;
    uint64_t control = versionAndControl >> 24;

    // Leaf node
    if (control == 1)
    {
        version = foundVersion;
        return ZKR_SUCCESS;
    }

    // Intermediate node
    if (control == 2)
    {
        return Read(pageNumber, root, version, level+1);
    }

    // Not found
    return ZKR_DB_KEY_NOT_FOUND;
}

zkresult RootVersionPage::Read (const uint64_t pageNumber, const string &root, uint64_t &version)
{
    zkassert(root.size() == 32);

    return Read(pageNumber, root, version, 0);
}

zkresult RootVersionPage::Write (uint64_t &pageNumber, const string &root, const uint64_t version, const uint64_t level)
{
    zkassert(root.size() == 32);
    zkassert(level < 32);

    // Get the data from this page
    RootVersionStruct * page = (RootVersionStruct *)(pageNumber*4096);
    uint8_t levelBits = root[level];
    uint64_t versionAndControl = page->versionAndControl[levelBits];
    uint64_t foundVersion = versionAndControl & 0xFFFFFF;
    uint64_t control = versionAndControl >> 48;

    // Check control
    switch (control)
    {
        // Empty slot
        case 0:
        {
            
        }
        // Leaf node
        case 1:
        {

        }

        // Intermediate node
        case 2:
        {
            return Write(pageNumber, root, version, level+1);
        }
        default:
        {
            zklog.error("RootVersionPage::Write() found invalid control=" + to_string(control) + " pageNumber=" + to_string(pageNumber));
            return ZKR_DB_ERROR;
        }
    }
}

zkresult RootVersionPage::Write (uint64_t &pageNumber, const string &root, const uint64_t version)
{
    zkassert(root.size() == 32);

    // Start searching with level 0
    return Write(pageNumber, root, version, 0);
}

void RootVersionPage::Print (const uint64_t pageNumber, bool details)
{
    RootVersionStruct * page = (RootVersionStruct *)(pageNumber*4096);

    zklog.info("RootVersionPage::Print() pageNumber=" + to_string(pageNumber));

    // Print entries
    if (details)
    {
        for (uint64_t i=0; i<128; i++)
        {
            uint64_t versionAndControl = page->versionAndControl[i];
            uint64_t version = versionAndControl & 0xFFFFFF;
            uint64_t control = versionAndControl >> 24;
            zklog.info("  i=" + to_string(i) + " control=" + to_string(control) + " version=" + to_string(version));
        }
    }

    // Iterate over the next page
    if (page->nextPage != 0)
    {
        Print(page->nextPage, details);
    }
}