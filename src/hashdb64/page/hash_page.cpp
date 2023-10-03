#include <string.h>
#include "hash_page.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"
#include "zkassert.hpp"
#include "page_manager.hpp"
#include "scalar.hpp"

zkresult HashPage::InitEmptyPage (const uint64_t pageNumber)
{
    HashStruct * page = (HashStruct *)pageManager.getPage(pageNumber);
    memset((void *)page, 0, 4096);
    return ZKR_SUCCESS;
}

zkresult HashPage::Read (const uint64_t pageNumber, const uint64_t position, string &hash)
{
    zkassert(position < 128);
    HashStruct * page = (HashStruct *)pageManager.getPage(pageNumber);
    hash.copy((char *)page->hash[position], 32, 0);
    return ZKR_SUCCESS;
}

zkresult HashPage::Write (const uint64_t pageNumber, const uint64_t position, const string &hash)
{
    zkassert(position < 128);
    zkassert(hash.size() == 32);
    HashStruct * page = (HashStruct *)pageManager.getPage(pageNumber);
    memcpy((uint8_t *)page->hash[position], hash.c_str(), 32);
    return ZKR_SUCCESS;
}

void HashPage::Print (const uint64_t pageNumber, bool details)
{
    zklog.info("HashPage::Print() pageNumber=" + to_string(pageNumber));
    if (details)
    {
        // Create a string with 32 zeros
        string zeroString;
        for (uint64_t i=0; i<32; i++) zeroString += (char)0;

        // For each entry of the page
        HashStruct * page = (HashStruct *)pageManager.getPage(pageNumber);
        for (uint64_t i=0; i<128; i++)
        {
            // Print hash
            string hash;
            hash.append((char *)page->hash[i], 32);
            if (hash != zeroString)
            {
                zklog.info("  i=" + to_string(i) + " hash=" + ba2string(hash));
            }     
        }
    }
}