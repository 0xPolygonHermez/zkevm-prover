#include <string.h>
#include "key_value_page.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"
#include "zkassert.hpp"
#include "page_manager.hpp"
#include "scalar.hpp"
#include "raw_data_page.hpp"
#include "header_page.hpp"

zkresult KeyValuePage::InitEmptyPage (const uint64_t pageNumber)
{
    KeyValueStruct * page = (KeyValueStruct *)pageManager.getPageAddress(pageNumber);
    memset((void *)page, 0, 4096);
    return ZKR_SUCCESS;
}

zkresult KeyValuePage::Read (const uint64_t pageNumber, const string &key, string &value, const uint64_t level)
{
    // Check input parameters
    if (level >= key.size())
    {
        zklog.error("KeyValuePage::Read() got invalid level=" + to_string(level) + " > key.size=" + to_string(key.size()));
        return ZKR_DB_ERROR;
    }

    zkresult zkr;
    KeyValueStruct * page = (KeyValueStruct *)pageManager.getPageAddress(pageNumber);
    uint8_t index = key[level];

    uint64_t control = page->key[index][0] >> 48;

    switch (control)
    {
        // Empty slot
        case 0:
        {
            zklog.error("KeyValuePage::Read() found empty slot pageNumber=" + to_string(pageNumber) + " index=" + to_string(index) + " level=" + to_string(level) + " key=" + ba2string(key));
            return ZKR_DB_KEY_NOT_FOUND;
        }

        // Leaf node
        case 1:
        {
            uint64_t length = page->key[index][0] & 0xFFFFFF;
            uint64_t rawDataPage = page->key[index][1] & 0xFFFFFF;
            uint64_t rawDataOffset = page->key[index][1] >> 48;
            string rawdata;
            zkr = RawDataPage::Read(rawDataPage, rawDataOffset, length, rawdata);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("KeyValuePage::Read() failed calling RawDataPage::Read() result=" + zkresult2string(zkr) + " pageNumber=" + to_string(pageNumber) + " index=" + to_string(index) + " level=" + to_string(level) + " key=" + ba2string(key));
                return zkr;
            }

            if (rawdata.size() < key.size())
            {
                zklog.error("KeyValuePage::Read() called RawDataPage::Read() and got invalid raw data size=" + to_string(rawdata.size()) + " pageNumber=" + to_string(pageNumber) + " index=" + to_string(index) + " level=" + to_string(level) + " key=" + ba2string(key));
                return ZKR_DB_ERROR;
            }

            if (memcmp(rawdata.c_str(), key.c_str(), key.size()) != 0)
            {
                zklog.error("KeyValuePage::Read() called RawDataPage::Read() and got different keys pageNumber=" + to_string(pageNumber) + " index=" + to_string(index) + " level=" + to_string(level) + " key=" + ba2string(key));
                return ZKR_DB_KEY_NOT_FOUND;
            }
            
            zklog.info("KeyValuePage::Read() read length=" + to_string(length) +
                " result=" + zkresult2string(zkr) +
                " pageNumber=" + to_string(pageNumber) +
                " index=" + to_string(index) +
                " level=" + to_string(level) +
                " key=" + ba2string(key) +
                " rawDataPage=" + to_string(rawDataPage) +
                " rawDataOffset=" + to_string(rawDataOffset));
            
            value = rawdata.substr(key.size());

            return ZKR_SUCCESS;
        }

        // Intermediate node
        case 2:
        {
            uint64_t nextKeyValuePage = page->key[index][0] & 0xFFFFFF;
            return Read(nextKeyValuePage, key, value, level+1);
        }

        // Default
        default:
        {
            zklog.error("KeyValuePage::Read() found invalid control=" + to_string(control) + " pageNumber=" + to_string(pageNumber) + " index=" + to_string(index) + " level=" + to_string(level) + " key=" + ba2string(key));
            exitProcess();
        }
    }

    return ZKR_DB_ERROR;
}

zkresult KeyValuePage::Read (const uint64_t pageNumber, const string &key, string &value)
{
    // Clear value
    value.clear();

    // Call Read with level=0
    return Read(pageNumber, key, value, 0);
}

zkresult KeyValuePage::Write (const uint64_t pageNumber, const string &key, const string &value, const uint64_t level, const uint64_t headerPageNumber)
{
    // Check input parameters
    if (level >= key.size())
    {
        zklog.error("KeyValuePage::write() got invalid level=" + to_string(level) + " > key.size=" + to_string(key.size()));
        return ZKR_DB_ERROR;
    }

    if (key.size() + value.size() > 0xFFFFFF)
    {
        zklog.error("KeyValuePage::write() got invalid value.size=" + to_string(value.size()) + " + key.size=" + to_string(key.size()) + " > 0xFFFFFF");
        return ZKR_DB_ERROR;
    }

    zkresult zkr;
    KeyValueStruct * page = (KeyValueStruct *)pageManager.getPageAddress(pageNumber);
    uint8_t index = key[level];

    uint64_t control = page->key[index][0] >> 48;

    switch (control)
    {
        // Empty slot: insert the new key here
        case 0:
        {
            //zklog.error("KeyValuePage::Write() found empty slot pageNumber=" + to_string(pageNumber) + " index=" + to_string(index) + " level=" + to_string(level) + " key=" + ba2string(key));
            string keyAndValue = key + value;
            uint64_t length = keyAndValue.size();

            // Get header page data
            HeaderStruct * headerPage = (HeaderStruct *)pageManager.getPageAddress(headerPageNumber);
            uint64_t rawDataPage = headerPage->rawDataPage;
            uint64_t rawDataOffset = RawDataPage::GetOffset(headerPage->rawDataPage);

            // Write to raw data page list
            zkr = RawDataPage::Write(rawDataPage, keyAndValue);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("KeyValuePage::Write() failed calling RawDataPage::Write() result=" + zkresult2string(zkr) + " pageNumber=" + to_string(pageNumber) + " index=" + to_string(index) + " level=" + to_string(level) + " key=" + ba2string(key));
                return zkr;
            }
            else
            {
                zklog.info("KeyValuePage::Write() wrote length=" + to_string(length) +
                    " result=" + zkresult2string(zkr) +
                    " pageNumber=" + to_string(pageNumber) +
                    " index=" + to_string(index) +
                    " level=" + to_string(level) +
                    " key=" + ba2string(key) +
                    " rawDataPage=" + to_string(headerPage->rawDataPage) +
                    " rawDataOffset=" + to_string(rawDataOffset) +
                    " leaving headerPage->rawDataPage=" + to_string(rawDataPage) +
                    " headerPage->rawDataOffset=" + to_string(rawDataOffset));
            }
            

            // Update this entry as a leaf node (control = 1)
            page->key[index][0] = length | (uint64_t(1)<<48);
            page->key[index][1] = headerPage->rawDataPage | (rawDataOffset << 48);

            // Update header page data
            headerPage->rawDataPage = rawDataPage;

            return ZKR_SUCCESS;
        }

        // Leaf node
        case 1:
        {
            // Read the key from the raw data page
            uint64_t length = page->key[index][0] & 0xFFFFFF;
            uint64_t rawPageNumber = page->key[index][1] & 0xFFFFFF;
            uint64_t rawPageOffset = page->key[index][1] >> 48;

            if (length < key.size())
            {
                zklog.error("KeyValuePage::Write() found invalid length=" + to_string(length) + " pageNumber=" + to_string(pageNumber) + " index=" + to_string(index) + " level=" + to_string(level) + " key=" + ba2string(key));
                exitProcess();
            }

            // Get the existing key
            string existingKey;
            zkr = RawDataPage::Read(rawPageNumber, rawPageOffset, key.size(), existingKey);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("KeyValuePage::Write() failed calling RawDataPage::Read() result=" + zkresult2string(zkr) + " length=" + to_string(length) + " pageNumber=" + to_string(pageNumber) + " index=" + to_string(index) + " level=" + to_string(level) + " key=" + ba2string(key));
                return zkr;
            }

            // If both keys are identical, values should be identical, too
            if (existingKey == key)
            {
                // Compare total length
                if (length != key.size() + value.size())
                {
                    zklog.error("KeyValuePage::Write() found not matching length=" + to_string(length) + " != key.size+value.size=" + to_string(key.size() + value.size()) + " pageNumber=" + to_string(pageNumber) + " index=" + to_string(index) + " level=" + to_string(level) + " key=" + ba2string(key));
                    exitProcess();
                }

                // Get the existing key + value
                string existingKeyAndValue;
                zkr = RawDataPage::Read(rawPageNumber, rawPageOffset, length, existingKeyAndValue);
                if (zkr != ZKR_SUCCESS)
                {
                    zklog.error("KeyValuePage::Write() failed calling RawDataPage::Read() result=" + zkresult2string(zkr) + " length=" + to_string(length) + " pageNumber=" + to_string(pageNumber) + " index=" + to_string(index) + " level=" + to_string(level) + " key=" + ba2string(key));
                    return zkr;
                }

                // Compare programs
                if (existingKeyAndValue.size() != key.size() + value.size())
                {
                    zklog.error("KeyValuePage::Write() found not matching existingKeyAndValue.size()=" + to_string(existingKeyAndValue.size()) + " != key.size+value.size=" + to_string(key.size() + value.size()) + " pageNumber=" + to_string(pageNumber) + " index=" + to_string(index) + " level=" + to_string(level) + " key=" + ba2string(key));
                    exitProcess();
                }
                if (value.compare(existingKeyAndValue.substr(key.size())) != 0)
                {
                    zklog.error("KeyValuePage::Write() found not matching value of size=" + to_string(value.size()) + " pageNumber=" + to_string(pageNumber) + " index=" + to_string(index) + " level=" + to_string(level) + " key=" + ba2string(key));
                    exitProcess();
                }

                return ZKR_SUCCESS;
            }

            // If keys are different, create a new page, move the existing key one level down, and call Write(level+1)
            uint64_t newPageNumber = pageManager.getFreePage();
            KeyValuePage::InitEmptyPage(newPageNumber);
            KeyValueStruct * newPage = (KeyValueStruct *)pageManager.getPageAddress(newPageNumber);
            uint8_t newIndex = existingKey[level+1];
            newPage->key[newIndex][0] = page->key[index][0];
            newPage->key[newIndex][1] = page->key[index][1];

            // Set this page entry as a intermeiate node
            page->key[index][0] = newPageNumber | (uint64_t(2) << 48);
            page->key[index][1] = 0;

            // Write the new key in the newly created page at the next level
            return KeyValuePage::Write(newPageNumber, key, value, level+1, headerPageNumber);
        }

        // Intermediate node
        case 2:
        {
            uint64_t nextKeyValuePage = page->key[index][0] & 0xFFFFFF;
            return Write(nextKeyValuePage, key, value, level+1, headerPageNumber);
        }

        // Default
        default:
        {
            zklog.error("KeyValuePage::Write() found invalid control=" + to_string(control) + " pageNumber=" + to_string(pageNumber) + " index=" + to_string(index) + " level=" + to_string(level) + " key=" + ba2string(key));
            exitProcess();
        }
    }

    return ZKR_DB_ERROR;
}

zkresult KeyValuePage::Write (const uint64_t pageNumber, const string &key, const string &value, const uint64_t headerPageNumber)
{
    // Check input parameters
    if (key.size() == 0)
    {
        zklog.error("KeyValuePage::Write() found key.size=0 pageNumber=" + to_string(pageNumber));
        exitProcess();
    }

    // Call Write with level=0
    return Write(pageNumber, key, value, 0, headerPageNumber);
}

void KeyValuePage::Print (const uint64_t pageNumber, bool details)
{
    zklog.info("KeyValuePage::Print() pageNumber=" + to_string(pageNumber));
    if (details)
    {
        vector<uint64_t> nextKeyValuePages;

        // For each entry of the page
        KeyValueStruct * page = (KeyValueStruct *)pageManager.getPageAddress(pageNumber);
        for (uint64_t i=0; i<256; i++)
        {
            uint64_t control = page->key[i][0] >> 48;

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
                    uint64_t length = page->key[i][0] & 0xFFFFFF;
                    uint64_t rawPageNumber = page->key[i][1] & 0xFFFFFF;
                    uint64_t rawPageOffset = page->key[i][1] >> 48;
                    zklog.info("  i=" + to_string(i) + " length=" + to_string(length) + " rawPageNumber=" + to_string(rawPageNumber) + " rawPageOffset=" + to_string(rawPageOffset));
                    continue;
                }

                // Intermediate node
                case 2:
                {
                    uint64_t nextKeyValuePage = page->key[i][0] & 0xFFFFFF;
                    nextKeyValuePages.emplace_back(nextKeyValuePage);
                    zklog.info("  i=" + to_string(i) + " nextKeyValuePage=" + to_string(nextKeyValuePage));
                    continue;
                }

                // Default
                default:
                {
                    zklog.error("KeyValuePage::Print() found invalid control=" + to_string(control) + " pageNumber=" + to_string(pageNumber) + " i=" + to_string(i));
                    exitProcess();
                }

            }      
        }

        for (uint64_t i=0; i<nextKeyValuePages.size(); i++)
        {
            Print(nextKeyValuePages[i], details);
        }
    }
}