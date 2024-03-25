#include <string.h>
#include "key_value_page.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"
#include "zkassert.hpp"
#include "page_manager.hpp"
#include "scalar.hpp"
#include "raw_data_page.hpp"
#include "header_page.hpp"
#include "key_utils.hpp"
#include "constants.hpp"

//#define LOG_KEY_VALUE_PAGE

zkresult KeyValuePage::InitEmptyPage (const uint64_t pageNumber)
{
    KeyValueStruct * page = (KeyValueStruct *)pageManager.getPageAddress(pageNumber);
    memset((void *)page, 0, 4096);
    return ZKR_SUCCESS;
}

zkresult KeyValuePage::Read (const uint64_t pageNumber, const string &key, const vector<uint64_t> &keyBits, string &value, const uint64_t level)
{
    // Check input parameters
    if (level >= keyBits.size())
    {
        zklog.error("KeyValuePage::Read() got invalid level=" + to_string(level) + " >= keyBits.size=" + to_string(keyBits.size()));
        return ZKR_DB_ERROR;
    }

    zkresult zkr;

    // Get control
    KeyValueStruct * page = (KeyValueStruct *)pageManager.getPageAddress(pageNumber);
    uint64_t index = keyBits[level];
    uint64_t control = page->key[index] >> 60;

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
            uint64_t rawDataPage = page->key[index] & U64Mask48;
            uint64_t rawDataOffset = (page->key[index] >> 48) & U64Mask12;
            string lengthBa;
            zkr = RawDataPage::Read(rawDataPage, rawDataOffset, 4, lengthBa);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("KeyValuePage::Read() failed calling RawDataPage::Read(4) result=" + zkresult2string(zkr) + " pageNumber=" + to_string(pageNumber) + " index=" + to_string(index) + " level=" + to_string(level) + " key=" + ba2string(key));
                return zkr;
            }
            if (lengthBa.size() != 4)
            {
                zklog.error("KeyValuePage::Read() called RawDataPage::Read(4) but got invalid size=" + to_string(lengthBa.size()) + " pageNumber=" + to_string(pageNumber) + " index=" + to_string(index) + " level=" + to_string(level) + " key=" + ba2string(key));
                return zkr;
            }

            uint64_t length;
            length = *(uint32_t *)lengthBa.c_str();
            string rawData;
            zkr = RawDataPage::Read(rawDataPage, rawDataOffset, length, rawData);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("KeyValuePage::Read() failed calling RawDataPage::Read() result=" + zkresult2string(zkr) + " pageNumber=" + to_string(pageNumber) + " index=" + to_string(index) + " level=" + to_string(level) + " key=" + ba2string(key));
                return zkr;
            }

            if (rawData.size() < (4 + key.size()))
            {
                zklog.error("KeyValuePage::Read() called RawDataPage::Read() and got invalid raw data size=" + to_string(rawData.size()) + " pageNumber=" + to_string(pageNumber) + " index=" + to_string(index) + " level=" + to_string(level) + " key=" + ba2string(key));
                return ZKR_DB_ERROR;
            }

            if (memcmp(rawData.c_str() + 4, key.c_str(), key.size()) != 0)
            {
                zklog.error("KeyValuePage::Read() called RawDataPage::Read() and got different keys pageNumber=" + to_string(pageNumber) + " index=" + to_string(index) + " level=" + to_string(level) + " key=" + ba2string(key));
                return ZKR_DB_KEY_NOT_FOUND;
            }
            
#ifdef LOG_KEY_VALUE_PAGE
            zklog.info("KeyValuePage::Read() read length=" + to_string(length) +
                " result=" + zkresult2string(zkr) +
                " pageNumber=" + to_string(pageNumber) +
                " index=" + to_string(index) +
                " level=" + to_string(level) +
                " key=" + ba2string(key) +
                " rawDataPage=" + to_string(rawDataPage) +
                " rawDataOffset=" + to_string(rawDataOffset));
#endif

            value = rawData.substr(4 + key.size());

            return ZKR_SUCCESS;
        }

        // Intermediate node
        case 2:
        {
            uint64_t nextKeyValuePage = page->key[index] & U64Mask48;
            return Read(nextKeyValuePage, key, keyBits, value, level+1);
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
    // Check input parameters
    if (key.size() == 0)
    {
        zklog.error("KeyValuePage::Read() found key.size=0 pageNumber=" + to_string(pageNumber));
        exitProcess();
    }

    // Clear value
    value.clear();

    // Get key bits
    vector<uint64_t> keyBits;
    splitKey9(key, keyBits);

    //zklog.info("KeyValuePage::Read() key=" + ba2string(key) + " keyBits=" + to_string(keyBits[0]) + ":" + to_string(keyBits[1]) + ":" + to_string(keyBits[2]));

    // Call Read with level=0
    return Read(pageNumber, key, keyBits, value, 0);
}

zkresult KeyValuePage::Write (uint64_t &pageNumber, const string &key, const vector<uint64_t> &keyBits, const string &value, const uint64_t level, const uint64_t headerPageNumber)
{
    // Check input parameters
    if (level >= keyBits.size())
    {
        zklog.error("KeyValuePage::write() got invalid level=" + to_string(level) + " >= keyBits.size=" + to_string(keyBits.size()));
        return ZKR_DB_ERROR;
    }

    // Get an editable page
    pageNumber = pageManager.editPage(pageNumber);

    zkresult zkr;
    KeyValueStruct * page = (KeyValueStruct *)pageManager.getPageAddress(pageNumber);
    uint64_t index = keyBits[level];

    uint64_t control = page->key[index] >> 60;

    switch (control)
    {
        // Empty slot: insert the new key here
        case 0:
        {
            //zklog.error("KeyValuePage::Write() found empty slot pageNumber=" + to_string(pageNumber) + " index=" + to_string(index) + " level=" + to_string(level) + " key=" + ba2string(key));
            uint64_t length = 4 + key.size() + value.size();
            if (length > U64Mask32)
            {
                zklog.error("KeyValuePage::Write() computed too big length=" + to_string(length) + " pageNumber=" + to_string(pageNumber) + " index=" + to_string(index) + " level=" + to_string(level) + " key=" + ba2string(key));
                return ZKR_DB_ERROR;
            }
            string lengthKeyValue;
            uint32_t length32 = length;
            lengthKeyValue.append((char *)&length32, 4);
            lengthKeyValue += key;
            lengthKeyValue += value;

            // Get header page data
            HeaderStruct * headerPage = (HeaderStruct *)pageManager.getPageAddress(headerPageNumber);
            uint64_t rawDataPage = headerPage->rawDataPage;
            uint64_t rawDataOffset = RawDataPage::GetOffset(headerPage->rawDataPage);

            // Write to raw data page list
            zkr = RawDataPage::Write(headerPage->rawDataPage, lengthKeyValue);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("KeyValuePage::Write() failed calling RawDataPage::Write() result=" + zkresult2string(zkr) + " pageNumber=" + to_string(pageNumber) + " index=" + to_string(index) + " level=" + to_string(level) + " key=" + ba2string(key));
                return zkr;
            }

#ifdef LOG_KEY_VALUE_PAGE
            zklog.info("KeyValuePage::Write() wrote length=" + to_string(length) +
                " totalLength=" + to_string(lengthKeyValue.size()) +
                " result=" + zkresult2string(zkr) +
                " pageNumber=" + to_string(pageNumber) +
                " index=" + to_string(index) +
                " level=" + to_string(level) +
                " key=" + ba2string(key) +
                " rawDataPage=" + to_string(rawDataPage) +
                " rawDataOffset=" + to_string(rawDataOffset) +
                " leaving headerPage->rawDataPage=" + to_string(headerPage->rawDataPage) +
                " headerPage->rawDataOffset=" + to_string(RawDataPage::GetOffset(headerPage->rawDataPage)));            
#endif

            // Update this entry as a leaf node (control = 1)
            page->key[index] = (uint64_t(1)<<60) | ((rawDataOffset & U64Mask12) << 48) | (rawDataPage & U64Mask48);

            return ZKR_SUCCESS;
        }

        // Leaf node
        case 1:
        {
            // Read the key from the raw data page
            uint64_t rawPageNumber = page->key[index] & U64Mask48;
            uint64_t rawPageOffset = (page->key[index] >> 48) & U64Mask12;
            string lengthString;

            // Get the length
            zkr = RawDataPage::Read(rawPageNumber, rawPageOffset, 4, lengthString);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("KeyValuePage::Write() failed calling RawDataPage::Read(4) result=" + zkresult2string(zkr) + " pageNumber=" + to_string(pageNumber) + " index=" + to_string(index) + " level=" + to_string(level) + " key=" + ba2string(key));
                return zkr;
            }
            uint64_t length = *(uint32_t *)(lengthString.c_str());
            if (length < 4 + key.size())
            {
                zklog.error("KeyValuePage::Write() found invalid length=" + to_string(length) + " pageNumber=" + to_string(pageNumber) + " index=" + to_string(index) + " level=" + to_string(level) + " key=" + ba2string(key));
                exitProcess();
            }

            // Get the existing key
            string existingLengthAndKey;
            zkr = RawDataPage::Read(rawPageNumber, rawPageOffset, 4 + key.size(), existingLengthAndKey);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("KeyValuePage::Write() failed calling RawDataPage::Read() result=" + zkresult2string(zkr) + " length=" + to_string(length) + " pageNumber=" + to_string(pageNumber) + " index=" + to_string(index) + " level=" + to_string(level) + " key=" + ba2string(key));
                return zkr;
            }

            // If both keys are identical, values should be identical, too
            if (existingLengthAndKey.substr(4) == key)
            {
                // Compare total length
                if (length != 4 + key.size() + value.size())
                {
                    zklog.error("KeyValuePage::Write() found not matching length=" + to_string(length) + " != key.size+value.size=" + to_string(key.size() + value.size()) + " pageNumber=" + to_string(pageNumber) + " index=" + to_string(index) + " level=" + to_string(level) + " key=" + ba2string(key));
                    exitProcess();
                }

                // Get the existing key + value
                string existingLengthKeyAndValue;
                zkr = RawDataPage::Read(rawPageNumber, rawPageOffset, length, existingLengthKeyAndValue);
                if (zkr != ZKR_SUCCESS)
                {
                    zklog.error("KeyValuePage::Write() failed calling RawDataPage::Read() result=" + zkresult2string(zkr) + " length=" + to_string(length) + " pageNumber=" + to_string(pageNumber) + " index=" + to_string(index) + " level=" + to_string(level) + " key=" + ba2string(key));
                    return zkr;
                }

                // Compare programs
                if (existingLengthKeyAndValue.size() != 4 + key.size() + value.size())
                {
                    zklog.error("KeyValuePage::Write() found not matching existingLengthKeyAndValue.size()=" + to_string(existingLengthKeyAndValue.size()) + " != key.size+value.size=" + to_string(key.size() + value.size()) + " pageNumber=" + to_string(pageNumber) + " index=" + to_string(index) + " level=" + to_string(level) + " key=" + ba2string(key));
                    exitProcess();
                }
                if (value.compare(existingLengthKeyAndValue.substr(4 + key.size())) != 0)
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
            string existingKey = existingLengthAndKey.substr(4);
            vector<uint64_t> existingKeyBits;
            splitKey9(existingKey, existingKeyBits);
            if (existingKeyBits.size() < (level+2))
            {
                zklog.error("KeyValuePage::Write() found not matching value of existingKeyBits.size=" + to_string(existingKeyBits.size()) + " < level+2=" + to_string(level + 2) + " pageNumber=" + to_string(pageNumber) + " index=" + to_string(index) + " level=" + to_string(level) + " key=" + ba2string(key));
                exitProcess();
            }
            uint64_t newIndex = existingKeyBits[level+1];
            newPage->key[newIndex] = page->key[index];
            
#ifdef LOG_KEY_VALUE_PAGE
            zklog.info("KeyValuePage::Write() moved existing key=" + ba2string(existingLengthAndKey.substr(4)) + " to new page=" + to_string(newPageNumber) + " at index=" + to_string(newIndex));
#endif

            // Write the new key in the newly created page at the next level
            zkr = Write(newPageNumber, key, keyBits, value, level+1, headerPageNumber);
            if (zkr != ZKR_SUCCESS)
            {
                return zkr;
            }

            // Set this page entry as a intermeiate node
            page->key[index] = newPageNumber | (uint64_t(2) << 60);

            return ZKR_SUCCESS;
        }

        // Intermediate node
        case 2:
        {
            // Call Write with the next level
            uint64_t nextKeyValuePage = page->key[index] & U64Mask48;
            zkr = Write(nextKeyValuePage, key, keyBits, value, level+1, headerPageNumber);
            if (zkr != ZKR_SUCCESS)
            {
                return zkr;
            }
            page->key[index] = nextKeyValuePage + (uint64_t(2) << 60);

            return ZKR_SUCCESS;
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

zkresult KeyValuePage::Write (uint64_t &pageNumber, const string &key, const string &value, const uint64_t headerPageNumber)
{
    // Check input parameters
    if (key.size() == 0)
    {
        zklog.error("KeyValuePage::Write() found key.size=0 pageNumber=" + to_string(pageNumber));
        exitProcess();
    }

    // Get key bits
    vector<uint64_t> keyBits;
    splitKey9(key, keyBits);


    //zklog.info("KeyValuePage::Write() key=" + ba2string(key) + " keyBits=" + to_string(keyBits[0]) + ":" + to_string(keyBits[1]) + ":" + to_string(keyBits[2]));

    // Call Write with level=0
    return Write(pageNumber, key, keyBits, value, 0, headerPageNumber);
}

void KeyValuePage::Print (const uint64_t pageNumber, bool details, const string& prefix, const uint64_t keySize)
{
    zklog.info(prefix + "KeyValuePage::Print() pageNumber=" + to_string(pageNumber));

    zkresult zkr;

    vector<uint64_t> nextKeyValuePages;

    // For each entry of the page
    KeyValueStruct * page = (KeyValueStruct *)pageManager.getPageAddress(pageNumber);
    for (uint64_t index = 0; index < 512; index++)
    {
        uint64_t control = page->key[index] >> 60;

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
                uint64_t rawDataPage = page->key[index] & U64Mask48;
                uint64_t rawDataOffset = (page->key[index] >> 48) & U64Mask12;

                if (details)
                {
                    string lengthBa;
                    zkr = RawDataPage::Read(rawDataPage, rawDataOffset, 4, lengthBa);
                    if (zkr != ZKR_SUCCESS)
                    {
                        zklog.error("KeyValuePage::Print() failed calling RawDataPage::Read(4) result=" + zkresult2string(zkr) + " pageNumber=" + to_string(pageNumber) + " index=" + to_string(index));
                        exitProcess();
                    }
                    if (lengthBa.size() != 4)
                    {
                        zklog.error("KeyValuePage::Print() called RawDataPage::Read(4) but got invalid size=" + to_string(lengthBa.size()) + " pageNumber=" + to_string(pageNumber) + " index=" + to_string(index));
                        exitProcess();
                    }

                    uint64_t length;
                    length = *(uint32_t *)lengthBa.c_str();
                    string rawData;
                    zkr = RawDataPage::Read(rawDataPage, rawDataOffset, length, rawData);
                    if (zkr != ZKR_SUCCESS)
                    {
                        zklog.error("KeyValuePage::Print() failed calling RawDataPage::Read() result=" + zkresult2string(zkr) + " pageNumber=" + to_string(pageNumber) + " index=" + to_string(index));
                        exitProcess();
                    }

                    if (rawData.size() < (4 + keySize))
                    {
                        zklog.error("KeyValuePage::Print() called RawDataPage::Read() and got invalid raw data size=" + to_string(rawData.size()) + " pageNumber=" + to_string(pageNumber) + " index=" + to_string(index));
                        exitProcess();
                    }

                    string key = rawData.substr(4, keySize);
                    string value = rawData.substr(4 + keySize);

                    zklog.info(prefix + "i=" + to_string(index) +
                        " rawDataPage=" + to_string(rawDataPage) + " rawDataOffset=" + to_string(rawDataOffset) +
                        " key=" + ba2string(key) + " value=" + ba2string(value));
                }
                else
                {
                    zklog.info(prefix + "i=" + to_string(index) + " rawDataPage=" + to_string(rawDataPage) + " rawDataOffset=" + to_string(rawDataOffset));
                }
                continue;
            }

            // Intermediate node
            case 2:
            {
                uint64_t nextKeyValuePage = page->key[index] & U64Mask48;
                nextKeyValuePages.emplace_back(nextKeyValuePage);
                if (details)
                {
                    zklog.info(prefix + "i=" + to_string(index) + " nextKeyValuePage=" + to_string(nextKeyValuePage));
                }
                continue;
            }

            // Default
            default:
            {
                zklog.error("KeyValuePage::Print() found invalid control=" + to_string(control) + " pageNumber=" + to_string(pageNumber) + " i=" + to_string(index));
                exitProcess();
            }

        }      
    }

    for (uint64_t i=0; i<nextKeyValuePages.size(); i++)
    {
        Print(nextKeyValuePages[i], details, prefix + " ", keySize);
    }
}