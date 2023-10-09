#include "key_value_history_page.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"
#include "zkassert.hpp"
#include "scalar.hpp"
#include "page_manager.hpp"
#include "hash_page.hpp"
#include "key_utils.hpp"
#include "zkglobals.hpp"
#include "raw_data_page.hpp"
#include "header_page.hpp"

zkresult KeyValueHistoryPage::InitEmptyPage (const uint64_t pageNumber)
{
    KeyValueHistoryStruct * page = (KeyValueHistoryStruct *)pageManager.getPageAddress(pageNumber);
    memset((void *)page, 0, 4096);

    // Create the 2 associated hash pages
    uint64_t hashPage1 = pageManager.getFreePage();
    HashPage::InitEmptyPage(hashPage1);
    uint64_t hashPage2 = pageManager.getFreePage();
    HashPage::InitEmptyPage(hashPage2);
    page->hashPage1AndHistoryCounter = hashPage1;
    page->hashPage2AndPadding = hashPage2;
    
    return ZKR_SUCCESS;
}

zkresult KeyValueHistoryPage::Read (const uint64_t pageNumber, const string &key, const string &keyBits, const uint64_t version, mpz_class &value, const uint64_t level)
{
    zkassert(key.size() == 32);
    zkassert(keyBits.size() == 42);
    zkassert(level < 42);

    zkresult zkr;

    // Get the data from this page
    KeyValueHistoryStruct * page = (KeyValueHistoryStruct *)pageManager.getPageAddress(pageNumber);
    uint64_t index = keyBits[level];
    uint64_t control = page->keyValueEntry[index][1] >> 60;

    // Check control
    switch (control)
    {
        // Empty slot
        case 0:
        {
            value = 0;
            return ZKR_SUCCESS;            
        }
        // Leaf node
        case 1:
        {
            uint64_t *keyValueEntry = page->keyValueEntry[index];

            while (true)
            {
                // Read the latest version of this key in this page
                uint64_t foundVersion = keyValueEntry[1] & 0xFFFFFF;

                // If it is equal or lower, then we found it
                if (version >= foundVersion)
                {
                    uint64_t rawDataPage = keyValueEntry[0] & 0xFFFFFF;
                    uint64_t rawDataOffset = keyValueEntry[0] >> 48;
                    string keyValue;
                    zkr = RawDataPage::Read(rawDataPage, rawDataOffset, 64, keyValue);
                    if (zkr != ZKR_SUCCESS)
                    {
                        zklog.error("KeyValueHistoryPage::Read() failed calling RawDataPage.Read result=" + zkresult2string(zkr) + " rawDataPage=" + to_string(rawDataPage) + " rawDataOffset=" + to_string(rawDataOffset) + " key=" + ba2string(key) + " version=" + to_string(version) + " level=" + to_string(level) + " index=" + to_string(index));
                        return zkr;
                    }
                    if (keyValue.size() != 64)
                    {
                        zklog.error("KeyValueHistoryPage::Read() called RawDataPage.Read and got invalid length=" + to_string(keyValue.size()) + " rawDataPage=" + to_string(rawDataPage) + " rawDataOffset=" + to_string(rawDataOffset) + " key=" + ba2string(key) + " version=" + to_string(version) + " level=" + to_string(level) + " index=" + to_string(index));
                        return zkr;
                    }

                    // If this is a different key, then the key does not exist, i.e. the value is 0
                    if (memcmp(key.c_str(), keyValue.c_str(), 32) != 0)
                    {
                        value = 0;
                        return ZKR_SUCCESS;
                    }

                    // Convert the value
                    ba2scalar((uint8_t *)keyValue.c_str() + 32, 32, value);

                    return ZKR_SUCCESS;
                }

                // Search for 
                uint64_t previousVersionOffset = (page->keyValueEntry[index][1] >> 48) & 0xFFF;

                // If there is no previous version for this key, then this is a zero
                if (previousVersionOffset == 0)
                {
                    value = 0;
                    return ZKR_SUCCESS;
                }

                // If not zero, then check the range of the previous version
                if ((previousVersionOffset < minVersionOffset) || (previousVersionOffset > maxVersionOffset) || ((previousVersionOffset & 0xF) != 0))
                {
                    zklog.error("KeyValueHistoryPage::Read() found invalid previousVersionOffset=" + to_string(previousVersionOffset));
                    return ZKR_DB_ERROR;
                }

                // Get the previous version entry
                uint64_t previousVersionIndex = (previousVersionOffset%16) - 65;
                keyValueEntry = page->historyEntry[previousVersionIndex];
            }
        }
        // Intermediate node
        case 2:
        {
            return Read(pageNumber, key, keyBits, version, value, level + 1);
        }
        default:
        {
            zklog.error("KeyValuePage::Write() found invalid control=" + to_string(control) + " pageNumber=" + to_string(pageNumber));
            return ZKR_DB_ERROR;
        }
    }

    // Not found
    return ZKR_DB_KEY_NOT_FOUND;
}

zkresult KeyValueHistoryPage::Read (const uint64_t pageNumber, const string &key, const uint64_t version, mpz_class &value)
{
    zkassert(key.size() == 32);
    zkassert((version & 0xFFFFFF) == version);

    // Get 256 key bits in SMT order, in sets of 6 bits
    Goldilocks::Element keyFea[4];
    string2fea(fr, key, keyFea);
    uint8_t keyBitsArray[43];
    splitKey6(fr, keyFea, keyBitsArray);
    string keyBits;
    keyBits.append((char *)keyBitsArray, 43);

    return Read(pageNumber, key, keyBits, version, value, 0);
}

zkresult KeyValueHistoryPage::Write (uint64_t &pageNumber, const string &key, const string &keyBits, const uint64_t version, const mpz_class &value, const uint64_t level, const uint64_t headerPageNumber)
{
    zkassert(key.size() == 32);
    zkassert(keyBits.size() == 42);
    zkassert(level < 42);

    zkresult zkr;

    // Get the data from this page
    KeyValueHistoryStruct * page = (KeyValueHistoryStruct *)pageManager.getPageAddress(pageNumber);
    uint64_t index = keyBits[level];
    uint64_t control = page->keyValueEntry[index][1] >> 60;

    // Check control
    switch (control)
    {
        // Empty slot
        case 0:
        {
            // Get an editable version of this page
            pageNumber = pageManager.editPage(pageNumber);
            KeyValueHistoryStruct * page = (KeyValueHistoryStruct *)pageManager.getPageAddress(pageNumber);

            // Get an editable version of the header page
            //headerPageNumber = pageManager.editPage(headerPageNumber);
            HeaderStruct *headerPage = (HeaderStruct *)pageManager.getPageAddress(headerPageNumber);

            uint64_t rawDataPage = headerPage->rawDataPage;
            uint64_t rawDataOffset = RawDataPage::GetOffset(rawDataPage);

            string keyAndValue = key + scalar2ba(value);
            zkr = RawDataPage::Write(headerPage->rawDataPage, keyAndValue);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("KeyValueHistoryPage::Write() failed calling RawDataPage.Write result=" + zkresult2string(zkr) + " rawDataPage=" + to_string(rawDataPage) + " rawDataOffset=" + to_string(rawDataOffset) + " key=" + ba2string(key) + " version=" + to_string(version) + " level=" + to_string(level) + " index=" + to_string(index));
                return zkr;
            }

            page->keyValueEntry[index][0] = (rawDataOffset << 48) + rawDataPage;
            page->keyValueEntry[index][1] = (uint64_t(1) << 60) + (version);

            return ZKR_SUCCESS;            
        }
        // Leaf node
        case 1:
        {
            // Read the key stored in raw data
            uint64_t rawDataPage = page->keyValueEntry[index][0] & 0xFFFFFFFFFFFF;
            uint64_t rawDataOffset = page->keyValueEntry[index][0] >> 48;
            string keyAndValue;
            zkr = RawDataPage::Read(rawDataPage, rawDataOffset, 64, keyAndValue);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("KeyValueHistoryPage::Write() failed calling RawDataPage.Read result=" + zkresult2string(zkr) + " rawDataPage=" + to_string(rawDataPage) + " rawDataOffset=" + to_string(rawDataOffset) + " key=" + ba2string(key) + " version=" + to_string(version) + " level=" + to_string(level) + " index=" + to_string(index));
                return zkr;
            }
            if (keyAndValue.size() != 64)
            {
                zklog.error("KeyValueHistoryPage::Write() called RawDataPage.Read but got invalid keyAndValue.size=" + zkresult2string(keyAndValue.size()) + " rawDataPage=" + to_string(rawDataPage) + " rawDataOffset=" + to_string(rawDataOffset) + " key=" + ba2string(key) + " version=" + to_string(version) + " level=" + to_string(level) + " index=" + to_string(index));
                return zkr;
            }

            // If the key is the same
            if (memcmp(key.c_str(), keyAndValue.c_str(), 32) == 0)
            {
                // If both the key and value are the same, there's nothing to do
                string valueBa = scalar2ba(value);
                if (memcmp(valueBa.c_str(), keyAndValue.c_str() + 32, 32) == 0)
                {
                    return ZKR_SUCCESS;
                }

                // If the value is different, then create a new value entry in raw data

                // Get an editable version of the header page
                //headerPageNumber = pageManager.editPage(headerPageNumber);
                HeaderStruct *headerPage = (HeaderStruct *)pageManager.getPageAddress(headerPageNumber);

                // Get the current rawDataPge and offset
                uint64_t rawDataPage = headerPage->rawDataPage;
                uint64_t rawDataOffset = RawDataPage::GetOffset(headerPage->rawDataPage);

                string keyAndValue = key + valueBa;
                zkr = RawDataPage::Write(rawDataPage, keyAndValue);
                if (zkr != ZKR_SUCCESS)
                {
                    zklog.error("KeyValueHistoryPage::Write() failed calling RawDataPage.Write result=" + zkresult2string(zkr) + " rawDataPage=" + to_string(rawDataPage) + " rawDataOffset=" + to_string(rawDataOffset) + " key=" + ba2string(key) + " version=" + to_string(version) + " level=" + to_string(level) + " index=" + to_string(index));
                    return zkr;
                }

                page->keyValueEntry[index][0] = (rawDataOffset << 48) + (rawDataPage & 0xFFFFFFFFFFFF);

                return ZKR_SUCCESS;
            }

            // If the key is different, move the key to a new KeyValuePage, and write the new key into the new page

            
        }
        // Intermediate node
        case 2:
        {
            return Write(pageNumber, key, keyBits, version, value, level + 1, headerPageNumber);
        }
        default:
        {
            zklog.error("KeyValuePage::Write() found invalid control=" + to_string(control) + " pageNumber=" + to_string(pageNumber));
            return ZKR_DB_ERROR;
        }
    }
}

zkresult KeyValueHistoryPage::Write (uint64_t &pageNumber, const string &key, const uint64_t version, const mpz_class &value, const uint64_t headerPageNumber)
{
    zkassert(key.size() == 32);
    zkassert((version & 0xFFFFFF) == version);

    // Get 256 key bits in SMT order, in sets of 6 bits
    Goldilocks::Element keyFea[4];
    string2fea(fr, key, keyFea);
    uint8_t keyBitsArray[43];
    splitKey6(fr, keyFea, keyBitsArray);
    string keyBits;
    keyBits.append((char *)keyBitsArray, 43);

    // Start searching with level 0
    return Write(pageNumber, key, keyBits, version, value, 0, headerPageNumber);
}

void KeyValueHistoryPage::calculateLeafHash (const Goldilocks::Element (&key)[4], const uint64_t level, const mpz_class &value, Goldilocks::Element (&hash)[4], vector<HashValueGL> *hashValues)
{
    // Prepare input = [value8, 0000]
    Goldilocks::Element input[12];
    scalar2fea(fr, value, input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7]);
    input[8] = fr.zero();
    input[9] = fr.zero();
    input[10] = fr.zero();
    input[11] = fr.zero();

    // Calculate the value hash
    Goldilocks::Element valueHash[4];
    poseidon.hash(valueHash, input);

    // Return the hash-value pair, if requested
    if (hashValues != NULL)
    {
        HashValueGL hashValue;
        for (uint64_t i=0; i<4; i++) hashValue.hash[i] = valueHash[i];
        for (uint64_t i=0; i<12; i++) hashValue.value[i] = input[i];
        hashValues->emplace_back(hashValue);
    }

    // Calculate the remaining key
    Goldilocks::Element rkey[4];
    removeKeyBits(fr, key, level, rkey);

    // Prepare input = [rkey, valueHash, 1000]
    input[0] = rkey[0];
    input[1] = rkey[1];
    input[2] = rkey[2];
    input[3] = rkey[3];
    input[4] = valueHash[0];
    input[5] = valueHash[1];
    input[6] = valueHash[2];
    input[7] = valueHash[3];
    input[8] = fr.one();
    input[9] = fr.zero();
    input[10] = fr.zero();
    input[11] = fr.zero();

    // Calculate the leaf node hash
    poseidon.hash(hash, input);

    // Return the hash-value pair, if requested
    if (hashValues != NULL)
    {
        HashValueGL hashValue;
        for (uint64_t i=0; i<4; i++) hashValue.hash[i] = hash[i];
        for (uint64_t i=0; i<12; i++) hashValue.value[i] = input[i];
        hashValues->emplace_back(hashValue);
    }
}

void KeyValueHistoryPage::calculateIntermediateHash (const Goldilocks::Element (&leftHash)[4], const Goldilocks::Element (&rightHash)[4], Goldilocks::Element (&hash)[4], vector<HashValueGL> *hashValues)
{
    // Prepare input = [leftHash, rightHash, 0000]
    Goldilocks::Element input[12];
    input[0] = leftHash[0];
    input[1] = leftHash[1];
    input[2] = leftHash[2];
    input[3] = leftHash[3];
    input[4] = rightHash[0];
    input[5] = rightHash[1];
    input[6] = rightHash[2];
    input[7] = rightHash[3];
    input[8] = fr.zero();
    input[9] = fr.zero();
    input[10] = fr.zero();
    input[11] = fr.zero();

    // Calculate the poseidon hash
    poseidon.hash(hash, input);

    // Return the hash-value pair, if requested
    if (hashValues != NULL)
    {
        HashValueGL hashValue;
        for (uint64_t i=0; i<4; i++) hashValue.hash[i] = hash[i];
        for (uint64_t i=0; i<12; i++) hashValue.value[i] = input[i];
        hashValues->emplace_back(hashValue);
    }
}

void KeyValueHistoryPage::Print (const uint64_t pageNumber, bool details)
{
    zklog.info("KeyValuePage::Print() pageNumber=" + to_string(pageNumber));

    // Get the data from this page
    KeyValueHistoryStruct * page = (KeyValueHistoryStruct *)pageManager.getPageAddress(pageNumber);
    uint64_t hashPage1Number = page->hashPage1AndHistoryCounter & 0xFFFFFF;
    uint64_t historyCounter = page->hashPage1AndHistoryCounter >> 48;
    uint64_t hashPage2Number = page->hashPage2AndPadding & 0xFFFFFF;
    zklog.info("  hashPage1Number=" + to_string(hashPage1Number));
    HashPage::Print(hashPage1Number, details);
    zklog.info("  hashPage2Number=" + to_string(hashPage2Number));
    HashPage::Print(hashPage2Number, details);
    zklog.info("  historyCounter=" + to_string(historyCounter));

    for (uint64_t i=0; i<64; i++)
    {
        uint64_t control = (page->keyValueEntry[i][1]) >> 60;

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

            }
            // Intermediate node
            case 2:
            {

            }
            default:
            {
                zklog.error("KeyValuePage::Print() found invalid control=" + to_string(control) + " pageNumber=" + to_string(pageNumber));
                exitProcess();
            }
        }
    }
}