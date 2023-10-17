#include "key_value_history_page.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"
#include "zkassert.hpp"
#include "scalar.hpp"
#include "page_manager.hpp"
#include "key_utils.hpp"
#include "zkglobals.hpp"
#include "raw_data_page.hpp"
#include "header_page.hpp"
#include "constants.hpp"
#include "tree_chunk.hpp"

zkresult KeyValueHistoryPage::InitEmptyPage (const uint64_t pageNumber)
{
    KeyValueHistoryStruct * page = (KeyValueHistoryStruct *)pageManager.getPageAddress(pageNumber);
    memset((void *)page, 0, 4096);
    page->historyOffset = minHistoryOffset;
    return ZKR_SUCCESS;
}

zkresult KeyValueHistoryPage::Read (const uint64_t pageNumber, const string &key, const string &keyBits, const uint64_t version, mpz_class &value, const uint64_t level, uint64_t &keyLevel)
{
    zkassert(key.size() == 32);
    zkassert(keyBits.size() == 43);
    zkassert(level < 43);

    zkresult zkr;

    // Get the data from this page
    KeyValueHistoryStruct * page = (KeyValueHistoryStruct *)pageManager.getPageAddress(pageNumber);
    uint64_t index = keyBits[level];
    uint64_t control = page->keyValueEntry[index][0] >> 60;

    // Check control
    switch (control)
    {
        // Empty slot
        case 0:
        {
            value = 0;
            keyLevel = (level + 1) * 6;
            return ZKR_SUCCESS;            
        }
        // Leaf node
        case 1:
        {
            uint64_t *keyValueEntry = page->keyValueEntry[index];

            while (true)
            {
                // Read the latest version of this key in this page
                uint64_t foundVersion = keyValueEntry[0] & U64Mask48;

                // If it is equal or lower, then we found the slot, although it could be occupied by a different key, so we need to check
                if (version >= foundVersion)
                {
                    uint64_t rawDataPage = keyValueEntry[1] & U64Mask48;
                    uint64_t rawDataOffset = keyValueEntry[1] >> 48;
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
                        //zklog.info("KeyValueHistoryPage::Read() found existing key=" + ba2string(keyValue.substr(0, 32)) + " != key=" + ba2string(key));
                        value = 0;

                        // Get the key level
                        keyLevel = (level + 1) * 6;

                        return ZKR_SUCCESS;
                    }

                    // Convert the value
                    ba2scalar((uint8_t *)keyValue.c_str() + 32, 32, value);

                    // Get the key level
                    keyLevel = (level + 1) * 6;

                    return ZKR_SUCCESS;
                }

                // Search for 
                uint64_t previousVersionOffset = (page->keyValueEntry[index][1] >> 48) & U64Mask12;

                // If there is no previous version for this key, then this is a zero
                if (previousVersionOffset == 0)
                {
                    value = 0;

                    // Get the key level
                    keyLevel = (level + 1) * 6;

                    return ZKR_SUCCESS;
                }

                // If not zero, then check the range of the previous version
                if ( (previousVersionOffset < minHistoryOffset) ||
                     (previousVersionOffset > maxHistoryOffset) ||
                     ((previousVersionOffset & U64Mask4) != 0) )
                {
                    zklog.error("KeyValueHistoryPage::Read() found invalid previousVersionOffset=" + to_string(previousVersionOffset));
                    return ZKR_DB_ERROR;
                }

                // Get the previous version entry
                keyValueEntry = (uint64_t *)((uint8_t *)page + previousVersionOffset);
            }
        }
        // Intermediate node
        case 2:
        {
            uint64_t nextPageNumber = page->keyValueEntry[index][1] & U64Mask48;
            return Read(nextPageNumber, key, keyBits, version, value, level + 1, keyLevel);
        }
        default:
        {
            zklog.error("KeyValuePage::Read() found invalid control=" + to_string(control) + " pageNumber=" + to_string(pageNumber));
            return ZKR_DB_ERROR;
        }
    }

    // Not found
    return ZKR_DB_KEY_NOT_FOUND;
}

zkresult KeyValueHistoryPage::Read (const uint64_t pageNumber, const string &key, const uint64_t version, mpz_class &value, uint64_t &keyLevel)
{
    zkassert(key.size() == 32);
    zkassert((version & U64Mask48) == version);

    // Get 256 key bits in SMT order, in sets of 6 bits
    Goldilocks::Element keyFea[4];
    string2fea(fr, ba2string(key), keyFea);
    uint8_t keyBitsArray[43];
    splitKey6(fr, keyFea, keyBitsArray);
    string keyBits;
    keyBits.append((char *)keyBitsArray, 43);

    return Read(pageNumber, key, keyBits, version, value, 0, keyLevel);
}

zkresult KeyValueHistoryPage::ReadLevel (const uint64_t pageNumber, const string &key, const string &keyBits, const uint64_t level, uint64_t &keyLevel)
{
    zkassert(key.size() == 32);
    zkassert(keyBits.size() == 43);
    zkassert(level < 43);

    zkresult zkr;

    // Get the data from this page
    KeyValueHistoryStruct * page = (KeyValueHistoryStruct *)pageManager.getPageAddress(pageNumber);
    uint64_t index = keyBits[level];
    uint64_t control = page->keyValueEntry[index][0] >> 60;

    // Check control
    switch (control)
    {
        // Empty slot
        case 0:
        {
            keyLevel = (level + 1) * 6;
            return ZKR_SUCCESS;            
        }
        // Leaf node
        case 1:
        {
            uint64_t rawDataPage = page->keyValueEntry[index][1] & U64Mask48;
            uint64_t rawDataOffset = page->keyValueEntry[index][1] >> 48;
            string keyValue;
            zkr = RawDataPage::Read(rawDataPage, rawDataOffset, 64, keyValue);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("KeyValueHistoryPage::ReadLevel() failed calling RawDataPage.Read result=" + zkresult2string(zkr) + " rawDataPage=" + to_string(rawDataPage) + " rawDataOffset=" + to_string(rawDataOffset) + " key=" + ba2string(key) + " level=" + to_string(level) + " index=" + to_string(index));
                return zkr;
            }
            if (keyValue.size() != 64)
            {
                zklog.error("KeyValueHistoryPage::ReadLevel() called RawDataPage.Read and got invalid length=" + to_string(keyValue.size()) + " rawDataPage=" + to_string(rawDataPage) + " rawDataOffset=" + to_string(rawDataOffset) + " key=" + ba2string(key) + " level=" + to_string(level) + " index=" + to_string(index));
                return zkr;
            }

            // If this is the same key
            if (memcmp(key.c_str(), keyValue.c_str(), 32) != 0)
            {
                // Get the key level
                keyLevel = (level + 1) * 6;

                return ZKR_SUCCESS;
            }

            // If keys are different, we need to know how different they are
            Goldilocks::Element keyFea[4];
            string2fea(fr, ba2string(keyValue.substr(0, 32)), keyFea);
            uint8_t foundKeyBitsArray[43];
            splitKey6(fr, keyFea, foundKeyBitsArray);
            string foundKeyBits;
            foundKeyBits.append((char *)foundKeyBitsArray, 43);


            // Find the first 6-bit set that is different
            uint64_t i=0;
            for (; i<43; i++)
            {
                if (keyBits[i] != foundKeyBits[i])
                {
                    break;
                }
            }

            // Set the level
            keyLevel = (i + 1) * 6;

            return ZKR_SUCCESS;
        }
        // Intermediate node
        case 2:
        {
            uint64_t nextPageNumber = page->keyValueEntry[index][1] & U64Mask48;
            return ReadLevel(nextPageNumber, key, keyBits, level + 1, keyLevel);
        }
        default:
        {
            zklog.error("KeyValuePage::ReadLevel() found invalid control=" + to_string(control) + " pageNumber=" + to_string(pageNumber));
            return ZKR_DB_ERROR;
        }
    }

    // Not found
    return ZKR_DB_KEY_NOT_FOUND;
}

zkresult KeyValueHistoryPage::ReadLevel (const uint64_t pageNumber, const string &key, uint64_t &keyLevel)
{
    zkassert(key.size() == 32);

    // Get 256 key bits in SMT order, in sets of 6 bits
    Goldilocks::Element keyFea[4];
    string2fea(fr, ba2string(key), keyFea);
    uint8_t keyBitsArray[43];
    splitKey6(fr, keyFea, keyBitsArray);
    string keyBits;
    keyBits.append((char *)keyBitsArray, 43);

    return ReadLevel(pageNumber, key, keyBits, 0, keyLevel);
}

zkresult KeyValueHistoryPage::Write (uint64_t &pageNumber, const string &key, const string &keyBits, const uint64_t version, const mpz_class &value, const uint64_t level, uint64_t &headerPageNumber)
{
    zkassert(key.size() == 32);
    zkassert(keyBits.size() == 43);
    zkassert(level < 43);

    zkresult zkr;

    // Get the data from this page
    KeyValueHistoryStruct * page = (KeyValueHistoryStruct *)pageManager.getPageAddress(pageNumber);
    uint64_t index = keyBits[level];
    uint64_t control = page->keyValueEntry[index][0] >> 60;

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

            uint64_t insertionRawDataPage = headerPage->rawDataPage;
            uint64_t insertionRawDataOffset = RawDataPage::GetOffset(insertionRawDataPage);

            string keyAndValue = key + scalar2ba32(value);  // TODO: Check that value size=32
            zkr = RawDataPage::Write(headerPage->rawDataPage, keyAndValue);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("KeyValueHistoryPage::Write() failed calling RawDataPage.Write result=" + zkresult2string(zkr) + " insertionRawDataPage=" + to_string(insertionRawDataPage) + " insertionRawDataOffset=" + to_string(insertionRawDataOffset) + " key=" + ba2string(key) + " version=" + to_string(version) + " level=" + to_string(level) + " index=" + to_string(index));
                return zkr;
            }

            // Update entry
            page->keyValueEntry[index][0] = (uint64_t(1) << 60) | version;
            page->keyValueEntry[index][1] = (insertionRawDataOffset << 48) + insertionRawDataPage;
            page->keyValueEntry[index][2] = 0; // Mask as no hash has been calculated

            return ZKR_SUCCESS;            
        }
        // Leaf node
        case 1:
        {
            // Read the key and value stored in raw data
            uint64_t rawDataPage = page->keyValueEntry[index][1] & U64Mask48;
            uint64_t rawDataOffset = page->keyValueEntry[index][1] >> 48;
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
                // If both the key and value are the same, there's nothing to do, even if versions are not the same (we leave the oldest one)
                string valueBa = scalar2ba32(value);
                if (memcmp(valueBa.c_str(), keyAndValue.c_str() + 32, 32) == 0)
                {
                    return ZKR_SUCCESS;
                }

                // If the value is different, then check versions and move this entry to the history array
                uint64_t currentVersion = page->keyValueEntry[index][0] & U64Mask48;
                if (version <= currentVersion)
                {
                    zklog.error("KeyValueHistoryPage::Write() version discrepancy currentVersion=" + to_string(currentVersion) + " key=" + ba2string(key) + " version=" + to_string(version) + " level=" + to_string(level) + " index=" + to_string(index));
                    return ZKR_DB_ERROR;
                }

                // If we run out of history space in this page, create a new one
                if (page->historyOffset == maxHistoryOffset)
                {
                    // Create a new page
                    uint64_t newPageNumber = pageManager.getFreePage();
                    KeyValueHistoryStruct *newPage = (KeyValueHistoryStruct *)pageManager.getPageAddress(newPageNumber);

                    // Copy data from the current page to the new one
                    memcpy(newPage, page, minHistoryOffset);
                    newPage->historyOffset = minHistoryOffset;
                    newPage->previousPage = pageNumber;

                    // Replace the current page by the new one
                    pageNumber = newPageNumber;
                    page = newPage;
                }

                // Copy the current entry to the next history one
                uint64_t previousVersionOffset = page->historyOffset;
                memcpy(page + page->historyOffset, &page->keyValueEntry[index], entrySize);
                page->historyOffset += entrySize;

                // Get an editable version of the header page
                headerPageNumber = pageManager.editPage(headerPageNumber);
                HeaderStruct *headerPage = (HeaderStruct *)pageManager.getPageAddress(headerPageNumber);

                // Get the current rawDataPage and offset
                uint64_t insertionRawDataPage = headerPage->rawDataPage;
                uint64_t insertionRawDataOffset = RawDataPage::GetOffset(headerPage->rawDataPage);

                string keyAndValue = key + valueBa;
                zkr = RawDataPage::Write(headerPage->rawDataPage, keyAndValue);
                if (zkr != ZKR_SUCCESS)
                {
                    zklog.error("KeyValueHistoryPage::Write() failed calling RawDataPage.Write result=" + zkresult2string(zkr) + " rawDataPage=" + to_string(rawDataPage) + " rawDataOffset=" + to_string(rawDataOffset) + " key=" + ba2string(key) + " version=" + to_string(version) + " level=" + to_string(level) + " index=" + to_string(index));
                    return zkr;
                }

                // Update entry
                page->keyValueEntry[index][0] = (uint64_t(1) << 60) | (previousVersionOffset << 48) | version;
                page->keyValueEntry[index][1] = (insertionRawDataOffset << 48) + (insertionRawDataPage & U64Mask48);
                page->keyValueEntry[index][2] = 0;

                return ZKR_SUCCESS;
            }

            // If the key is different, move the key to a new KeyValuePage, and write the new key into the new page
            uint64_t newPageNumber = pageManager.getFreePage();
            KeyValueHistoryPage::InitEmptyPage(newPageNumber);
            KeyValueHistoryStruct *newPage = (KeyValueHistoryStruct *)pageManager.getPageAddress(newPageNumber);
            Goldilocks::Element keyFea[4];
            string2fea(fr, ba2string(keyAndValue.substr(0, 32)), keyFea);
            uint8_t newKeyBits[43];
            splitKey6(fr, keyFea, newKeyBits);
            uint64_t newIndex = newKeyBits[level+1];
            newPage->keyValueEntry[newIndex][0] = page->keyValueEntry[index][0] & (U64Mask4<<60 | U64Mask48);
            newPage->keyValueEntry[newIndex][1] = page->keyValueEntry[index][1];
            newPage->keyValueEntry[newIndex][2] = 0; // Invalidate hash, since level has changed

            page->keyValueEntry[index][0] = uint64_t(2) << 60;
            page->keyValueEntry[index][1] = newPageNumber;
            page->keyValueEntry[index][2] = 0; // Invalidate hash, since now it is an intermediate node hash

            return Write(newPageNumber, key, keyBits, version, value, level+1, headerPageNumber);            
        }
        // Intermediate node
        case 2:
        {
            // Call Write with the next page number, which can be modified in it runs out of history
            uint64_t nextPageNumber = page->keyValueEntry[index][1] & U64Mask48;
            zkr = Write(nextPageNumber, key, keyBits, version, value, level + 1, headerPageNumber);
            page->keyValueEntry[index][1] = nextPageNumber;
            return zkr;
        }
        default:
        {
            zklog.error("KeyValueHistoryPage::Write() found invalid control=" + to_string(control) + " pageNumber=" + to_string(pageNumber));
            return ZKR_DB_ERROR;
        }
    }
}

zkresult KeyValueHistoryPage::Write (uint64_t &pageNumber, const string &key, const uint64_t version, const mpz_class &value, uint64_t &headerPageNumber)
{
    zkassert(key.size() == 32);
    zkassert((version & U64Mask48) == version);

    // Get 256 key bits in SMT order, in sets of 6 bits
    Goldilocks::Element keyFea[4];
    string2fea(fr, ba2string(key), keyFea);
    uint8_t keyBitsArray[43];
    splitKey6(fr, keyFea, keyBitsArray);
    string keyBits;
    keyBits.append((char *)keyBitsArray, 43);

    // Start searching with level 0
    return Write(pageNumber, key, keyBits, version, value, 0, headerPageNumber);
}

zkresult KeyValueHistoryPage::calculateHash (const uint64_t pageNumber, Goldilocks::Element (&hash)[4], uint64_t &headerPageNumber)
{
    return calculatePageHash(pageNumber, 0, hash, headerPageNumber);
}
zkresult KeyValueHistoryPage::calculatePageHash (const uint64_t pageNumber, const uint64_t level, Goldilocks::Element (&hash)[4], uint64_t &headerPageNumber)
{
    zkassert(level < 43);
    zkresult zkr;

    // Get the page
    KeyValueHistoryStruct *page = (KeyValueHistoryStruct *)pageManager.getPageAddress(pageNumber);

    // Get the header page
    HeaderStruct *headerPage = (HeaderStruct *)pageManager.getPageAddress(headerPageNumber);

    // Get the SMT level
    uint64_t smtLevel = level*6;

    TreeChunk treeChunk(poseidon);
    treeChunk.resetToZero(smtLevel);

    // For each entry, calculate the hash depending on its type
    unordered_map<uint64_t, uint64_t> positionMap;
    for (uint64_t index = 0; index < 64; index++)
    {
        uint64_t control = page->keyValueEntry[index][0] >> 60;
        switch (control)
        {
            // Empty slot
            case (0):
            {
                continue;
            }

            // Leaf node
            case (1):
            {
                // Read the key and value stored in raw data
                uint64_t rawDataPage = page->keyValueEntry[index][1] & U64Mask48;
                uint64_t rawDataOffset = page->keyValueEntry[index][1] >> 48;
                string keyAndValue;
                zkr = RawDataPage::Read(rawDataPage, rawDataOffset, 64, keyAndValue);
                if (zkr != ZKR_SUCCESS)
                {
                    zklog.error("KeyValueHistoryPage::calculatePageHash() failed calling RawDataPage.Read result=" + zkresult2string(zkr) + " rawDataPage=" + to_string(rawDataPage) + " rawDataOffset=" + to_string(rawDataOffset) + " level=" + to_string(level) + " index=" + to_string(index));
                    return zkr;
                }
                if (keyAndValue.size() != 64)
                {
                    zklog.error("KeyValueHistoryPage::calculatePageHash() called RawDataPage.Read but got invalid keyAndValue.size=" + zkresult2string(keyAndValue.size()) + " rawDataPage=" + to_string(rawDataPage) + " rawDataOffset=" + to_string(rawDataOffset) + " level=" + to_string(level) + " index=" + to_string(index));
                    return zkr;
                }

                // Build child
                Child child;
                child.type = LEAF;
                string2fea(fr, ba2string(keyAndValue.substr(0, 32)), child.leaf.key);
                ba2scalar(child.leaf.value, keyAndValue.c_str() + 32);

                // Calculate position
                uint64_t position = getKeyChildren64Position(index);
                positionMap[index] = position;

                // Set child
                treeChunk.setChild(position, child);
                //zklog.info("KeyValueHistoryPage::calculatePageHash() setting leaf child at position=" + to_string(position));

                continue;
            }

            // Intermediate node
            case 2:
            {
                // Store the intermediate node hash here
                Goldilocks::Element hash[4];

                // If hash is zero, we need to calculate it, and store it in raw data
                if (page->keyValueEntry[index][2] == 0)
                {
                    // Calculate the hash by calling this function recursively
                    uint64_t nextPageNumber = page->keyValueEntry[index][1];
                    zkr = calculatePageHash(nextPageNumber, level+1, hash, headerPageNumber);
                    if (zkr != ZKR_SUCCESS)
                    {
                        return zkr;
                    }

                    // Get the current rawDataPage and offset
                    uint64_t insertionRawDataPage = headerPage->rawDataPage;
                    uint64_t insertionRawDataOffset = RawDataPage::GetOffset(headerPage->rawDataPage);

                    // Store the hash in raw page
                    string hashBa;
                    hashBa = string2ba(fea2string(fr, hash));
                    zkassert(hashBa.size() == 32);
                    zkr = RawDataPage::Write(headerPage->rawDataPage, hashBa);
                    if (zkr != ZKR_SUCCESS)
                    {
                        zklog.error("KeyValueHistoryPage::calculatePageHash() failed calling RawDataPage.Write result=" + zkresult2string(zkr) + " insertionRawDataPage=" + to_string(insertionRawDataPage) + " insertionRawDataOffset=" + to_string(insertionRawDataOffset) + " level=" + to_string(level) + " index=" + to_string(index));
                        return zkr;
                    }

                    // Record the new hash and its raw data
                    page->keyValueEntry[index][2] = (insertionRawDataOffset << 48) | (insertionRawDataPage & U64Mask48);
                }
                // If hash was calculated, get it from raw data
                else
                {
                    uint64_t rawDataPage = page->keyValueEntry[index][2] & U64Mask48;
                    uint64_t rawDataOffset = page->keyValueEntry[index][2] >> 48;
                    string hashString;
                    zkr = RawDataPage::Read(rawDataPage, rawDataOffset, 32, hashString);
                    if (zkr != ZKR_SUCCESS)
                    {
                        zklog.error("KeyValueHistoryPage::calculatePageHash() failed calling RawDataPage.Read result=" + zkresult2string(zkr) + " rawDataPage=" + to_string(rawDataPage) + " rawDataOffset=" + to_string(rawDataOffset) + " level=" + to_string(level) + " index=" + to_string(index));
                        return zkr;
                    }
                    string2fea(fr, ba2string(hashString), hash);
                }

                // Build child
                Child child;
                child.type = INTERMEDIATE;
                child.intermediate.hash[0] = hash[0];
                child.intermediate.hash[1] = hash[1];
                child.intermediate.hash[2] = hash[2];
                child.intermediate.hash[3] = hash[3];

                // Calculate position
                uint64_t position = getKeyChildren64Position(index);

                // Set child
                treeChunk.setChild(position, child);
                //zklog.info("KeyValueHistoryPage::calculatePageHash() setting intermediate child at position=" + to_string(position));

                continue;
            }

            default:
            {
                zklog.error("KeyValueHistoryPage::calculatePageHash() found invalid control=" + to_string(control) + " level=" + to_string(level) + " index=" + to_string(index));
                return ZKR_DB_ERROR;
            }
        }
    }

    // Calculate the hash of this page
    zkr = treeChunk.calculateHash(NULL);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("KeyValueHistoryPage::calculatePageHash() failed calling treeChunk.calculateHash() result=" + zkresult2string(zkr) + " rawDataPage=" + " level=" + to_string(level));
        return zkr;
    }

    //
    // For every leaf node, get the hash
    for (uint64_t index = 0; index < 64; index++)
    {
        uint64_t control = page->keyValueEntry[index][0] >> 60;
        if (control == 1)
        {
            Goldilocks::Element hash[4];
            treeChunk.getLeafHash(positionMap[index], hash);
            
            // Get the current rawDataPage and offset
            uint64_t insertionRawDataPage = headerPage->rawDataPage;
            uint64_t insertionRawDataOffset = RawDataPage::GetOffset(headerPage->rawDataPage);

            // Store the hash in raw page
            string hashBa;
            hashBa = string2ba(fea2string(fr, hash));
            zkassert(hashBa.size() == 32);
            zkr = RawDataPage::Write(headerPage->rawDataPage, hashBa);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("KeyValueHistoryPage::calculatePageHash() failed calling RawDataPage.Write result=" + zkresult2string(zkr) + " insertionRawDataPage=" + to_string(insertionRawDataPage) + " insertionRawDataOffset=" + to_string(insertionRawDataOffset) + " level=" + to_string(level) + " index=" + to_string(index));
                return zkr;
            }

            // Record the new hash and its raw data
            page->keyValueEntry[index][2] = (insertionRawDataOffset << 48) | (insertionRawDataPage & U64Mask48);
        }
    }

    // Get the hash of this page
    const Child &child1 = treeChunk.getChild1();
    zkassert(child1.type == INTERMEDIATE);
    hash[0] = child1.intermediate.hash[0];
    hash[1] = child1.intermediate.hash[1];
    hash[2] = child1.intermediate.hash[2];
    hash[3] = child1.intermediate.hash[3];

    return ZKR_SUCCESS;
}
/*
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
}*/

void KeyValueHistoryPage::Print (const uint64_t pageNumber, bool details, const string &prefix)
{
    zklog.info(prefix + "KeyValueHistoryPage::Print() pageNumber=" + to_string(pageNumber));

    zkresult zkr;

    vector<uint64_t> nextKeyValueHistoryPages;

    // Get the data from this page
    KeyValueHistoryStruct * page = (KeyValueHistoryStruct *)pageManager.getPageAddress(pageNumber);
    zklog.info(prefix + "historyOffset=" + to_string(page->historyOffset) + "=" + to_string((page->historyOffset-minHistoryOffset)/entrySize));

    for (uint64_t i=0; i<64; i++)
    {
        uint64_t control = (page->keyValueEntry[i][0]) >> 60;

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
                if (details)
                {
                    uint64_t version = page->keyValueEntry[i][0] & U64Mask48;
                    // Read the key-value stored in raw data
                    uint64_t rawDataPage = page->keyValueEntry[i][1] & U64Mask48;
                    uint64_t rawDataOffset = page->keyValueEntry[i][1] >> 48;
                    string keyAndValue;
                    zkr = RawDataPage::Read(rawDataPage, rawDataOffset, 64, keyAndValue);
                    if (zkr != ZKR_SUCCESS)
                    {
                        zklog.error(prefix + "KeyValueHistoryPage::Print() failed calling RawDataPage::Read(64) result=" + zkresult2string(zkr) + " i=" + to_string(i) + " rawDataPage=" + to_string(rawDataPage) + " rawDataOffset=" + to_string(rawDataOffset));
                        continue;
                    }

                    // Read the hash stored in raw data
                    string hash;
                    if (page->keyValueEntry[i][2] != 0)
                    {
                        rawDataPage = page->keyValueEntry[i][2] & U64Mask48;
                        rawDataOffset = page->keyValueEntry[i][2] >> 48;
                        zkr = RawDataPage::Read(rawDataPage, rawDataOffset, 32, hash);
                        if (zkr != ZKR_SUCCESS)
                        {
                            zklog.error(prefix + "KeyValueHistoryPage::Print() failed calling RawDataPage::Read(32) result=" + zkresult2string(zkr) + " i=" + to_string(i) + " rawDataPage=" + to_string(rawDataPage) + " rawDataOffset=" + to_string(rawDataOffset));
                            continue;
                        }
                    }

                    zklog.info(prefix + "i=" + to_string(i) + " key=" + ba2string(keyAndValue.substr(0, 32)) + " value=" + ba2string(keyAndValue.substr(32, 32)) + " hash=" + ba2string(hash) + " version=" + to_string(version));
                }
                continue;
            }
            // Intermediate node
            case 2:
            {
                uint64_t nextKeyValueHistoryPage = page->keyValueEntry[i][1] & U64Mask48;
                nextKeyValueHistoryPages.emplace_back(nextKeyValueHistoryPage);

                if (details)
                {
                    string hash;
                    if (page->keyValueEntry[i][2] != 0)
                    {
                        // Read the hash stored in raw data
                        uint64_t rawDataPage = page->keyValueEntry[i][2] & U64Mask48;
                        uint64_t rawDataOffset = page->keyValueEntry[i][2] >> 48;
                        zkr = RawDataPage::Read(rawDataPage, rawDataOffset, 32, hash);
                        if (zkr != ZKR_SUCCESS)
                        {
                            zklog.error(prefix + "KeyValueHistoryPage::Print() failed calling RawDataPage::Read(32) result=" + zkresult2string(zkr) + " i=" + to_string(i) + " rawDataPage=" + to_string(rawDataPage) + " rawDataOffset=" + to_string(rawDataOffset));
                            continue;
                        }
                    }

                    zklog.info(prefix + "i=" + to_string(i) + " nextKeyValueHistoryPage=" + to_string(nextKeyValueHistoryPage) + " hash=" + ba2string(hash));
                }
                continue;
            }
            default:
            {
                zklog.error(prefix + "KeyValuePage::Print() found invalid control=" + to_string(control) + " pageNumber=" + to_string(pageNumber));
                exitProcess();
            }
        }
    }

    for (uint64_t i=0; i<nextKeyValueHistoryPages.size(); i++)
    {
        Print(nextKeyValueHistoryPages[i], details, prefix + " ");
    }
}