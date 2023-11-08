#include "tree_chunk.hpp"
#include "zklog.hpp"
#include "scalar.hpp"
#include "timer.hpp"
#include "zkglobals.hpp"
#include "key_value_history_page.hpp"
#include "page_manager.hpp"
#include "key_utils.hpp"
#include "constants.hpp"
#include "raw_data_page.hpp"

Goldilocks::Element zeroHash[4] = {0, 0, 0, 0};

uint64_t TreeChunk::numberOfNonZeroChildren (void)
{
    uint64_t numberOfNonZero = 0;
    if (bDataValid)
    {
        const char *pData = data.c_str();
        uint64_t isZero = ((const uint64_t *)pData)[0];

        // Parse the 64 children
        uint64_t mask = 1;
        for (uint64_t i=0; i<TREE_CHUNK_WIDTH; i++)
        {
            // If this is a zero child, simply take note of it
            if ((isZero & mask) == 0)
            {
                numberOfNonZero++;
            }
            mask = mask << 1;
        }
    }
    else if (bChildren64Valid)
    {
        for (uint64_t i=0; i<TREE_CHUNK_WIDTH; i++)
        {
            if (children64[i].type != ZERO)
            {
                numberOfNonZero++;
            }
        }
    }
    else
    {
        zklog.error("TreeChunk::numberOfNonZeroChildren() found bDataValid=bChildren64Valid=false");
        exitProcess();
    }
    
    return numberOfNonZero;
}

zkresult TreeChunk::calculateHash (vector<HashValueGL> *hashValues)
{
    if (bHashValid && bChildrenRestValid)
    {
        return ZKR_SUCCESS;
    }
    bChildrenRestValid = false;

    //TimerStart(TREE_CHUNK_CALCULATE_HASH);

    if (level%6 != 0)
    {
        zklog.error("TreeChunk::calculateHash() found level not multiple of 6 level=" + to_string(level));
        return ZKR_UNSPECIFIED; // TODO: return specific errors
    }

    zkresult zkr;

    zkr = calculateChildren(level+5, children64, children32, 32, hashValues);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("TreeChunk::calculateHash() failed calling calculateChildren(children64, children32, 64) result=" + zkresult2string(zkr));
        return zkr;
    }

    zkr = calculateChildren(level+4, children32, children16, 16, hashValues);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("TreeChunk::calculateHash() failed calling calculateChildren(children32, children16, 32) result=" + zkresult2string(zkr));
        return zkr;
    }
    zkr = calculateChildren(level+3, children16, children8, 8, hashValues);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("TreeChunk::calculateHash() failed calling calculateChildren(children16, children8, 16) result=" + zkresult2string(zkr));
        return zkr;
    }

    zkr = calculateChildren(level+2, children8, children4, 4, hashValues);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("TreeChunk::calculateHash() failed calling calculateChildren(children8, children4, 8) result=" + zkresult2string(zkr));
        return zkr;
    }

    zkr = calculateChildren(level+1, children4, children2, 2, hashValues);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("TreeChunk::calculateHash() failed calling calculateChildren(children4, children2, 4) result=" + zkresult2string(zkr));
        return zkr;
    }

    zkr = calculateChildren(level, children2, &child1, 1, hashValues);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("TreeChunk::calculateHash() failed calling calculateChildren(children2, &child1, 2) result=" + zkresult2string(zkr));
        return zkr;
    }

    switch(child1.type)
    {
        case ZERO:
        {
            // Set hash to zero
            hash[0] = fr.zero();
            hash[1] = fr.zero();
            hash[2] = fr.zero();
            hash[3] = fr.zero();

            // Set flags
            bHashValid = true;
            bChildrenRestValid = true;

            //TimerStopAndLog(TREE_CHUNK_CALCULATE_HASH);

            return ZKR_SUCCESS;
        }
        case LEAF:
        {
            // Copy hash from child1 leaf
            hash[0] = child1.leaf.hash[0];
            hash[1] = child1.leaf.hash[1];
            hash[2] = child1.leaf.hash[2];
            hash[3] = child1.leaf.hash[3];

            // Set flags
            bHashValid = true;
            bChildrenRestValid = true;

            //TimerStopAndLog(TREE_CHUNK_CALCULATE_HASH);

            return ZKR_SUCCESS;
        }
        case INTERMEDIATE:
        {
            // Copy hash from child1 intermediate
            hash[0] = child1.intermediate.hash[0];
            hash[1] = child1.intermediate.hash[1];
            hash[2] = child1.intermediate.hash[2];
            hash[3] = child1.intermediate.hash[3];

            // Set flags
            bHashValid = true;
            bChildrenRestValid = true;

            //TimerStopAndLog(TREE_CHUNK_CALCULATE_HASH);

            return ZKR_SUCCESS;
        }
        default:
        {
            zklog.error("TreeChunk::calculateHash() found unexpected child1.type=" + to_string(child1.type));
            //TimerStopAndLog(TREE_CHUNK_CALCULATE_HASH);
            return ZKR_UNSPECIFIED;
        }
    }
}

zkresult TreeChunk::calculateChildren (const uint64_t level, Child * inputChildren, Child * outputChildren, uint64_t outputSize, vector<HashValueGL> *hashValues)
{
    zkassert(inputChildren != NULL);
    zkassert(outputChildren != NULL);

    zkresult zkr;
// TODO: parallelize this for
    for (uint64_t i=0; i<outputSize; i++)
    {
        zkr = calculateChild (level, *(inputChildren + 2*i), *(inputChildren + 2*i + 1), *(outputChildren + i), hashValues);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("TreeChunk::calculateChildren() failed calling calculateChild() outputSize=" + to_string(outputSize) + " i=" + to_string(i) + " result=" + zkresult2string(zkr));
            return zkr;
        }
    }
    return ZKR_SUCCESS;
}

zkresult TreeChunk::calculateChild (const uint64_t level, Child &leftChild, Child &rightChild, Child &outputChild, vector<HashValueGL> *hashValues)
{
    switch (leftChild.type)
    {
        case ZERO:
        {
            switch (rightChild.type)
            {
                case ZERO:
                {
                    outputChild = rightChild;
                    return ZKR_SUCCESS;
                }
                case LEAF:
                {
                    if (level == 0)
                    {
                        rightChild.leaf.level = level;
                        rightChild.leaf.calculateHash(fr, poseidon, hashValues);
                    }
                    outputChild = rightChild;
                    return ZKR_SUCCESS;
                }
                case INTERMEDIATE:
                {
                    outputChild.type = INTERMEDIATE;
                    outputChild.intermediate.calculateHash(fr, poseidon, zeroHash, rightChild.intermediate.hash, hashValues);
                    return ZKR_SUCCESS;
                }
                default:
                {
                    zklog.error("TreeChunk::calculateChild() found invalid rightChild.type=" + to_string(rightChild.type));
                    exitProcess();
                }
            }
        }
        case LEAF:
        {
            switch (rightChild.type)
            {
                case ZERO:
                {
                    if (level == 0)
                    {
                        leftChild.leaf.level = level;
                        leftChild.leaf.calculateHash(fr, poseidon, hashValues);
                    }
                    outputChild = leftChild;
                    return ZKR_SUCCESS;
                }
                case LEAF:
                {
                    leftChild.leaf.level = level + 1;
                    leftChild.leaf.calculateHash(fr, poseidon, hashValues);
                    rightChild.leaf.level = level + 1;
                    rightChild.leaf.calculateHash(fr, poseidon, hashValues);
                    outputChild.type = INTERMEDIATE;
                    outputChild.intermediate.calculateHash(fr, poseidon, leftChild.leaf.hash, rightChild.leaf.hash, hashValues);
                    return ZKR_SUCCESS;
                }
                case INTERMEDIATE:
                {
                    leftChild.leaf.level = level + 1;
                    leftChild.leaf.calculateHash(fr, poseidon, hashValues);
                    outputChild.type = INTERMEDIATE;
                    outputChild.intermediate.calculateHash(fr, poseidon, leftChild.leaf.hash, rightChild.intermediate.hash, hashValues);
                    return ZKR_SUCCESS;
                }
                default:
                {
                    zklog.error("TreeChunk::calculateChild() found invalid rightChild.type=" + to_string(rightChild.type));
                    exitProcess();
                }
            }
        }
        case INTERMEDIATE:
        {
            switch (rightChild.type)
            {
                case ZERO:
                {
                    outputChild.type = INTERMEDIATE;
                    outputChild.intermediate.calculateHash(fr, poseidon, leftChild.intermediate.hash, zeroHash, hashValues);
                    return ZKR_SUCCESS;
                }
                case LEAF:
                {
                    rightChild.leaf.level = level + 1;
                    rightChild.leaf.calculateHash(fr, poseidon, hashValues);
                    outputChild.type = INTERMEDIATE;
                    outputChild.intermediate.calculateHash(fr, poseidon, leftChild.intermediate.hash, rightChild.leaf.hash, hashValues);
                    return ZKR_SUCCESS;
                }
                case INTERMEDIATE:
                {
                    outputChild.type = INTERMEDIATE;
                    outputChild.intermediate.calculateHash(fr, poseidon, leftChild.intermediate.hash, rightChild.intermediate.hash, hashValues);
                    return ZKR_SUCCESS;
                }
                default:
                {
                    zklog.error("TreeChunk::calculateChild() found invalid rightChild.type=" + to_string(rightChild.type));
                    exitProcess();
                }
            }
        }
        default:
        {
            zklog.error("TreeChunk::calculateChild() found invalid leftChild.type=" + to_string(leftChild.type));
            exitProcess();
        }
    }

    return ZKR_UNSPECIFIED;
}

void TreeChunk::getLeafHash(const uint64_t _position, Goldilocks::Element (&result)[4])
{
    zkassert(_position < 64);
    zkassert(bHashValid == true);

    // Search in children64
    uint64_t position = _position;
    if (children64[position].type != LEAF)
    {
        zklog.error("TreeChunk::getLeafHash() found children64[position].type=" + to_string(children64[position].type) + " position=" + to_string(position));
        exitProcess();
    }
    if (!feaIsZero(children64[position].leaf.hash))
    {
        result[0] = children64[position].leaf.hash[0];
        result[1] = children64[position].leaf.hash[1];
        result[2] = children64[position].leaf.hash[2];
        result[3] = children64[position].leaf.hash[3];
        return;
    }

    // Search in children32
    position = position >> 1;
    if (children32[position].type != LEAF)
    {
        zklog.error("TreeChunk::getLeafHash() found children32[position].type=" + to_string(children64[position].type) + " position=" + to_string(position));
        exitProcess();
    }
    if (!feaIsZero(children32[position].leaf.hash))
    {
        result[0] = children32[position].leaf.hash[0];
        result[1] = children32[position].leaf.hash[1];
        result[2] = children32[position].leaf.hash[2];
        result[3] = children32[position].leaf.hash[3];
        return;
    }

    // Search in children16
    position = position >> 1;
    if (children16[position].type != LEAF)
    {
        zklog.error("TreeChunk::getLeafHash() found children16[position].type=" + to_string(children64[position].type) + " position=" + to_string(position));
        exitProcess();
    }
    if (!feaIsZero(children16[position].leaf.hash))
    {
        result[0] = children16[position].leaf.hash[0];
        result[1] = children16[position].leaf.hash[1];
        result[2] = children16[position].leaf.hash[2];
        result[3] = children16[position].leaf.hash[3];
        return;
    }

    // Search in children8
    position = position >> 1;
    if (children8[position].type != LEAF)
    {
        zklog.error("TreeChunk::getLeafHash() found children8[position].type=" + to_string(children64[position].type) + " position=" + to_string(position));
        exitProcess();
    }
    if (!feaIsZero(children8[position].leaf.hash))
    {
        result[0] = children8[position].leaf.hash[0];
        result[1] = children8[position].leaf.hash[1];
        result[2] = children8[position].leaf.hash[2];
        result[3] = children8[position].leaf.hash[3];
        return;
    }

    // Search in children4
    position = position >> 1;
    if (children4[position].type != LEAF)
    {
        zklog.error("TreeChunk::getLeafHash() found children4[position].type=" + to_string(children64[position].type) + " position=" + to_string(position));
        exitProcess();
    }
    if (!feaIsZero(children4[position].leaf.hash))
    {
        result[0] = children4[position].leaf.hash[0];
        result[1] = children4[position].leaf.hash[1];
        result[2] = children4[position].leaf.hash[2];
        result[3] = children4[position].leaf.hash[3];
        return;
    }

    // Search in children2
    position = position >> 1;
    if (children2[position].type != LEAF)
    {
        zklog.error("TreeChunk::getLeafHash() found children2[position].type=" + to_string(children64[position].type) + " position=" + to_string(position));
        exitProcess();
    }
    if (!feaIsZero(children2[position].leaf.hash))
    {
        result[0] = children2[position].leaf.hash[0];
        result[1] = children2[position].leaf.hash[1];
        result[2] = children2[position].leaf.hash[2];
        result[3] = children2[position].leaf.hash[3];
        return;
    }

    zklog.error("TreeChunk::getLeafHash() failed searching for the lef hash position=" + to_string(_position) + " position=" + to_string(position));
    exitProcess();    
}

void TreeChunk::print(void) const
{
    string aux = "";
    zklog.info("TreeChunk::print():");
    zklog.info("  level=" + to_string(level));
    zklog.info("  bHashValid=" + to_string(bHashValid));
    zklog.info("  hash=" + fea2string(fr, hash));
    zklog.info("  bChildrenRestValid=" + to_string(bChildrenRestValid));
    zklog.info("  child1=" + child1.print(fr));

    aux = "";
    for (uint64_t i=0; i<2; i++) aux += children2[i].getTypeLetter();
    zklog.info("  children2=" + aux);

    for (uint64_t i=0; i<2; i++)
    {
        if ((children2[i].type != ZERO) && (children2[i].type != UNSPECIFIED) )
        {
            zklog.info( "    children2[" + to_string(i) + "]=" + children2[i].print(fr));
        }
    }

    aux = "";
    for (uint64_t i=0; i<4; i++) aux += children4[i].getTypeLetter();
    zklog.info("  children4=" + aux);

    for (uint64_t i=0; i<4; i++)
    {
        if ((children4[i].type != ZERO) && (children4[i].type != UNSPECIFIED))
        {
            zklog.info( "    children4[" + to_string(i) + "]=" + children4[i].print(fr));
        }
    }

    aux = "";
    for (uint64_t i=0; i<8; i++) aux += children8[i].getTypeLetter();
    zklog.info("  children8=" + aux);

    for (uint64_t i=0; i<8; i++)
    {
        if ((children8[i].type != ZERO) && (children8[i].type != UNSPECIFIED))
        {
            zklog.info( "    children8[" + to_string(i) + "]=" + children8[i].print(fr));
        }
    }

    aux = "";
    for (uint64_t i=0; i<16; i++) aux += children16[i].getTypeLetter();
    zklog.info("  children16=" + aux);

    for (uint64_t i=0; i<16; i++)
    {
        if ((children16[i].type != ZERO) && (children16[i].type != UNSPECIFIED))
        {
            zklog.info( "    children16[" + to_string(i) + "]=" + children16[i].print(fr));
        }
    }

    aux = "";
    for (uint64_t i=0; i<32; i++) aux += children32[i].getTypeLetter();
    zklog.info("  children32=" + aux);

    for (uint64_t i=0; i<32; i++)
    {
        if ((children32[i].type != ZERO) && (children32[i].type != UNSPECIFIED))
        {
            zklog.info( "    children32[" + to_string(i) + "]=" + children32[i].print(fr));
        }
    }

    zklog.info("  bChildren64Valid=" + to_string(bChildren64Valid));

    aux = "";
    for (uint64_t i=0; i<64; i++) aux += children64[i].getTypeLetter();
    zklog.info("  children64=" + aux);

    for (uint64_t i=0; i<64; i++)
    {
        if ((children64[i].type != ZERO) && (children64[i].type != UNSPECIFIED))
        {
            zklog.info( "    children64[" + to_string(i) + "]=" + children64[i].print(fr));
        }
    }
    zklog.info("  bDataValid=" + to_string(bDataValid));
    zklog.info("  data.size=" + to_string(data.size()));
}

zkresult TreeChunk::loadFromKeyValueHistoryPage (PageContext &ctx, const uint64_t pageNumber, const uint64_t version, const uint64_t _level)
{
    zkresult zkr;

    // Copy level
    level = _level;

    // Get the data from this page
    KeyValueHistoryStruct * page = (KeyValueHistoryStruct *)ctx.pageManager.getPageAddress(pageNumber);

    for (uint64_t index = 0; index < 64; index++)
    {
        // Get control
        uint64_t control = page->keyValueEntry[index][0] >> 60;

        Child child;

        // Check control
        switch (control)
        {
            // Empty slot
            case 0:
            {
                children64[index].type = ZERO;
                continue;            
            }
            
            // Leaf node
            case 1:
            {
                uint64_t *keyValueEntry = page->keyValueEntry[index];
                uint64_t foundVersion = 0;
                while (true)
                {
                    // Read the latest version of this key in this page
                    foundVersion = keyValueEntry[0] & U64Mask48;

                    // If it is equal or lower, then we found the slot, although it could be occupied by a different key, so we need to check
                    if (version >= foundVersion)
                    {
                        // Check that we have calculated the hash
                        if (page->keyValueEntry[2] == 0)
                        {
                            zklog.error("TreeChunk::loadFromKeyValueHistoryPage() found leaf node with hash=0 version=" + to_string(version) + " level=" + to_string(level) + " index=" + to_string(index) + " page=" + to_string(pageNumber));
                            return ZKR_DB_ERROR;
                        }

                        // Get key and value from raw data
                        uint64_t rawDataPage = keyValueEntry[1] & U64Mask48;
                        uint64_t rawDataOffset = keyValueEntry[1] >> 48;
                        string keyValue;
                        zkr = RawDataPage::Read(ctx, rawDataPage, rawDataOffset, 64, keyValue);
                        if (zkr != ZKR_SUCCESS)
                        {
                            zklog.error("TreeChunk::loadFromKeyValueHistoryPage() failed calling RawDataPage.Read result=" + zkresult2string(zkr) + " rawDataPage=" + to_string(rawDataPage) + " rawDataOffset=" + to_string(rawDataOffset) + " version=" + to_string(version) + " level=" + to_string(level) + " index=" + to_string(index));
                            return zkr;
                        }
                        if (keyValue.size() != 64)
                        {
                            zklog.error("TreeChunk::loadFromKeyValueHistoryPage() called RawDataPage.Read and got invalid length=" + to_string(keyValue.size()) + " rawDataPage=" + to_string(rawDataPage) + " rawDataOffset=" + to_string(rawDataOffset) + " version=" + to_string(version) + " level=" + to_string(level) + " index=" + to_string(index));
                            return zkr;
                        }

                        // Set key and value in child
                        string keyBa = keyValue.substr(0, 32);
                        string keyString = ba2string(keyBa);                        
                        children64[index].type = LEAF;
                        string2fea(fr, keyString, children64[index].leaf.key); 
                        ba2scalar((uint8_t *)keyValue.c_str() + 32, 32, children64[index].leaf.value);

                        // Get hash from raw data
                        rawDataPage = keyValueEntry[2] & U64Mask48;
                        rawDataOffset = keyValueEntry[2] >> 48;
                        if (rawDataPage == 0)
                        {
                            zklog.error("TreeChunk::loadFromKeyValueHistoryPage() found hash rawDataPage=0 pageNumber=" + to_string(pageNumber) + " version=" + to_string(version) + " level=" + to_string(level) + " index=" + to_string(index));
                            return ZKR_DB_ERROR;
                        }
                        string hashBa;
                        zkr = RawDataPage::Read(ctx, rawDataPage, rawDataOffset, 32, hashBa);
                        if (zkr != ZKR_SUCCESS)
                        {
                            zklog.error("TreeChunk::loadFromKeyValueHistoryPage() failed calling RawDataPage.Read result=" + zkresult2string(zkr) + " rawDataPage=" + to_string(rawDataPage) + " rawDataOffset=" + to_string(rawDataOffset) + " version=" + to_string(version) + " level=" + to_string(level) + " index=" + to_string(index));
                            return zkr;
                        }
                        if (hashBa.size() != 32)
                        {
                            zklog.error("TreeChunk::loadFromKeyValueHistoryPage() called RawDataPage.Read and got invalid length=" + to_string(keyValue.size()) + " rawDataPage=" + to_string(rawDataPage) + " rawDataOffset=" + to_string(rawDataOffset) + " version=" + to_string(version) + " level=" + to_string(level) + " index=" + to_string(index));
                            return zkr;
                        }

                        // Set hash in child
                        string hashString = ba2string(hashBa);
                        string2fea(fr, hashString, children64[index].leaf.hash);

                        break;
                    }

                    if (version > foundVersion)
                    {
                        continue;
                    }
                    
                    // Search for 
                    uint64_t previousVersionOffset = (page->keyValueEntry[index][1] >> 48) & U64Mask12;

                    // If there is no previous version for this key, then this is a zero
                    if (previousVersionOffset == 0)
                    {
                        children64[index].type = ZERO;
                        return ZKR_SUCCESS;
                    }

                    // If not zero, then check the range of the previous version
                    if ( (previousVersionOffset < KeyValueHistoryPage::minHistoryOffset) ||
                        (previousVersionOffset > KeyValueHistoryPage::maxHistoryOffset) ||
                        ((previousVersionOffset & U64Mask4) != 0) )
                    {
                        zklog.error("TreeChunk::loadFromKeyValueHistoryPage() found invalid previousVersionOffset=" + to_string(previousVersionOffset));
                        return ZKR_DB_ERROR;
                    }

                    // Get the previous version entry
                    keyValueEntry = (uint64_t *)((uint8_t *)page + previousVersionOffset);
                }
            }

            // Intermediate node
            case 2:
            {
                // Check that we have calculated the hash
                if (page->keyValueEntry[index][2] == 0)
                {
                    zklog.error("TreeChunk::loadFromKeyValueHistoryPage() found intermediate node with hash=0 version=" + to_string(version) + " level=" + to_string(level) + " index=" + to_string(index) + " page=" + to_string(pageNumber));
                    return ZKR_DB_ERROR;
                }

                // Get hash from raw data
                uint64_t rawDataPage = page->keyValueEntry[index][2] & U64Mask48;
                uint64_t rawDataOffset = page->keyValueEntry[index][2] >> 48;
                string hashBa;
                zkr = RawDataPage::Read(ctx, rawDataPage, rawDataOffset, 32, hashBa);
                if (zkr != ZKR_SUCCESS)
                {
                    zklog.error("TreeChunk::loadFromKeyValueHistoryPage() failed calling RawDataPage.Read result=" + zkresult2string(zkr) + " rawDataPage=" + to_string(rawDataPage) + " rawDataOffset=" + to_string(rawDataOffset) + " version=" + to_string(version) + " level=" + to_string(level) + " index=" + to_string(index));
                    return zkr;
                }
                if (hashBa.size() != 32)
                {
                    zklog.error("TreeChunk::loadFromKeyValueHistoryPage() called RawDataPage.Read and got invalid length=" + to_string(hashBa.size()) + " rawDataPage=" + to_string(rawDataPage) + " rawDataOffset=" + to_string(rawDataOffset) + " version=" + to_string(version) + " level=" + to_string(level) + " index=" + to_string(index));
                    return zkr;
                }

                // Set hash in child
                string hashString = ba2string(hashBa);
                children64[index].type = INTERMEDIATE;
                string2fea(fr, hashString, children64[index].intermediate.hash);

                continue;
            }

            default:
            {
                zklog.error("TreeChunk::loadFromKeyValueHistoryPage() found invalid control=" + to_string(control) + " pageNumber=" + to_string(pageNumber));
                return ZKR_DB_ERROR;
            }
        }
    }

    // Calculate all intermediate hashes
    zkr = calculateHash(NULL);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("TreeChunk::loadFromKeyValueHistoryPage() failed calling calculateHash() result=" + zkresult2string(zkr) + " version=" + to_string(version) + " level=" + to_string(level));
        return zkr;
    }

    return ZKR_SUCCESS;
}

#define DOUBLE_CHECK_HASH_VALUES

zkresult TreeChunk::getHashValues (const uint64_t children64Position, vector<HashValueGL> *hashValues)
{
    if (hashValues == NULL)
    {
        return ZKR_SUCCESS;
    }

    zkresult zkr;

    zkassert(children64Position < 64);
    zkassert(bHashValid == true);

    // Search in children64
    uint64_t position = children64Position;
    switch (children64[position].type)
    {
        case ZERO:
        {
            break;
        }
        case LEAF:
        {
            // Re-create the value hash
            HashValueGL hashValue;
            scalar2fea(fr, children64[position].leaf.value, hashValue.value[0], hashValue.value[1], hashValue.value[2], hashValue.value[3], hashValue.value[4], hashValue.value[5], hashValue.value[6], hashValue.value[7]);
            hashValue.value[8] = fr.zero();
            hashValue.value[9] = fr.zero();
            hashValue.value[10] = fr.zero();
            hashValue.value[11] = fr.zero();
            poseidon.hash(hashValue.hash, hashValue.value);
            hashValues->emplace_back(hashValue);

            // Re-create the leaf hash
            removeKeyBits(fr, children64[position].leaf.key, level + 5, (Goldilocks::Element (&)[4])hashValue.value);
            hashValue.value[4] = hashValue.hash[0];
            hashValue.value[5] = hashValue.hash[1];
            hashValue.value[6] = hashValue.hash[2];
            hashValue.value[7] = hashValue.hash[3];
            hashValue.value[8] = fr.one();
            hashValue.value[9] = fr.zero();
            hashValue.value[10] = fr.zero();
            hashValue.value[11] = fr.zero();
            hashValue.hash[0] = children64[position].leaf.hash[0];
            hashValue.hash[1] = children64[position].leaf.hash[1];
            hashValue.hash[2] = children64[position].leaf.hash[2];
            hashValue.hash[3] = children64[position].leaf.hash[3];
            hashValues->emplace_back(hashValue);

#ifdef DOUBLE_CHECK_HASH_VALUES
            Goldilocks::Element hash[4];
            poseidon.hash(hash, hashValue.value);
            zkassert(fr.equal(hash[0], hashValue.hash[0]));
            zkassert(fr.equal(hash[1], hashValue.hash[1]));
            zkassert(fr.equal(hash[2], hashValue.hash[2]));
            zkassert(fr.equal(hash[3], hashValue.hash[3]));
#endif

            break;
        }
        case INTERMEDIATE:
        {
            break;
        }
        default:
        {
            zklog.error("TreeChunk::getHashValues() found unexpected children64[position].type=" + to_string(children64[position].type) + " position=" + to_string(position));
            return ZKR_DB_ERROR;
        }
    }

    // Search in children32
    position = position >> 1;
    switch (children32[position].type)
    {
        case ZERO:
        {
            break;
        }
        case LEAF:
        {
            // Re-create the value hash
            HashValueGL hashValue;
            scalar2fea(fr, children32[position].leaf.value, hashValue.value[0], hashValue.value[1], hashValue.value[2], hashValue.value[3], hashValue.value[4], hashValue.value[5], hashValue.value[6], hashValue.value[7]);
            hashValue.value[8] = fr.zero();
            hashValue.value[9] = fr.zero();
            hashValue.value[10] = fr.zero();
            hashValue.value[11] = fr.zero();
            poseidon.hash(hashValue.hash, hashValue.value);
            hashValues->emplace_back(hashValue);

            // Re-create the leaf node hash
            removeKeyBits(fr, children32[position].leaf.key, level + 5, (Goldilocks::Element (&)[4])hashValue.value);
            hashValue.value[4] = hashValue.hash[0];
            hashValue.value[5] = hashValue.hash[1];
            hashValue.value[6] = hashValue.hash[2];
            hashValue.value[7] = hashValue.hash[3];

            // Set the capacity to {1,0,0,0}
            hashValue.value[8] = fr.one();
            hashValue.value[9] = fr.zero();
            hashValue.value[10] = fr.zero();
            hashValue.value[11] = fr.zero();

            // Set the hash
            hashValue.hash[0] = children32[position].leaf.hash[0];
            hashValue.hash[1] = children32[position].leaf.hash[1];
            hashValue.hash[2] = children32[position].leaf.hash[2];
            hashValue.hash[3] = children32[position].leaf.hash[3];

            // Add to the hash values vector
            hashValues->emplace_back(hashValue);

#ifdef DOUBLE_CHECK_HASH_VALUES
            Goldilocks::Element hash[4];
            poseidon.hash(hash, hashValue.value);
            zkassert(fr.equal(hash[0], hashValue.hash[0]));
            zkassert(fr.equal(hash[1], hashValue.hash[1]));
            zkassert(fr.equal(hash[2], hashValue.hash[2]));
            zkassert(fr.equal(hash[3], hashValue.hash[3]));
#endif

            break;
        }
        case INTERMEDIATE:
        {
            // Calculate the positions of the left and right nodes
            uint64_t positionLeft = position << 1;
            uint64_t positionRight = positionLeft + 1;

            // Re-create the intermediate node hash
            HashValueGL hashValue;

            // Get the left node hash
            zkr = children64[positionLeft].getHash((Goldilocks::Element (&)[4])hashValue.value[0]);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("TreeChunk::getHashValues() failed calling children64[positionLeft].getHash() zkr=" + zkresult2string(zkr) + " positionLeft=" + to_string(positionLeft));
                return zkr;
            }

            // Get the right node hash
            zkr = children64[positionRight].getHash((Goldilocks::Element (&)[4])hashValue.value[4]);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("TreeChunk::getHashValues() failed calling children64[positionRight].getHash() zkr=" + zkresult2string(zkr) + " positionRight=" + to_string(positionRight));
                return zkr;
            }

            // Set the capacity to {0,0,0,0}
            hashValue.value[8] = fr.zero();
            hashValue.value[9] = fr.zero();
            hashValue.value[10] = fr.zero();
            hashValue.value[11] = fr.zero();

            // Set the hash
            hashValue.hash[0] = children32[position].intermediate.hash[0];
            hashValue.hash[1] = children32[position].intermediate.hash[1];
            hashValue.hash[2] = children32[position].intermediate.hash[2];
            hashValue.hash[3] = children32[position].intermediate.hash[3];

            // Add to the hash values vector
            hashValues->emplace_back(hashValue);

#ifdef DOUBLE_CHECK_HASH_VALUES
            Goldilocks::Element hash[4];
            poseidon.hash(hash, hashValue.value);
            zkassert(fr.equal(hash[0], hashValue.hash[0]));
            zkassert(fr.equal(hash[1], hashValue.hash[1]));
            zkassert(fr.equal(hash[2], hashValue.hash[2]));
            zkassert(fr.equal(hash[3], hashValue.hash[3]));
#endif
            break;
        }
        default:
        {
            zklog.error("TreeChunk::getHashValues() found unexpected children32[position].type=" + to_string(children32[position].type) + " position=" + to_string(position));
            return ZKR_DB_ERROR;
        }
    }

    // Search in children16
    position = position >> 1;
    switch (children16[position].type)
    {
        case ZERO:
        {
            break;
        }
        case LEAF:
        {
            // Re-create the value hash
            HashValueGL hashValue;
            scalar2fea(fr, children16[position].leaf.value, hashValue.value[0], hashValue.value[1], hashValue.value[2], hashValue.value[3], hashValue.value[4], hashValue.value[5], hashValue.value[6], hashValue.value[7]);
            hashValue.value[8] = fr.zero();
            hashValue.value[9] = fr.zero();
            hashValue.value[10] = fr.zero();
            hashValue.value[11] = fr.zero();
            poseidon.hash(hashValue.hash, hashValue.value);
            hashValues->emplace_back(hashValue);

            // Re-create the leaf node hash
            removeKeyBits(fr, children16[position].leaf.key, level + 5, (Goldilocks::Element (&)[4])hashValue.value);
            hashValue.value[4] = hashValue.hash[0];
            hashValue.value[5] = hashValue.hash[1];
            hashValue.value[6] = hashValue.hash[2];
            hashValue.value[7] = hashValue.hash[3];

            // Set the capacity to {1,0,0,0}
            hashValue.value[8] = fr.one();
            hashValue.value[9] = fr.zero();
            hashValue.value[10] = fr.zero();
            hashValue.value[11] = fr.zero();

            // Set the hash
            hashValue.hash[0] = children16[position].leaf.hash[0];
            hashValue.hash[1] = children16[position].leaf.hash[1];
            hashValue.hash[2] = children16[position].leaf.hash[2];
            hashValue.hash[3] = children16[position].leaf.hash[3];

            // Add to the hash values vector
            hashValues->emplace_back(hashValue);

#ifdef DOUBLE_CHECK_HASH_VALUES
            Goldilocks::Element hash[4];
            poseidon.hash(hash, hashValue.value);
            zkassert(fr.equal(hash[0], hashValue.hash[0]));
            zkassert(fr.equal(hash[1], hashValue.hash[1]));
            zkassert(fr.equal(hash[2], hashValue.hash[2]));
            zkassert(fr.equal(hash[3], hashValue.hash[3]));
#endif

            break;
        }
        case INTERMEDIATE:
        {
            // Calculate the positions of the left and right nodes
            uint64_t positionLeft = position << 1;
            uint64_t positionRight = positionLeft + 1;

            // Re-create the intermediate node hash
            HashValueGL hashValue;

            // Get the left node hash
            zkr = children32[positionLeft].getHash((Goldilocks::Element (&)[4])hashValue.value[0]);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("TreeChunk::getHashValues() failed calling children32[positionLeft].getHash() zkr=" + zkresult2string(zkr) + " positionLeft=" + to_string(positionLeft));
                return zkr;
            }

            // Get the right node hash
            zkr = children32[positionRight].getHash((Goldilocks::Element (&)[4])hashValue.value[4]);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("TreeChunk::getHashValues() failed calling children32[positionRight].getHash() zkr=" + zkresult2string(zkr) + " positionRight=" + to_string(positionRight));
                return zkr;
            }

            // Set the capacity to {0,0,0,0}
            hashValue.value[8] = fr.zero();
            hashValue.value[9] = fr.zero();
            hashValue.value[10] = fr.zero();
            hashValue.value[11] = fr.zero();

            // Set the hash
            hashValue.hash[0] = children16[position].intermediate.hash[0];
            hashValue.hash[1] = children16[position].intermediate.hash[1];
            hashValue.hash[2] = children16[position].intermediate.hash[2];
            hashValue.hash[3] = children16[position].intermediate.hash[3];

            // Add to the hash values vector
            hashValues->emplace_back(hashValue);

#ifdef DOUBLE_CHECK_HASH_VALUES
            Goldilocks::Element hash[4];
            poseidon.hash(hash, hashValue.value);
            zkassert(fr.equal(hash[0], hashValue.hash[0]));
            zkassert(fr.equal(hash[1], hashValue.hash[1]));
            zkassert(fr.equal(hash[2], hashValue.hash[2]));
            zkassert(fr.equal(hash[3], hashValue.hash[3]));
#endif
            break;
        }
        default:
        {
            zklog.error("TreeChunk::getHashValues() found unexpected children16[position].type=" + to_string(children16[position].type) + " position=" + to_string(position));
            return ZKR_DB_ERROR;
        }
    }

    // Search in children8
    position = position >> 1;
    switch (children8[position].type)
    {
        case ZERO:
        {
            break;
        }
        case LEAF:
        {
            // Re-create the value hash
            HashValueGL hashValue;
            scalar2fea(fr, children8[position].leaf.value, hashValue.value[0], hashValue.value[1], hashValue.value[2], hashValue.value[3], hashValue.value[4], hashValue.value[5], hashValue.value[6], hashValue.value[7]);
            hashValue.value[8] = fr.zero();
            hashValue.value[9] = fr.zero();
            hashValue.value[10] = fr.zero();
            hashValue.value[11] = fr.zero();
            poseidon.hash(hashValue.hash, hashValue.value);
            hashValues->emplace_back(hashValue);

            // Re-create the leaf node hash
            removeKeyBits(fr, children8[position].leaf.key, level + 5, (Goldilocks::Element (&)[4])hashValue.value);
            hashValue.value[4] = hashValue.hash[0];
            hashValue.value[5] = hashValue.hash[1];
            hashValue.value[6] = hashValue.hash[2];
            hashValue.value[7] = hashValue.hash[3];

            // Set the capacity to {1,0,0,0}
            hashValue.value[8] = fr.one();
            hashValue.value[9] = fr.zero();
            hashValue.value[10] = fr.zero();
            hashValue.value[11] = fr.zero();

            // Set the hash
            hashValue.hash[0] = children8[position].leaf.hash[0];
            hashValue.hash[1] = children8[position].leaf.hash[1];
            hashValue.hash[2] = children8[position].leaf.hash[2];
            hashValue.hash[3] = children8[position].leaf.hash[3];

            // Add to the hash values vector
            hashValues->emplace_back(hashValue);

#ifdef DOUBLE_CHECK_HASH_VALUES
            Goldilocks::Element hash[4];
            poseidon.hash(hash, hashValue.value);
            zkassert(fr.equal(hash[0], hashValue.hash[0]));
            zkassert(fr.equal(hash[1], hashValue.hash[1]));
            zkassert(fr.equal(hash[2], hashValue.hash[2]));
            zkassert(fr.equal(hash[3], hashValue.hash[3]));
#endif

            break;
        }
        case INTERMEDIATE:
        {
            // Calculate the positions of the left and right nodes
            uint64_t positionLeft = position << 1;
            uint64_t positionRight = positionLeft + 1;

            // Re-create the intermediate node hash
            HashValueGL hashValue;

            // Get the left node hash
            zkr = children16[positionLeft].getHash((Goldilocks::Element (&)[4])hashValue.value[0]);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("TreeChunk::getHashValues() failed calling children16[positionLeft].getHash() zkr=" + zkresult2string(zkr) + " positionLeft=" + to_string(positionLeft));
                return zkr;
            }

            // Get the right node hash
            zkr = children16[positionRight].getHash((Goldilocks::Element (&)[4])hashValue.value[4]);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("TreeChunk::getHashValues() failed calling children16[positionRight].getHash() zkr=" + zkresult2string(zkr) + " positionRight=" + to_string(positionRight));
                return zkr;
            }

            // Set the capacity to {0,0,0,0}
            hashValue.value[8] = fr.zero();
            hashValue.value[9] = fr.zero();
            hashValue.value[10] = fr.zero();
            hashValue.value[11] = fr.zero();

            // Set the hash
            hashValue.hash[0] = children8[position].intermediate.hash[0];
            hashValue.hash[1] = children8[position].intermediate.hash[1];
            hashValue.hash[2] = children8[position].intermediate.hash[2];
            hashValue.hash[3] = children8[position].intermediate.hash[3];

            // Add to the hash values vector
            hashValues->emplace_back(hashValue);

#ifdef DOUBLE_CHECK_HASH_VALUES
            Goldilocks::Element hash[4];
            poseidon.hash(hash, hashValue.value);
            zkassert(fr.equal(hash[0], hashValue.hash[0]));
            zkassert(fr.equal(hash[1], hashValue.hash[1]));
            zkassert(fr.equal(hash[2], hashValue.hash[2]));
            zkassert(fr.equal(hash[3], hashValue.hash[3]));
#endif
            break;
        }
        default:
        {
            zklog.error("TreeChunk::getHashValues() found unexpected children8[position].type=" + to_string(children8[position].type) + " position=" + to_string(position));
            return ZKR_DB_ERROR;
        }
    }

    // Search in children4
    position = position >> 1;
    switch (children4[position].type)
    {
        case ZERO:
        {
            break;
        }
        case LEAF:
        {
            // Re-create the value hash
            HashValueGL hashValue;
            scalar2fea(fr, children4[position].leaf.value, hashValue.value[0], hashValue.value[1], hashValue.value[2], hashValue.value[3], hashValue.value[4], hashValue.value[5], hashValue.value[6], hashValue.value[7]);
            hashValue.value[8] = fr.zero();
            hashValue.value[9] = fr.zero();
            hashValue.value[10] = fr.zero();
            hashValue.value[11] = fr.zero();
            poseidon.hash(hashValue.hash, hashValue.value);
            hashValues->emplace_back(hashValue);

            // Re-create the leaf node hash
            removeKeyBits(fr, children4[position].leaf.key, level + 5, (Goldilocks::Element (&)[4])hashValue.value);
            hashValue.value[4] = hashValue.hash[0];
            hashValue.value[5] = hashValue.hash[1];
            hashValue.value[6] = hashValue.hash[2];
            hashValue.value[7] = hashValue.hash[3];

            // Set the capacity to {1,0,0,0}
            hashValue.value[8] = fr.one();
            hashValue.value[9] = fr.zero();
            hashValue.value[10] = fr.zero();
            hashValue.value[11] = fr.zero();

            // Set the hash
            hashValue.hash[0] = children4[position].leaf.hash[0];
            hashValue.hash[1] = children4[position].leaf.hash[1];
            hashValue.hash[2] = children4[position].leaf.hash[2];
            hashValue.hash[3] = children4[position].leaf.hash[3];

            // Add to the hash values vector
            hashValues->emplace_back(hashValue);

#ifdef DOUBLE_CHECK_HASH_VALUES
            Goldilocks::Element hash[4];
            poseidon.hash(hash, hashValue.value);
            zkassert(fr.equal(hash[0], hashValue.hash[0]));
            zkassert(fr.equal(hash[1], hashValue.hash[1]));
            zkassert(fr.equal(hash[2], hashValue.hash[2]));
            zkassert(fr.equal(hash[3], hashValue.hash[3]));
#endif

            break;
        }
        case INTERMEDIATE:
        {
            // Calculate the positions of the left and right nodes
            uint64_t positionLeft = position << 1;
            uint64_t positionRight = positionLeft + 1;

            // Re-create the intermediate node hash
            HashValueGL hashValue;

            // Get the left node hash
            zkr = children8[positionLeft].getHash((Goldilocks::Element (&)[4])hashValue.value[0]);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("TreeChunk::getHashValues() failed calling children8[positionLeft].getHash() zkr=" + zkresult2string(zkr) + " positionLeft=" + to_string(positionLeft));
                return zkr;
            }

            // Get the right node hash
            zkr = children8[positionRight].getHash((Goldilocks::Element (&)[4])hashValue.value[4]);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("TreeChunk::getHashValues() failed calling children8[positionRight].getHash() zkr=" + zkresult2string(zkr) + " positionRight=" + to_string(positionRight));
                return zkr;
            }

            // Set the capacity to {0,0,0,0}
            hashValue.value[8] = fr.zero();
            hashValue.value[9] = fr.zero();
            hashValue.value[10] = fr.zero();
            hashValue.value[11] = fr.zero();

            // Set the hash
            hashValue.hash[0] = children4[position].intermediate.hash[0];
            hashValue.hash[1] = children4[position].intermediate.hash[1];
            hashValue.hash[2] = children4[position].intermediate.hash[2];
            hashValue.hash[3] = children4[position].intermediate.hash[3];

            // Add to the hash values vector
            hashValues->emplace_back(hashValue);

#ifdef DOUBLE_CHECK_HASH_VALUES
            Goldilocks::Element hash[4];
            poseidon.hash(hash, hashValue.value);
            zkassert(fr.equal(hash[0], hashValue.hash[0]));
            zkassert(fr.equal(hash[1], hashValue.hash[1]));
            zkassert(fr.equal(hash[2], hashValue.hash[2]));
            zkassert(fr.equal(hash[3], hashValue.hash[3]));
#endif
            break;
        }
        default:
        {
            zklog.error("TreeChunk::getHashValues() found unexpected children4[position].type=" + to_string(children4[position].type) + " position=" + to_string(position));
            return ZKR_DB_ERROR;
        }
    }

    // Search in children2
    position = position >> 1;
    switch (children2[position].type)
    {
        case ZERO:
        {
            break;
        }
        case LEAF:
        {
            // Re-create the value hash
            HashValueGL hashValue;
            scalar2fea(fr, children2[position].leaf.value, hashValue.value[0], hashValue.value[1], hashValue.value[2], hashValue.value[3], hashValue.value[4], hashValue.value[5], hashValue.value[6], hashValue.value[7]);
            hashValue.value[8] = fr.zero();
            hashValue.value[9] = fr.zero();
            hashValue.value[10] = fr.zero();
            hashValue.value[11] = fr.zero();
            poseidon.hash(hashValue.hash, hashValue.value);
            hashValues->emplace_back(hashValue);

            // Re-create the leaf node hash
            removeKeyBits(fr, children2[position].leaf.key, level + 5, (Goldilocks::Element (&)[4])hashValue.value);
            hashValue.value[4] = hashValue.hash[0];
            hashValue.value[5] = hashValue.hash[1];
            hashValue.value[6] = hashValue.hash[2];
            hashValue.value[7] = hashValue.hash[3];

            // Set the capacity to {1,0,0,0}
            hashValue.value[8] = fr.one();
            hashValue.value[9] = fr.zero();
            hashValue.value[10] = fr.zero();
            hashValue.value[11] = fr.zero();

            // Set the hash
            hashValue.hash[0] = children2[position].leaf.hash[0];
            hashValue.hash[1] = children2[position].leaf.hash[1];
            hashValue.hash[2] = children2[position].leaf.hash[2];
            hashValue.hash[3] = children2[position].leaf.hash[3];

            // Add to the hash values vector
            hashValues->emplace_back(hashValue);

#ifdef DOUBLE_CHECK_HASH_VALUES
            Goldilocks::Element hash[4];
            poseidon.hash(hash, hashValue.value);
            zkassert(fr.equal(hash[0], hashValue.hash[0]));
            zkassert(fr.equal(hash[1], hashValue.hash[1]));
            zkassert(fr.equal(hash[2], hashValue.hash[2]));
            zkassert(fr.equal(hash[3], hashValue.hash[3]));
#endif

            break;
        }
        case INTERMEDIATE:
        {
            // Calculate the positions of the left and right nodes
            uint64_t positionLeft = position << 1;
            uint64_t positionRight = positionLeft + 1;

            // Re-create the intermediate node hash
            HashValueGL hashValue;

            // Get the left node hash
            zkr = children4[positionLeft].getHash((Goldilocks::Element (&)[4])hashValue.value[0]);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("TreeChunk::getHashValues() failed calling children4[positionLeft].getHash() zkr=" + zkresult2string(zkr) + " positionLeft=" + to_string(positionLeft));
                return zkr;
            }

            // Get the right node hash
            zkr = children4[positionRight].getHash((Goldilocks::Element (&)[4])hashValue.value[4]);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("TreeChunk::getHashValues() failed calling children4[positionRight].getHash() zkr=" + zkresult2string(zkr) + " positionRight=" + to_string(positionRight));
                return zkr;
            }

            // Set the capacity to {0,0,0,0}
            hashValue.value[8] = fr.zero();
            hashValue.value[9] = fr.zero();
            hashValue.value[10] = fr.zero();
            hashValue.value[11] = fr.zero();

            // Set the hash
            hashValue.hash[0] = children2[position].intermediate.hash[0];
            hashValue.hash[1] = children2[position].intermediate.hash[1];
            hashValue.hash[2] = children2[position].intermediate.hash[2];
            hashValue.hash[3] = children2[position].intermediate.hash[3];

            // Add to the hash values vector
            hashValues->emplace_back(hashValue);

#ifdef DOUBLE_CHECK_HASH_VALUES
            Goldilocks::Element hash[4];
            poseidon.hash(hash, hashValue.value);
            zkassert(fr.equal(hash[0], hashValue.hash[0]));
            zkassert(fr.equal(hash[1], hashValue.hash[1]));
            zkassert(fr.equal(hash[2], hashValue.hash[2]));
            zkassert(fr.equal(hash[3], hashValue.hash[3]));
#endif
            break;
        }
        default:
        {
            zklog.error("TreeChunk::getHashValues() found unexpected children2[position].type=" + to_string(children2[position].type) + " position=" + to_string(position));
            return ZKR_DB_ERROR;
        }
    }

    // Search in child1
    position = position >> 1;
    switch (child1.type)
    {
        case ZERO:
        {
            break;
        }
        case LEAF:
        {
            // Re-create the value hash
            HashValueGL hashValue;
            scalar2fea(fr, child1.leaf.value, hashValue.value[0], hashValue.value[1], hashValue.value[2], hashValue.value[3], hashValue.value[4], hashValue.value[5], hashValue.value[6], hashValue.value[7]);
            hashValue.value[8] = fr.zero();
            hashValue.value[9] = fr.zero();
            hashValue.value[10] = fr.zero();
            hashValue.value[11] = fr.zero();
            poseidon.hash(hashValue.hash, hashValue.value);
            hashValues->emplace_back(hashValue);

            // Re-create the leaf node hash
            removeKeyBits(fr, child1.leaf.key, level + 5, (Goldilocks::Element (&)[4])hashValue.value[0]);
            hashValue.value[4] = hashValue.hash[0];
            hashValue.value[5] = hashValue.hash[1];
            hashValue.value[6] = hashValue.hash[2];
            hashValue.value[7] = hashValue.hash[3];

            // Set the capacity to {1,0,0,0}
            hashValue.value[8] = fr.one();
            hashValue.value[9] = fr.zero();
            hashValue.value[10] = fr.zero();
            hashValue.value[11] = fr.zero();

            // Set the hash
            hashValue.hash[0] = child1.leaf.hash[0];
            hashValue.hash[1] = child1.leaf.hash[1];
            hashValue.hash[2] = child1.leaf.hash[2];
            hashValue.hash[3] = child1.leaf.hash[3];

            // Add to the hash values vector
            hashValues->emplace_back(hashValue);

#ifdef DOUBLE_CHECK_HASH_VALUES
            Goldilocks::Element hash[4];
            poseidon.hash(hash, hashValue.value);
            zkassert(fr.equal(hash[0], hashValue.hash[0]));
            zkassert(fr.equal(hash[1], hashValue.hash[1]));
            zkassert(fr.equal(hash[2], hashValue.hash[2]));
            zkassert(fr.equal(hash[3], hashValue.hash[3]));
#endif

            break;
        }
        case INTERMEDIATE:
        {
            // Calculate the positions of the left and right nodes
            uint64_t positionLeft = position << 1;
            uint64_t positionRight = positionLeft + 1;

            // Re-create the intermediate node hash
            HashValueGL hashValue;

            // Get the left node hash
            zkr = children2[positionLeft].getHash((Goldilocks::Element (&)[4])hashValue.value[0]);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("TreeChunk::getHashValues() failed calling children2[positionLeft].getHash() zkr=" + zkresult2string(zkr) + " positionLeft=" + to_string(positionLeft));
                return zkr;
            }

            // Get the right node hash
            zkr = children2[positionRight].getHash((Goldilocks::Element (&)[4])hashValue.value[4]);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("TreeChunk::getHashValues() failed calling children2[positionRight].getHash() zkr=" + zkresult2string(zkr) + " positionRight=" + to_string(positionRight));
                return zkr;
            }

            // Set the capacity to {0,0,0,0}
            hashValue.value[8] = fr.zero();
            hashValue.value[9] = fr.zero();
            hashValue.value[10] = fr.zero();
            hashValue.value[11] = fr.zero();

            // Set the hash
            hashValue.hash[0] = child1.intermediate.hash[0];
            hashValue.hash[1] = child1.intermediate.hash[1];
            hashValue.hash[2] = child1.intermediate.hash[2];
            hashValue.hash[3] = child1.intermediate.hash[3];

            // Add to the hash values vector
            hashValues->emplace_back(hashValue);

#ifdef DOUBLE_CHECK_HASH_VALUES
            Goldilocks::Element hash[4];
            poseidon.hash(hash, hashValue.value);
            zkassert(fr.equal(hash[0], hashValue.hash[0]));
            zkassert(fr.equal(hash[1], hashValue.hash[1]));
            zkassert(fr.equal(hash[2], hashValue.hash[2]));
            zkassert(fr.equal(hash[3], hashValue.hash[3]));
#endif
            break;
        }
        default:
        {
            zklog.error("TreeChunk::getHashValues() found unexpected child1.type=" + to_string(child1.type) + " position=" + to_string(position));
            return ZKR_DB_ERROR;
        }
    }

    return ZKR_SUCCESS;
}