#include "tree_chunk.hpp"
#include "zklog.hpp"
#include "scalar.hpp"
#include "timer.hpp"
#include "zkglobals.hpp"

Goldilocks::Element zeroHash[4] = {0, 0, 0, 0};
/*
zkresult TreeChunk::readDataFromDb (const Goldilocks::Element (&_hash)[4])
{
    // Copy the hash
    hash[0] = _hash[0];
    hash[1] = _hash[1];
    hash[2] = _hash[2];
    hash[3] = _hash[3];
    bHashValid = true;

    // Reset children flags
    bChildren64Valid = false;
    bChildrenRestValid = false;

    // Get the hash string
    string hashString;
    hashString = fea2string(fr, hash);

    // Call the database
    zkresult zkr = ZKR_UNSPECIFIED;
    //zkr = db.read(hashString, hash, data, NULL);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("TreeChunk::readDataFromDb() failed calling db.read() result=" + zkresult2string(zkr) + " hash=" + hashString);
        bDataValid = false;
    }
    else
    {
        bDataValid = true;
    }

    return zkr;
}
*/
zkresult TreeChunk::data2children (void)
{
    if (bChildren64Valid)
    {
        return ZKR_SUCCESS;
    }

    const uint8_t * pData = (const uint8_t *)data.c_str();
    uint64_t dataSize = data.size();

    // Parse first 2 64-bit integers
    if (dataSize < 2*sizeof(uint64_t))
    {
        zklog.error("TreeChunk::data2children() failed invalid data.size=" + to_string(dataSize));
        return ZKR_UNSPECIFIED;
    }
    uint64_t isZero = ((uint64_t *)pData)[0];
    uint64_t isLeaf = ((uint64_t *)pData)[1];
    uint64_t decodedSize = 2*sizeof(uint64_t);

    // Parse the 64 children
    uint64_t mask = 1;
    for (uint64_t i=0; i<TREE_CHUNK_WIDTH; i++)
    {
        // If this is a zero child, simply take note of it
        if ((isZero & mask) != 0)
        {
            children64[i].type = ZERO;
        }

        // If this is a leaf child, parse the key and value
        else if ((isLeaf & mask) != 0)
        {
            // Check there is enough remaining data
            if ((dataSize - decodedSize) < 64)
            {
                zklog.error("TreeChunk::data2children() unexpectedly run out of data dataSize=" + to_string(dataSize) + " decodedSize=" + to_string(decodedSize) + " hash=" + fea2string(fr, hash));
                return ZKR_UNSPECIFIED;
            }

            // Mark child as a leaf node
            children64[i].type = LEAF;

            // Decode the leaf key
            mpz_class auxScalar;
            ba2scalar(pData + decodedSize, 32, auxScalar);
            scalar2fea(fr, auxScalar, children64[i].leaf.key);

            // Increase decoded size
            decodedSize += 32;

            // Decode the leaf value
            ba2scalar(pData + decodedSize, 32, children64[i].leaf.value);

            // Increase decoded size
            decodedSize += 32;
        }

        // If this is an intermediate child, parse the hash
        else
        {
            // Check there is enough remaining data
            if ((dataSize - decodedSize) < 32)
            {
                zklog.error("TreeChunk::data2children() unexpectedly run out of data dataSize=" + to_string(dataSize) + " decodedSize=" + to_string(decodedSize) + " hash=" + fea2string(fr, hash));
                return ZKR_UNSPECIFIED;
            }

            // Mark child as an intermediate node
            children64[i].type = INTERMEDIATE;

            // Decode the intermediate hash
            mpz_class auxScalar;
            ba2scalar(pData + decodedSize, 32, auxScalar);
            scalar2fea(fr, auxScalar, children64[i].intermediate.hash);

            // Increase decoded size
            decodedSize += 32;
        }

        // Move mask bit
        mask = mask << 1;
    }

    // Set children64 as valid
    bChildren64Valid = true;

    return ZKR_SUCCESS;
}

zkresult TreeChunk::children2data (void)
{
    if (bDataValid)
    {
        return ZKR_SUCCESS;
    }

    uint64_t isZero = 0;
    uint64_t isLeaf = 0;
    uint64_t encodedSize = 2*sizeof(uint64_t); // Skip the first 2 bitmaps

    // Get data pointer
    uint8_t localData[TREE_CHUNK_MAX_DATA_SIZE];
    uint8_t * pData = localData;

    // Encode the 64 children
    uint64_t mask = 1;
    for (uint64_t i=0; i<TREE_CHUNK_WIDTH; i++)
    {
        // If this is a zero child, simply take note of it
        if (children64[i].type == ZERO)
        {
            isZero |= mask;

            // Move mask bit
            mask = mask << 1;

            continue;
        }

        // If this is a leaf child, encode the key and value
        else if (children64[i].type == LEAF)
        {
            // Set the corresponding bit in isLeaf
            isLeaf |= mask;

            // Check there is enough remaining data
            if ((TREE_CHUNK_MAX_DATA_SIZE - encodedSize) < 64)
            {
                zklog.error("TreeChunk::children2data() unexpectedly run out of data encodedSize=" + to_string(encodedSize) + " hash=" + fea2string(fr, hash));
                return ZKR_UNSPECIFIED;
            }

            // Encode the leaf key
            mpz_class auxScalar;
            fea2scalar(fr, auxScalar, children64[i].leaf.key);
            scalar2bytesBE(auxScalar, pData + encodedSize);

            // Increase decoded size
            encodedSize += 32;

            // Encode the leaf value
            scalar2bytesBE(children64[i].leaf.value, pData + encodedSize);

            // Increase decoded size
            encodedSize += 32;
        }

        // If this is an intermediate child, encode the hash
        else if (children64[i].type == INTERMEDIATE)
        {
            // Check there is enough remaining data
            if ((TREE_CHUNK_MAX_DATA_SIZE - encodedSize) < 32)
            {
                zklog.error("TreeChunk::children2data() unexpectedly run out of data encodedSize=" + to_string(encodedSize) + " hash=" + fea2string(fr, hash));
                return ZKR_UNSPECIFIED;
            }

            // Encode the intermediate hash
            mpz_class auxScalar;
            fea2scalar(fr, auxScalar, children64[i].intermediate.hash);
            scalar2bytesBE(auxScalar, pData + encodedSize);

            // Increase decoded size
            encodedSize += 32;
        }
        
        data.resize(encodedSize);

        // Move mask bit
        mask = mask << 1;
    }

    // Save the first 2 bitmaps
    ((uint64_t *)pData)[0] = isZero;
    ((uint64_t *)pData)[1] = isLeaf;

    data.clear();
    data.append((const char *)pData, encodedSize);

    bDataValid = true;

    return ZKR_SUCCESS;
}

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