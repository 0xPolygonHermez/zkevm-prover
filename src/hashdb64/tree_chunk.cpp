#include "tree_chunk.hpp"
#include "zklog.hpp"
#include "scalar.hpp"
#include "timer.hpp"

zkresult TreeChunk::data2children (void)
{
    const uint8_t * pData = (const uint8_t *)data.c_str();
    uint64_t dataSize = data.size();

    // Parse first 2 64-bit integers
    if (dataSize < 2)
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
            continue;
        }

        // If this is a leaf child, parse the key and value
        else if ((isLeaf & mask) != 0)
        {
            // Check there is enough remaining data
            if ((dataSize - decodedSize) < 64)
            {
                zklog.error("TreeChunk::data2children() unexpectedly run out of data dataSize=" + to_string(dataSize) + " decodedSize=" + to_string(decodedSize) + " hash=" + fea2string(db.fr, hash));
                return ZKR_UNSPECIFIED;
            }

            // Mark child as a leaf node
            children64[i].type = LEAF;

            // Decode the leaf key
            mpz_class auxScalar;
            ba2scalar(pData + decodedSize, 32, auxScalar);
            scalar2fea(db.fr, auxScalar, children64[i].leaf.key);

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
                zklog.error("TreeChunk::data2children() unexpectedly run out of data dataSize=" + to_string(dataSize) + " decodedSize=" + to_string(decodedSize) + " hash=" + fea2string(db.fr, hash));
                return ZKR_UNSPECIFIED;
            }

            // Mark child as an intermediate node
            children64[i].type = INTERMEDIATE;

            // Decode the intermediate hash
            mpz_class auxScalar;
            ba2scalar(pData + decodedSize, 32, auxScalar);
            scalar2fea(db.fr, auxScalar, children64[i].intermediate.hash);

            // Increase decoded size
            decodedSize += 32;
        }

        // Move mask bit
        mask = mask << 1;
    }

    return ZKR_SUCCESS;
}

zkresult TreeChunk::children2data (void)
{
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
                zklog.error("TreeChunk::children2data() unexpectedly run out of data encodedSize=" + to_string(encodedSize) + " hash=" + fea2string(db.fr, hash));
                return ZKR_UNSPECIFIED;
            }

            // Encode the leaf key
            mpz_class auxScalar;
            fea2scalar(db.fr, auxScalar, children64[i].leaf.key);
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
                zklog.error("TreeChunk::children2data() unexpectedly run out of data encodedSize=" + to_string(encodedSize) + " hash=" + fea2string(db.fr, hash));
                return ZKR_UNSPECIFIED;
            }

            // Encode the intermediate hash
            mpz_class auxScalar;
            fea2scalar(db.fr, auxScalar, children64[i].intermediate.hash);
            scalar2bytesBE(auxScalar, pData + encodedSize);

            // Increase decoded size
            encodedSize += 32;
        }
        data.resize(encodedSize);
    }

    // Save the first 2 bitmaps
    ((uint64_t *)pData)[0] = isZero;
    ((uint64_t *)pData)[1] = isLeaf;

    data.clear();
    data.append((const char *)pData, encodedSize);

    return ZKR_SUCCESS;
}

zkresult TreeChunk::calculateHash (void)
{
    TimerStart(TREE_CHUNK_CALCULATE_HASH);

    if (level%6 != 0)
    {
        zklog.error("TreeChunk::calculateHash() found level not multiple of 6 level=" + to_string(level));
        return ZKR_UNSPECIFIED; // TODO: return specific errors
    }

    zkresult zkr;

    zkr = calculateChildren(level+5, children64, children32, 32);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("TreeChunk::calculateHash() failed calling calculateChildren(children64, children32, 64) result=" + zkresult2string(zkr));
        return zkr;
    }

    zkr = calculateChildren(level+4, children32, children16, 16);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("TreeChunk::calculateHash() failed calling calculateChildren(children32, children16, 32) result=" + zkresult2string(zkr));
        return zkr;
    }

    zkr = calculateChildren(level+3, children16, children8, 8);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("TreeChunk::calculateHash() failed calling calculateChildren(children16, children8, 16) result=" + zkresult2string(zkr));
        return zkr;
    }

    zkr = calculateChildren(level+2, children8, children4, 4);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("TreeChunk::calculateHash() failed calling calculateChildren(children8, children4, 8) result=" + zkresult2string(zkr));
        return zkr;
    }

    zkr = calculateChildren(level+1, children4, children2, 2);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("TreeChunk::calculateHash() failed calling calculateChildren(children4, children2, 4) result=" + zkresult2string(zkr));
        return zkr;
    }

    zkr = calculateChildren(level, children2, &child1, 1);
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
            hash[0] = db.fr.zero();
            hash[1] = db.fr.zero();
            hash[2] = db.fr.zero();
            hash[3] = db.fr.zero();

            TimerStopAndLog(TREE_CHUNK_CALCULATE_HASH);

            return ZKR_SUCCESS;
        }
        case LEAF:
        {
            // Copy hash from child1 leaf
            hash[0] = child1.leaf.hash[0];
            hash[1] = child1.leaf.hash[1];
            hash[2] = child1.leaf.hash[2];
            hash[3] = child1.leaf.hash[3];

            TimerStopAndLog(TREE_CHUNK_CALCULATE_HASH);

            return ZKR_SUCCESS;
        }
        case INTERMEDIATE:
        {
            // Copy hash from child1 intermediate
            hash[0] = child1.intermediate.hash[0];
            hash[1] = child1.intermediate.hash[1];
            hash[2] = child1.intermediate.hash[2];
            hash[3] = child1.intermediate.hash[3];

            TimerStopAndLog(TREE_CHUNK_CALCULATE_HASH);

            return ZKR_SUCCESS;
        }
        default:
        {
            zklog.error("TreeChunk::calculateHash() found unexpected child1.type=" + to_string(child1.type));
            TimerStopAndLog(TREE_CHUNK_CALCULATE_HASH);
            return ZKR_UNSPECIFIED;
        }
    }
}

zkresult TreeChunk::calculateChildren (const uint64_t level, Child * inputChildren, Child * outputChildren, uint64_t outputSize)
{
    zkassert(inputChildren != NULL);
    zkassert(outputChildren != NULL);

    zkresult zkr;
// TODO: parallelize this for
    for (uint64_t i=0; i<outputSize; i++)
    {
        zkr = calculateChild (level, *(inputChildren + 2*i), *(inputChildren + 2*i + 1), *(outputChildren + i));
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("TreeChunk::calculateChildren() failed calling calculateChild() outputSize=" + to_string(outputSize) + " i=" + to_string(i) + " result=" + zkresult2string(zkr));
            return zkr;
        }
    }
    return ZKR_SUCCESS;
}

zkresult TreeChunk::calculateChild (const uint64_t level, Child &leftChild, Child &rightChild, Child &outputChild)
{
    // Get the left hash
    const Goldilocks::Element (* pLeftHash)[4] = NULL;
    switch (leftChild.type)
    {
        case ZERO:
        {
            outputChild = rightChild;
            return ZKR_SUCCESS;
        }
        case LEAF:
        {
            // Calculate left leaf node hash
            leftChild.leaf.calculateHash();

            pLeftHash = &leftChild.leaf.hash;
        }
        case INTERMEDIATE:
        {
            pLeftHash = &leftChild.intermediate.hash;
        }
        default:
        {
            zklog.error("TreeChunk::calculateChild() found invalid leftChild.type=" + to_string(leftChild.type));
            return ZKR_UNSPECIFIED;
        }
    }

    // Get the right hash
    const Goldilocks::Element (* pRightHash)[4] = NULL;
    switch (rightChild.type)
    {
        case ZERO:
        {
            outputChild = leftChild;
            return ZKR_SUCCESS;
        }
        case LEAF:
        {
            // Calculate right leaf node hash
            rightChild.leaf.calculateHash();

            pRightHash = &rightChild.leaf.hash;
        }
        case INTERMEDIATE:
        {
            pRightHash = &rightChild.intermediate.hash;
        }
        default:
        {
            zklog.error("TreeChunk::calculateChild() found invalid rightChild.type=" + to_string(rightChild.type));
            return ZKR_UNSPECIFIED;
        }
    }

    // Calculate intermediate node hash
    outputChild.type = INTERMEDIATE;
    outputChild.intermediate.calculateHash(*pLeftHash, *pRightHash);

    return ZKR_SUCCESS;
}