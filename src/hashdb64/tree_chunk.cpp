#include "tree_chunk.hpp"
#include "zklog.hpp"
#include "scalar.hpp"

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
            children[i].type = ZERO;
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
            children[i].type = LEAF;

            // Decode the leaf key
            mpz_class auxScalar;
            ba2scalar(pData + decodedSize, 32, auxScalar);
            scalar2fea(db.fr, auxScalar, children[i].leaf.key);

            // Increase decoded size
            decodedSize += 32;

            // Decode the leaf value
            ba2scalar(pData + decodedSize, 32, children[i].leaf.value);

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
            children[i].type = INTERMEDIATE;

            // Decode the intermediate hash
            mpz_class auxScalar;
            ba2scalar(pData + decodedSize, 32, auxScalar);
            scalar2fea(db.fr, auxScalar, children[i].intermediate.hash);

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
        if (children[i].type == ZERO)
        {
            isZero |= mask;
            continue;
        }

        // If this is a leaf child, encode the key and value
        else if (children[i].type == LEAF)
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
            fea2scalar(db.fr, auxScalar, children[i].leaf.key);
            scalar2bytesBE(auxScalar, pData + encodedSize);

            // Increase decoded size
            encodedSize += 32;

            // Encode the leaf value
            scalar2bytesBE(children[i].leaf.value, pData + encodedSize);

            // Increase decoded size
            encodedSize += 32;
        }

        // If this is an intermediate child, encode the hash
        else if (children[i].type == INTERMEDIATE)
        {
            // Check there is enough remaining data
            if ((TREE_CHUNK_MAX_DATA_SIZE - encodedSize) < 32)
            {
                zklog.error("TreeChunk::children2data() unexpectedly run out of data encodedSize=" + to_string(encodedSize) + " hash=" + fea2string(db.fr, hash));
                return ZKR_UNSPECIFIED;
            }

            // Encode the intermediate hash
            mpz_class auxScalar;
            fea2scalar(db.fr, auxScalar, children[i].intermediate.hash);
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