#include "merkleTreeGL.hpp"
#include <cassert>
#include <algorithm> // std::max

void MerkleTreeGL::getElement(Goldilocks::Element &element, uint64_t idx, uint64_t subIdx)
{
    assert((idx > 0) || (idx < width));

    element = source[idx * width + subIdx];
};

void MerkleTreeGL::getGroupProof(Goldilocks::Element *proof, uint64_t idx)
{

#pragma omp parallel for
    for (uint64_t i = 0; i < width; i++)
    {
        getElement(proof[i], idx, i);
    }

    genMerkleProof(&proof[width], idx, 0, height * HASH_SIZE);
}

void MerkleTreeGL::genMerkleProof(Goldilocks::Element *proof, uint64_t idx, uint64_t offset, uint64_t n)
{
    if (n <= HASH_SIZE)
        return;
    uint64_t nextIdx = idx >> 1;
    uint64_t si = (idx ^ 1) * HASH_SIZE;

    std::memcpy(proof, &nodes[offset + si], HASH_SIZE * sizeof(Goldilocks::Element));

    uint64_t nextN = (std::floor((n - 1) / 8) + 1) * HASH_SIZE;
    genMerkleProof(&proof[HASH_SIZE], nextIdx, offset + nextN * 2, nextN);
}

void MerkleTreeGL::merkelize()
{
    if (height == 0)
    {
        return;
    }
    uint64_t batch_size = std::max((uint64_t)8, (width + 3) / 4);
    uint64_t nbatches = 1;
    if (width > 0)
    {
        nbatches = (width + batch_size - 1) / batch_size;
    }
    uint64_t nlastb = width - (nbatches - 1) * batch_size;

    // Hash the leaves
#pragma omp parallel for
    for (uint64_t i = 0; i < height; i++)
    {
        Goldilocks::Element buff0[nbatches * CAPACITY];
        for (uint64_t j = 0; j < nbatches; j++)
        {
            uint64_t nn = batch_size;
            if (j == nbatches - 1)
                nn = nlastb;
            Goldilocks::Element buff1[batch_size];
            std::memcpy(&buff1[0], &source[i * width + j * batch_size], nn * sizeof(Goldilocks::Element));
            PoseidonGoldilocks::linear_hash(&buff0[j * CAPACITY], buff1, nn);
        }
        PoseidonGoldilocks::linear_hash(&nodes[i * CAPACITY], buff0, nbatches * CAPACITY);
    }

    // Build the merkle tree
    uint64_t pending = height;
    uint64_t nextN = floor((pending - 1) / 2) + 1;

    Goldilocks::Element *cursor_read = nodes;
    Goldilocks::Element *cursor_write = &nodes[height * CAPACITY];

    while (pending > 1)
    {
#pragma omp parallel for
        for (uint64_t i = 0; i < nextN; i++)
        {
            Goldilocks::Element pol_input[SPONGE_WIDTH];
            std::memset(pol_input, 0, SPONGE_WIDTH * sizeof(Goldilocks::Element));
            std::memcpy(pol_input, &cursor_read[i * RATE], RATE * sizeof(Goldilocks::Element));
            PoseidonGoldilocks::hash((Goldilocks::Element(&)[CAPACITY])cursor_write[i * CAPACITY], pol_input);
        }
        pending = pending / 2;
        cursor_read = cursor_write;
        cursor_write = &cursor_write[nextN * CAPACITY];
        nextN = floor((pending - 1) / 2) + 1;
    }
}
