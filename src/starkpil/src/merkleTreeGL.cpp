#include "merkleTreeGL.hpp"

void MerkleTreeGL::linearHash()
{
}

void MerkleTreeGL::merkelize()
{
    if (width == 0)
    {
        return;
    }

#pragma omp parallel for
    for (uint64_t i = 0; i < height; i++)
    {
        Goldilocks::Element intermediate[width];
        std::memcpy(&intermediate[0], &source[i * width], width * sizeof(Goldilocks::Element));
        PoseidonGoldilocks::linear_hash(&nodes[i * CAPACITY], intermediate, width);
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
