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
#ifdef __AVX512__
    PoseidonGoldilocks::merkletree_avx512(nodes, source, width, height);
#else
    PoseidonGoldilocks::merkletree_avx(nodes, source, width, height);
#endif
}
