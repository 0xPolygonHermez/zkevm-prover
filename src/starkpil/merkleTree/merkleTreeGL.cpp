#include "merkleTreeGL.hpp"
#include <cassert>
#include <algorithm> // std::max

MerkleTreeGL::MerkleTreeGL(Goldilocks::Element *tree)
{
    width = Goldilocks::toU64(tree[0]);
    height = Goldilocks::toU64(tree[1]);
    source = &tree[2];
    nodes = &tree[2 + height * width];
    isNodesAllocated = false;
    isSourceAllocated = false;
};

MerkleTreeGL::MerkleTreeGL(uint64_t _height, uint64_t _width, Goldilocks::Element *_source) : height(_height), width(_width), source(_source)
{

    if (source == NULL)
    {
        source = (Goldilocks::Element *)calloc(height * width, sizeof(Goldilocks::Element));
        isSourceAllocated = true;
    }
    nodes = (Goldilocks::Element *)calloc(getTreeNumElements(), sizeof(Goldilocks::Element));
    isNodesAllocated = true;
};

MerkleTreeGL::~MerkleTreeGL()
{
    if (isSourceAllocated)
    {
        free(source);
    }
    if (isNodesAllocated)
    {
        free(nodes);
    }
}

void MerkleTreeGL::getRoot(Goldilocks::Element *root)
{
    std::memcpy(root, &nodes[getTreeNumElements() - elementSize], elementSize * sizeof(Goldilocks::Element));
    zklog.info("MerkleTree root: [ " + Goldilocks::toString(root[0]) + ", " + Goldilocks::toString(root[1]) + ", " + Goldilocks::toString(root[2]) + ", " + Goldilocks::toString(root[3]) + " ]");
}

void MerkleTreeGL::copySource(Goldilocks::Element *_source)
{
    std::memcpy(source, _source, height * width * sizeof(Goldilocks::Element));
}

uint64_t MerkleTreeGL::getTreeNumElements()
{
    return height * elementSize + (height - 1) * elementSize;
}

uint64_t MerkleTreeGL::getElementSize() 
{
    return elementSize;
}

uint64_t MerkleTreeGL::getMerkleTreeWidth() 
{
    return width;
}

uint64_t MerkleTreeGL::getMerkleProofSize() {
    if(height > 1) {
        return (uint64_t)ceil(log10(height) / log10(hashArity)) * elementSize;
    } 
    return 0;
}

uint64_t MerkleTreeGL::getMerkleProofLength() {
    if(height > 1) {
        return (uint64_t)ceil(log10(height) / log10(hashArity));
    } 
    return 0;
}

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