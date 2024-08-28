#include "merkleTreeGL.hpp"
#include <cassert>
#include <algorithm> // std::max


MerkleTreeGL::MerkleTreeGL(uint64_t _arity, bool _custom, uint64_t _height, uint64_t _width, Goldilocks::Element *_source, bool allocate) : height(_height), width(_width), source(_source)
{

    if (source == NULL && allocate)
    {
        source = (Goldilocks::Element *)calloc(height * width, sizeof(Goldilocks::Element));
        isSourceAllocated = true;
    }
    arity = _arity;
    custom = _custom;
    numNodes = getNumNodes(height);
    nodes = (Goldilocks::Element *)calloc(numNodes, sizeof(Goldilocks::Element));
    isNodesAllocated = true;
};

MerkleTreeGL::MerkleTreeGL(uint64_t _arity, bool _custom, Goldilocks::Element *tree)
{
    width = Goldilocks::toU64(tree[0]);
    height = Goldilocks::toU64(tree[1]);
    source = &tree[2];
    arity = _arity;
    custom = _custom;
    numNodes = getNumNodes(height);
    nodes = &tree[2 + height * width];
    isNodesAllocated = false;
    isSourceAllocated = false;
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

uint64_t MerkleTreeGL::getNumSiblings() 
{
    return (arity - 1) * nFieldElements;
}

uint64_t MerkleTreeGL::getMerkleTreeWidth() 
{
    return width;
}

uint64_t MerkleTreeGL::getMerkleProofLength() {
    if(height > 1) {
        return (uint64_t)ceil(log10(height) / log10(arity));
    } 
    return 0;
}

uint64_t MerkleTreeGL::getMerkleProofSize() {
    return getMerkleProofLength() * nFieldElements;
}

uint64_t MerkleTreeGL::getNumNodes(uint64_t height)
{
    return height * nFieldElements + (height - 1) * nFieldElements;
}

void MerkleTreeGL::getRoot(Goldilocks::Element *root)
{
    std::memcpy(root, &nodes[numNodes - nFieldElements], nFieldElements * sizeof(Goldilocks::Element));
    zklog.info("MerkleTree root: [ " + Goldilocks::toString(root[0]) + ", " + Goldilocks::toString(root[1]) + ", " + Goldilocks::toString(root[2]) + ", " + Goldilocks::toString(root[3]) + " ]");
}

void MerkleTreeGL::copySource(Goldilocks::Element *_source)
{
    std::memcpy(source, _source, height * width * sizeof(Goldilocks::Element));
}

void MerkleTreeGL::setSource(Goldilocks::Element *_source)
{
    source = _source;
}

Goldilocks::Element MerkleTreeGL::getElement(uint64_t idx, uint64_t subIdx)
{
    assert((idx > 0) || (idx < width));
    return source[idx * width + subIdx];
};

void MerkleTreeGL::getGroupProof(Goldilocks::Element *proof, uint64_t idx) {
    assert(idx < height);

#pragma omp parallel for
    for (uint64_t i = 0; i < width; i++)
    {
        proof[i] = getElement(idx, i);
    }

    genMerkleProof(&proof[width], idx, 0, height * nFieldElements);
}

void MerkleTreeGL::genMerkleProof(Goldilocks::Element *proof, uint64_t idx, uint64_t offset, uint64_t n)
{
    if (n <= nFieldElements) return;
    
    uint64_t nextIdx = idx >> 1;
    uint64_t si = (idx ^ 1) * nFieldElements;

    std::memcpy(proof, &nodes[offset + si], nFieldElements * sizeof(Goldilocks::Element));

    uint64_t nextN = (std::floor((n - 1) / 8) + 1) * nFieldElements;
    genMerkleProof(&proof[nFieldElements], nextIdx, offset + nextN * 2, nextN);
}

void MerkleTreeGL::merkelize()
{
#ifdef __AVX512__
    PoseidonGoldilocks::merkletree_avx512(nodes, source, width, height);
#else
    PoseidonGoldilocks::merkletree_avx(nodes, source, width, height);
#endif
}