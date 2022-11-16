#ifndef MERKLETREEGL
#define MERKLETREEGL

#include "goldilocks_base_field.hpp"
#include "poseidon_goldilocks.hpp"
#include <math.h>

#define MERKLEHASHGL_ARITY 2
class MerkleTreeGL
{
private:
    void linearHash();
    void getElement(Goldilocks::Element &element, uint64_t idx, uint64_t subIdx);
    void genMerkleProof(Goldilocks::Element *proof, uint64_t idx, uint64_t offset, uint64_t n);

public:
    uint64_t height;
    uint64_t width;
    Goldilocks::Element *source;
    Goldilocks::Element *nodes;
    bool isSourceAllocated = false;
    bool isNodesAllocated = false;
    MerkleTreeGL(){};
    MerkleTreeGL(Goldilocks::Element *tree)
    {
        width = Goldilocks::toU64(tree[0]);
        height = Goldilocks::toU64(tree[1]);
        source = &tree[2];
        nodes = &tree[2 + height * width];
        isNodesAllocated = false;
        isSourceAllocated = false;
    };
    MerkleTreeGL(uint64_t _height, uint64_t _width, Goldilocks::Element *_source) : height(_height), width(_width), source(_source)
    {

        if (source == NULL)
        {
            source = (Goldilocks::Element *)calloc(height * width, sizeof(Goldilocks::Element));
            isSourceAllocated = true;
        }
        nodes = (Goldilocks::Element *)calloc(getTreeNumElements(), sizeof(Goldilocks::Element));
        isNodesAllocated = true;
    };
    ~MerkleTreeGL()
    {
        if (isSourceAllocated)
        {
            free(source);
        }
        if (isNodesAllocated)
        {
            free(nodes);
        }
    };
    void copySource(Goldilocks::Element *_source)
    {
        std::memcpy(source, _source, height * width * sizeof(Goldilocks::Element));
    }

    void merkelize();
    uint64_t getTreeNumElements()
    {
        return height * HASH_SIZE + (height - 1) * HASH_SIZE;
    }
    void getRoot(Goldilocks::Element *root)
    {
        std::memcpy(root, &nodes[getTreeNumElements() - HASH_SIZE], HASH_SIZE * sizeof(Goldilocks::Element));
    }
    void getGroupProof(Goldilocks::Element *proof, uint64_t idx);

    uint64_t MerkleProofSize()
    {
        if (height > 1)
        {
            return (uint64_t)ceil(log10(height) / log10(MERKLEHASHGL_ARITY));
        }
        return 0;
    }
};

#endif