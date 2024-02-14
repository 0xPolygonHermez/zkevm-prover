#ifndef MERKLETREEGL
#define MERKLETREEGL

#include "goldilocks_base_field.hpp"
#include "poseidon_goldilocks.hpp"
#include "zklog.hpp"
#include <math.h>

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
    uint64_t hashArity = 2;
    uint64_t elementSize = 4;

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
        return height * elementSize + (height - 1) * elementSize;
    }
    void getRoot(Goldilocks::Element *root)
    {
        std::memcpy(root, &nodes[getTreeNumElements() - elementSize], elementSize * sizeof(Goldilocks::Element));
        zklog.info("MerkleTree root: [ " + Goldilocks::toString(root[0]) + ", " + Goldilocks::toString(root[1]) + ", " + Goldilocks::toString(root[2]) + ", " + Goldilocks::toString(root[3]) + " ]");
    }
    void getGroupProof(Goldilocks::Element *proof, uint64_t idx);


    uint64_t getElementSize() {
        return elementSize;
    }

    uint64_t getMerkleTreeWidth() {
        return width;
    }

    uint64_t getMerkleProofSize() {
        if(height > 1) {
            return (uint64_t)ceil(log10(height) / log10(hashArity)) * elementSize;
        } 
        return 0;
    }

    uint64_t getMerkleProofLength() {
        if(height > 1) {
            return (uint64_t)ceil(log10(height) / log10(hashArity));
        } 
        return 0;
    }
};

#endif