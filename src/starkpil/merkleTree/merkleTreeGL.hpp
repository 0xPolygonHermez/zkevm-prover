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
    MerkleTreeGL(Goldilocks::Element *tree);
    MerkleTreeGL(uint64_t _height, uint64_t _width, Goldilocks::Element *_source);
    ~MerkleTreeGL();
    void getRoot(Goldilocks::Element *root);
    void copySource(Goldilocks::Element *_source);
    void merkelize();
    uint64_t getTreeNumElements();
    void getGroupProof(Goldilocks::Element *proof, uint64_t idx);
    uint64_t getElementSize(); 
    uint64_t getMerkleTreeWidth(); 
    uint64_t getMerkleProofSize(); 
    uint64_t getMerkleProofLength();
};

#endif