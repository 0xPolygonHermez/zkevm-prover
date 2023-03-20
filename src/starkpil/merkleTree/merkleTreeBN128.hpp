#ifndef MERKLETREEBN128
#define MERKLETREEBN128

#include <math.h>
#include "fr.hpp"
#include "goldilocks_base_field.hpp"
#include "poseidon_opt.hpp"

#define MT_BN128_ARITY 16
#define GOLDILOCKS_ELEMENTS 3
#define HASH_SIZE 4

class MerkleTreeBN128
{
private:
    void fromGoldilocksToBN128(Goldilocks::Element *source);
    void linearHash();

public:
    RawFr::Element *nodes;
    Goldilocks::Element *source;
    uint64_t numNodes;
    uint64_t height;
    uint64_t width;
    uint64_t source_width;
    bool intialized = false;
    bool isSourceAllocated = false;
    bool isNodesAllocated = false;
    MerkleTreeBN128(){};
    MerkleTreeBN128(uint64_t _height, uint64_t _width);
    MerkleTreeBN128(uint64_t _height, uint64_t _width, Goldilocks::Element *source);
    MerkleTreeBN128(void *source);
    ~MerkleTreeBN128();
    void getRoot(RawFr::Element *root);
    static uint64_t getNumNodes(uint64_t n);
    static uint64_t getMerkleProofLength(uint64_t n);
    static uint64_t getMerkleProofSize(uint64_t n);
    RawFr::Element *address() { return nodes; };
    void getGroupProof(void *res, uint64_t idx);
    Goldilocks::Element getElement(uint64_t idx, uint64_t subIdx);
    void merkle_genMerkleProof(RawFr::Element *proof, uint64_t idx, uint64_t offset, uint64_t n);
    void initialize(Goldilocks::Element *source);
    void merkelize();
};
#endif