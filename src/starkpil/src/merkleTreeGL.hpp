#ifndef MERKLETREEGL
#define MERKLETREEGL

#include "goldilocks_base_field.hpp"
#include "poseidon_goldilocks.hpp"
#include <math.h>

class MerkleTreeGL
{
private:
    void linearHash();

public:
    MerkleTreeGL(uint64_t _height, uint64_t _width, Goldilocks::Element *_source) : height(_height), width(_width), source(_source)
    {
        nodes = (Goldilocks::Element *)calloc(getTreeNumElements(), sizeof(Goldilocks::Element));
    };
    ~MerkleTreeGL()
    {
        free(nodes);
    };
    uint64_t height;
    uint64_t width;
    Goldilocks::Element *source;
    Goldilocks::Element *nodes;
    void merkelize();
    uint64_t getTreeNumElements()
    {
        return height * HASH_SIZE + (height - 1) * HASH_SIZE;
    }
};

#endif