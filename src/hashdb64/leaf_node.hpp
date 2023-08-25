#ifndef LEAF_NODE_HPP
#define LEAF_NODE_HPP

#include "goldilocks_base_field.hpp"
#include "zkresult.hpp"

class LeafNode
{
public:
    uint64_t            level;
    Goldilocks::Element key[4];
    mpz_class           value; // 256 bits
    Goldilocks::Element hash[4]; // = Poseidon(rkey + Poseidon(value, 0000) + 1000)

    void calculateHash (void);
};

#endif