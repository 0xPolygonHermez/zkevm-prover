#ifndef LEAF_NODE_HPP
#define LEAF_NODE_HPP

#include <vector>
#include "goldilocks_base_field.hpp"
#include "poseidon_goldilocks.hpp"
#include "zkresult.hpp"
#include "hash_value_gl.hpp"
#include "zkglobals.hpp"

using namespace std;

class LeafNode
{
public:
    uint64_t            level;
    Goldilocks::Element key[4];
    mpz_class           value; // 256 bits
    Goldilocks::Element hash[4]; // = Poseidon(rkey + Poseidon(value, 0000) + 1000)

    LeafNode()
    {
        hash[0] = fr.zero();
        hash[1] = fr.zero();
        hash[2] = fr.zero();
        hash[3] = fr.zero();
    }

    void calculateHash (Goldilocks &fr, PoseidonGoldilocks &poseidon, vector<HashValueGL> *hashValues);
};

#endif