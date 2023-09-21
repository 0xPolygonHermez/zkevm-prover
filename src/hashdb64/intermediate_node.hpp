#ifndef INTERMEDIATE_NODE_HPP
#define INTERMEDIATE_NODE_HPP

#include "goldilocks_base_field.hpp"
#include "poseidon_goldilocks.hpp"
#include "zkresult.hpp"
#include "hash_value_gl.hpp"
#include <vector>

class IntermediateNode
{
public:
    Goldilocks::Element hash[4]; // = Poseidon(leftHash + rightHash + 0000)

    void calculateHash (Goldilocks &fr, PoseidonGoldilocks &poseidon, const Goldilocks::Element (&leftHash)[4], const Goldilocks::Element (&rightHash)[4], vector<HashValueGL> *hashValues);
};

#endif