#ifndef INTERMEDIATE_NODE_HPP
#define INTERMEDIATE_NODE_HPP

#include "goldilocks_base_field.hpp"
#include "zkresult.hpp"

class IntermediateNode
{
public:
    Goldilocks::Element hash[4]; // = Poseidon(leftHash + rightHash + 0000)

    void calculateHash (const Goldilocks::Element (&leftHash)[4], const Goldilocks::Element (&rightHash)[4]);
};

#endif