#include "leaf_node.hpp"
#include "scalar.hpp"
#include "poseidon_goldilocks.hpp"
#include "goldilocks_base_field.hpp"
#include "smt_64.hpp"
#include "key.hpp"

Goldilocks fr;
PoseidonGoldilocks poseidon;

void LeafNode::calculateHash (void)
{
    // Prepare input = [value8, 0000]
    Goldilocks::Element input[12];
    scalar2fea(fr, value, input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7]);
    input[8] = fr.zero();
    input[9] = fr.zero();
    input[10] = fr.zero();
    input[11] = fr.zero();

    // Calculate the value hash
    Goldilocks::Element valueHash[4];
    poseidon.hash(valueHash, input);

    // Calculate the remaining key
    Goldilocks::Element rkey[4];
    removeKeyBits(fr, key, level, rkey);

    // Prepare input = [rkey, valueHash, 1000]
    input[0] = rkey[0];
    input[1] = rkey[1];
    input[2] = rkey[2];
    input[3] = rkey[3];
    input[4] = valueHash[0];
    input[5] = valueHash[1];
    input[6] = valueHash[2];
    input[7] = valueHash[3];
    input[8] = fr.one();
    input[9] = fr.zero();
    input[10] = fr.zero();
    input[11] = fr.zero();

    // Calculate the leaf node hash
    poseidon.hash(hash, input);
}