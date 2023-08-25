#include "intermediate_node.hpp"
#include "poseidon_goldilocks.hpp"

Goldilocks fr;
PoseidonGoldilocks poseidon;

void IntermediateNode::calculateHash (Goldilocks &fr, PoseidonGoldilocks &poseidon, const Goldilocks::Element (&leftHash)[4], const Goldilocks::Element (&rightHash)[4])
{
    // Prepare input = [leftHash, rightHash, 0000]
    Goldilocks::Element input[12];
    input[0] = leftHash[0];
    input[1] = leftHash[1];
    input[2] = leftHash[2];
    input[3] = leftHash[3];
    input[4] = rightHash[0];
    input[5] = rightHash[1];
    input[6] = rightHash[2];
    input[7] = rightHash[3];
    input[8] = fr.zero();
    input[9] = fr.zero();
    input[10] = fr.zero();
    input[11] = fr.zero();

    // Calculate the poseidon hash
    poseidon.hash(hash, input);
}
