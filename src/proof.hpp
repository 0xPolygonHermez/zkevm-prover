#ifndef PROOF_HPP
#define PROOF_HPP

#include <string>
#include <vector>
#include "public_inputs_extended.hpp"

using namespace std;

class ProofX
{
public:
    vector<string> proof;
};

class Proof
{
public:
    vector<string> proofA;
    vector<ProofX> proofB;
    vector<string> proofC;
    PublicInputsExtended publicInputsExtended;
};

#endif