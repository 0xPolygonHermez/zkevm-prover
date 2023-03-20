#ifndef PROOF_HPP
#define PROOF_HPP

#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "public_inputs_extended.hpp"

using namespace std;
using json = nlohmann::json;

class ProofX
{
public:
    vector<string> proof;
};

class ProofGroth16
{
public:
    vector<string> proofA;
    vector<ProofX> proofB;
    vector<string> proofC;
    PublicInputsExtended publicInputsExtended;

    void load (json &proof, PublicInputsExtended &publicInputsExtended);
};

#endif