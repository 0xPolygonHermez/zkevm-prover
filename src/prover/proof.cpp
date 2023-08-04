#include <iostream>
#include "config.hpp"
#include "zkassert.hpp"
#include "proof.hpp"
#include "utils.hpp"
#include "zklog.hpp"

using namespace std;

void ProofGroth16::load (json &proof, PublicInputsExtended &publicinputsextended)
{
    string aux;

    // Parse pi_a array
    if (!proof.contains("pi_a") || !proof["pi_a"].is_array())
    {
        zklog.error("proof does not contain a pi_a array");
        exitProcess();
    }
    for (uint64_t i=0; i<proof["pi_a"].size(); i++)
    {
        zkassert(proof["pi_a"][i].is_string());
        aux = proof["pi_a"][i];
        proofA.push_back(aux);
    }

    // Parse pi_b array of arrays
    if (!proof.contains("pi_b") || !proof["pi_b"].is_array())
    {
        zklog.error("proof does not contain a pi_b array");
        exitProcess();
    }
    for (uint64_t i=0; i<proof["pi_b"].size(); i++)
    {
        ProofX proofx;
        zkassert(proof["pi_b"][i].is_array());
        for (uint64_t j=0; j<proof["pi_b"][i].size(); j++)
        {
            zkassert(proof["pi_b"][i][j].is_string());
            aux = proof["pi_b"][i][j];
            proofx.proof.push_back(aux);
        }
        proofB.push_back(proofx);
    }

    // Parse pi_c array
    if (!proof.contains("pi_c") || !proof["pi_c"].is_array())
    {
        zklog.error("proof does not contain a pi_c array");
        exitProcess();
    }
    for (uint64_t i=0; i<proof["pi_c"].size(); i++)
    {
        zkassert(proof["pi_c"][i].is_string());
        aux = proof["pi_c"][i];
        proofC.push_back(aux);
    }

    // Load public inputs extended
    publicInputsExtended = publicinputsextended;
}