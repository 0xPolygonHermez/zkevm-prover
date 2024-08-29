

#include <string>
#include <iostream>
#include "proof2zkinStark.hpp"
using namespace std;

ordered_json proof2zkinStark(ordered_json &proof, StarkInfo &starkInfo)
{
    uint64_t friSteps = starkInfo.starkStruct.steps.size();
    uint64_t nQueries = starkInfo.starkStruct.nQueries;
    uint64_t nStages = starkInfo.nStages;
    uint64_t nSubProofValues = starkInfo.nSubProofValues;
   
    string valsQ = "s0_vals" + to_string(nStages + 1);
    string siblingsQ = "s0_siblings" + to_string(nStages + 1);
    string rootQ = "root" + to_string(nStages + 1);

    ordered_json zkinOut = ordered_json::object();

    for(uint64_t stage = 1; stage <= nStages; stage++) {
        zkinOut["root" + to_string(stage)] = proof["root" + to_string(stage)];
    }

    zkinOut[rootQ] = proof["root" + to_string(nStages + 1)];
    zkinOut["evals"] = proof["evals"];

    ordered_json friProof = proof["fri"];
    for (uint64_t i = 1; i < friSteps; i++)
    {
        zkinOut["s" + std::to_string(i) + "_root"] = friProof[i]["root"];
        zkinOut["s" + std::to_string(i) + "_vals"] = ordered_json::array();
        zkinOut["s" + std::to_string(i) + "_siblings"] = ordered_json::array();
        for (uint q = 0; q < nQueries; q++)
        {
            zkinOut["s" + std::to_string(i) + "_vals"][q] = friProof[i]["polQueries"][q][0];
            zkinOut["s" + std::to_string(i) + "_siblings"][q] = friProof[i]["polQueries"][q][1];
        }
    }
  
    zkinOut["s0_valsC"] = ordered_json::array();
    zkinOut["s0_siblingsC"] = ordered_json::array();
    
    zkinOut[valsQ] = ordered_json::array();
    zkinOut[siblingsQ] = ordered_json::array();

    for(uint64_t i = 0; i < nStages; ++i) {
        uint64_t stage = i + 1;
        if (friProof[0]["polQueries"][0][i][0].size()) {
            zkinOut["s0_siblings" + to_string(stage)] = ordered_json::array();
            zkinOut["s0_vals" + to_string(stage)] = ordered_json::array();
        }
    }

    for (uint64_t i = 0; i < nQueries; i++) {
        for (uint64_t j = 0; j < nStages; ++j) {
            uint64_t stage = j + 1;
            if (friProof[0]["polQueries"][i][j][0].size()) {
                zkinOut["s0_vals" + to_string(stage)][i] = friProof[0]["polQueries"][i][j][0];
                zkinOut["s0_siblings" + to_string(stage)][i] = friProof[0]["polQueries"][i][j][1];
            }
        }

        zkinOut[valsQ][i] = friProof[0]["polQueries"][i][nStages][0];
        zkinOut[siblingsQ][i] = friProof[0]["polQueries"][i][nStages][1];

        zkinOut["s0_valsC"][i] = friProof[0]["polQueries"][i][nStages + 1][0];
        zkinOut["s0_siblingsC"][i] = friProof[0]["polQueries"][i][nStages + 1][1];
    }

    zkinOut["finalPol"] = friProof[friSteps];

    if (nSubProofValues > 0) {
        zkinOut["subproofValues"] = proof["subproofValues"];
    }
    
    return zkinOut;
};

ordered_json joinzkin(ordered_json &zkin1, ordered_json &zkin2, ordered_json &verKey, StarkInfo &starkInfo)
{

    uint64_t friSteps = starkInfo.starkStruct.steps.size();
    uint64_t nStages = starkInfo.nStages;

    string valsQ = "s0_vals" + to_string(nStages + 1);
    string siblingsQ = "s0_siblings" + to_string(nStages + 1);
    string rootQ = "root" + to_string(nStages + 1);

    ordered_json zkinOut = ordered_json::object();

    // Load oldStateRoot
    for (int i = 0; i < 8; i++)
    {
        zkinOut["publics"][i] = zkin1["publics"][i];
    }

    // Load oldAccInputHash0
    for (int i = 0; i < 8; i++)
    {
        zkinOut["publics"][i + 8] = zkin1["publics"][8 + i];
    }

    zkinOut["publics"][16] = zkin1["publics"][16]; // oldBatchNum

    zkinOut["publics"][17] = zkin1["publics"][17]; // chainId

    zkinOut["publics"][18] = zkin1["publics"][18]; // forkid

    // newStateRoot
    for (int i = 0; i < 8; i++)
    {
        zkinOut["publics"][19 + i] = zkin2["publics"][19 + i];
    }
    // newAccInputHash0
    for (int i = 0; i < 8; i++)
    {
        zkinOut["publics"][27 + i] = zkin2["publics"][27 + i];
    }
    // newLocalExitRoot
    for (int i = 0; i < 8; i++)
    {
        zkinOut["publics"][35 + i] = zkin2["publics"][35 + i];
    }

    zkinOut["publics"][43] = zkin2["publics"][43]; // oldBatchNum

    zkinOut["a_publics"] = zkin1["publics"];

    for(uint64_t stage = 1; stage <= nStages; stage++) {
        zkinOut["a_root" + to_string(stage)] = zkin1["root" + to_string(stage)];
    }
    zkinOut["a_" + rootQ] = zkin1[rootQ];

    zkinOut["a_evals"] = zkin1["evals"];
    zkinOut["a_s0_valsC"] = zkin1["s0_valsC"];
    zkinOut["a_s0_siblingsC"] = zkin1["s0_siblingsC"];
    for(uint64_t stage = 1; stage <= nStages; ++stage) {
        if(starkInfo.mapSectionsN["cm" + to_string(stage)] > 0) {
            zkinOut["a_s0_vals" + to_string(stage)] = zkin1["s0_vals" + to_string(stage)];
            zkinOut["a_s0_siblings" + to_string(stage)] = zkin1["s0_siblings" + to_string(stage)];
        }
    }
    zkinOut["a_" + siblingsQ] = zkin1[siblingsQ];
    zkinOut["a_" + valsQ] = zkin1[valsQ];

    for (uint64_t i = 1; i < friSteps; i++)
    {
        zkinOut["a_s" + std::to_string(i) + "_root"] = zkin1["s" + std::to_string(i) + "_root"];
        zkinOut["a_s" + std::to_string(i) + "_siblings"] = zkin1["s" + std::to_string(i) + "_siblings"];
        zkinOut["a_s" + std::to_string(i) + "_vals"] = zkin1["s" + std::to_string(i) + "_vals"];
    }
    zkinOut["a_finalPol"] = zkin1["finalPol"];

    zkinOut["b_publics"] = zkin2["publics"];
    for(uint64_t stage = 1; stage <= nStages; stage++) {
        zkinOut["b_root" + to_string(stage)] = zkin2["root" + to_string(stage)];
    }
    zkinOut["b_" + rootQ] = zkin2[rootQ];

    zkinOut["b_evals"] = zkin2["evals"];
    zkinOut["b_s0_valsC"] = zkin2["s0_valsC"];
    zkinOut["b_s0_siblingsC"] = zkin2["s0_siblingsC"];
    for(uint64_t stage = 1; stage <= nStages; ++stage) {
        if(starkInfo.mapSectionsN["cm" + to_string(stage)] > 0) {
            zkinOut["b_s0_vals" + to_string(stage)] = zkin2["s0_vals" + to_string(stage)];
            zkinOut["b_s0_siblings" + to_string(stage)] = zkin2["s0_siblings" + to_string(stage)];
        }
    }
    zkinOut["b_" + siblingsQ] = zkin2[siblingsQ];
    zkinOut["b_" + valsQ] = zkin2[valsQ];

    for (uint64_t i = 1; i < friSteps; i++)
    {
        zkinOut["b_s" + std::to_string(i) + "_root"] = zkin2["s" + std::to_string(i) + "_root"];
        zkinOut["b_s" + std::to_string(i) + "_siblings"] = zkin2["s" + std::to_string(i) + "_siblings"];
        zkinOut["b_s" + std::to_string(i) + "_vals"] = zkin2["s" + std::to_string(i) + "_vals"];
    }
    zkinOut["b_finalPol"] = zkin2["finalPol"];

    zkinOut["rootC"] = ordered_json::array();
    for (int i = 0; i < 4; i++)
    {
        zkinOut["rootC"][i] = to_string(verKey["constRoot"][i]);
    }

    return zkinOut;
}