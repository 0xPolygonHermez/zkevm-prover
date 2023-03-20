

#include <string>
#include <iostream>
#include "proof2zkinStark.hpp"
using namespace std;

ordered_json proof2zkinStark(ordered_json &proof)
{
    ordered_json zkinOut = ordered_json::object();

    zkinOut["root1"] = proof["root1"];
    zkinOut["root2"] = proof["root2"];
    zkinOut["root3"] = proof["root3"];
    zkinOut["root4"] = proof["root4"];
    zkinOut["evals"] = proof["evals"];

    ordered_json friProof = proof["fri"];
    for (uint i = 1; i < friProof.size() - 1; i++)
    {
        zkinOut["s" + std::to_string(i) + "_root"] = friProof[i]["root"];
        zkinOut["s" + std::to_string(i) + "_vals"] = ordered_json::array();
        zkinOut["s" + std::to_string(i) + "_siblings"] = ordered_json::array();
        for (uint q = 0; q < friProof[0]["polQueries"].size(); q++)
        {
            zkinOut["s" + std::to_string(i) + "_vals"][q] = friProof[i]["polQueries"][q][0];
            zkinOut["s" + std::to_string(i) + "_siblings"][q] = friProof[i]["polQueries"][q][1];
        }
    }

    zkinOut["s0_vals1"] = ordered_json::array();
    if (friProof[0]["polQueries"][0][1][0].size())
    {
        zkinOut["s0_vals2"] = ordered_json::array();
    }
    if (friProof[0]["polQueries"][0][2][0].size())
    {
        zkinOut["s0_vals3"] = ordered_json::array();
    }

    zkinOut["s0_vals4"] = ordered_json::array();
    zkinOut["s0_valsC"] = ordered_json::array();
    zkinOut["s0_siblings1"] = ordered_json::array();
    if (friProof[0]["polQueries"][0][1][0].size())
    {
        zkinOut["s0_siblings2"] = ordered_json::array();
    }
    if (friProof[0]["polQueries"][0][2][0].size())
    {
        zkinOut["s0_siblings3"] = ordered_json::array();
    }
    zkinOut["s0_siblings4"] = ordered_json::array();
    zkinOut["s0_siblingsC"] = ordered_json::array();

    for (uint i = 0; i < friProof[0]["polQueries"].size(); i++)
    {

        zkinOut["s0_vals1"][i] = friProof[0]["polQueries"][i][0][0];
        zkinOut["s0_siblings1"][i] = friProof[0]["polQueries"][i][0][1];

        if (friProof[0]["polQueries"][0][1][0].size())
        {
            zkinOut["s0_vals2"][i] = friProof[0]["polQueries"][i][1][0];
            zkinOut["s0_siblings2"][i] = friProof[0]["polQueries"][i][1][1];
        }
        if (friProof[0]["polQueries"][0][2][0].size())
        {
            zkinOut["s0_vals3"][i] = friProof[0]["polQueries"][i][2][0];
            zkinOut["s0_siblings3"][i] = friProof[0]["polQueries"][i][2][1];
        }

        zkinOut["s0_vals4"][i] = friProof[0]["polQueries"][i][3][0];
        zkinOut["s0_siblings4"][i] = friProof[0]["polQueries"][i][3][1];

        zkinOut["s0_valsC"][i] = friProof[0]["polQueries"][i][4][0];
        zkinOut["s0_siblingsC"][i] = friProof[0]["polQueries"][i][4][1];
    }

    zkinOut["finalPol"] = friProof[friProof.size() - 1];

    return zkinOut;
};

ordered_json joinzkin(ordered_json &zkin1, ordered_json &zkin2, ordered_json &verKey)
{
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
    zkinOut["a_root1"] = zkin1["root1"];
    zkinOut["a_root2"] = zkin1["root2"];
    zkinOut["a_root3"] = zkin1["root3"];
    zkinOut["a_root4"] = zkin1["root4"];
    zkinOut["a_evals"] = zkin1["evals"];
    zkinOut["a_s0_vals1"] = zkin1["s0_vals1"];
    zkinOut["a_s0_vals3"] = zkin1["s0_vals3"];
    zkinOut["a_s0_vals4"] = zkin1["s0_vals4"];
    zkinOut["a_s0_valsC"] = zkin1["s0_valsC"];
    zkinOut["a_s0_siblings1"] = zkin1["s0_siblings1"];
    zkinOut["a_s0_siblings3"] = zkin1["s0_siblings3"];
    zkinOut["a_s0_siblings4"] = zkin1["s0_siblings4"];
    zkinOut["a_s0_siblingsC"] = zkin1["s0_siblingsC"];
    zkinOut["a_s1_root"] = zkin1["s1_root"];
    zkinOut["a_s2_root"] = zkin1["s2_root"];
    zkinOut["a_s3_root"] = zkin1["s3_root"];
    zkinOut["a_s4_root"] = zkin1["s4_root"];
    zkinOut["a_s1_siblings"] = zkin1["s1_siblings"];
    zkinOut["a_s2_siblings"] = zkin1["s2_siblings"];
    zkinOut["a_s3_siblings"] = zkin1["s3_siblings"];
    zkinOut["a_s4_siblings"] = zkin1["s4_siblings"];
    zkinOut["a_s1_vals"] = zkin1["s1_vals"];
    zkinOut["a_s2_vals"] = zkin1["s2_vals"];
    zkinOut["a_s3_vals"] = zkin1["s3_vals"];
    zkinOut["a_s4_vals"] = zkin1["s4_vals"];
    zkinOut["a_finalPol"] = zkin1["finalPol"];

    zkinOut["b_publics"] = zkin2["publics"];
    zkinOut["b_root1"] = zkin2["root1"];
    zkinOut["b_root2"] = zkin2["root2"];
    zkinOut["b_root3"] = zkin2["root3"];
    zkinOut["b_root4"] = zkin2["root4"];
    zkinOut["b_evals"] = zkin2["evals"];
    zkinOut["b_s0_vals1"] = zkin2["s0_vals1"];
    zkinOut["b_s0_vals3"] = zkin2["s0_vals3"];
    zkinOut["b_s0_vals4"] = zkin2["s0_vals4"];
    zkinOut["b_s0_valsC"] = zkin2["s0_valsC"];
    zkinOut["b_s0_siblings1"] = zkin2["s0_siblings1"];
    zkinOut["b_s0_siblings3"] = zkin2["s0_siblings3"];
    zkinOut["b_s0_siblings4"] = zkin2["s0_siblings4"];
    zkinOut["b_s0_siblingsC"] = zkin2["s0_siblingsC"];
    zkinOut["b_s1_root"] = zkin2["s1_root"];
    zkinOut["b_s2_root"] = zkin2["s2_root"];
    zkinOut["b_s3_root"] = zkin2["s3_root"];
    zkinOut["b_s4_root"] = zkin2["s4_root"];
    zkinOut["b_s1_siblings"] = zkin2["s1_siblings"];
    zkinOut["b_s2_siblings"] = zkin2["s2_siblings"];
    zkinOut["b_s3_siblings"] = zkin2["s3_siblings"];
    zkinOut["b_s4_siblings"] = zkin2["s4_siblings"];
    zkinOut["b_s1_vals"] = zkin2["s1_vals"];
    zkinOut["b_s2_vals"] = zkin2["s2_vals"];
    zkinOut["b_s3_vals"] = zkin2["s3_vals"];
    zkinOut["b_s4_vals"] = zkin2["s4_vals"];
    zkinOut["b_finalPol"] = zkin2["finalPol"];

    zkinOut["rootC"] = ordered_json::array();
    for (int i = 0; i < 4; i++)
    {
        zkinOut["rootC"][i] = to_string(verKey["constRoot"][i]);
    }

    return zkinOut;
}