

#include <string>
#include <iostream>
#include "proof2zkinStark.hpp"
using namespace std;

ordered_json proof2zkinStark(ordered_json &proof)
{
    ordered_json zkin = ordered_json::object();

    zkin["root1"] = proof["root1"];
    zkin["root2"] = proof["root2"];
    zkin["root3"] = proof["root3"];
    zkin["root4"] = proof["root4"];
    zkin["evals"] = proof["evals"];

    ordered_json friProof = proof["fri"];
    for (uint i = 1; i < friProof.size() - 1; i++)
    {
        zkin["s" + std::to_string(i) + "_root"] = friProof[i]["root"][0];
        zkin["s" + std::to_string(i) + "_vals"] = ordered_json::array();
        zkin["s" + std::to_string(i) + "_siblings"] = ordered_json::array();
        for (uint q = 0; q < friProof[0]["polQueries"].size(); q++)
        {
            zkin["s" + std::to_string(i) + "_vals"][q] = friProof[i]["polQueries"][q][0];
            zkin["s" + std::to_string(i) + "_siblings"][q] = friProof[i]["polQueries"][q][1];
        }
    }

    zkin["s0_vals1"] = ordered_json::array();
    if (friProof[0]["polQueries"][0][1][0].size())
    {
        zkin["s0_vals2"] = ordered_json::array();
    }
    if (friProof[0]["polQueries"][0][2][0].size())
    {
        zkin["s0_vals3"] = ordered_json::array();
    }

    zkin["s0_vals4"] = ordered_json::array();
    zkin["s0_valsC"] = ordered_json::array();
    zkin["s0_siblings1"] = ordered_json::array();
    if (friProof[0]["polQueries"][0][1][0].size())
    {
        zkin["s0_siblings2"] = ordered_json::array();
    }
    if (friProof[0]["polQueries"][0][2][0].size())
    {
        zkin["s0_siblings3"] = ordered_json::array();
    }
    zkin["s0_siblings4"] = ordered_json::array();
    zkin["s0_siblingsC"] = ordered_json::array();

    for (uint i = 0; i < friProof[0]["polQueries"].size(); i++)
    {

        zkin["s0_vals1"][i] = friProof[0]["polQueries"][i][0][0];
        zkin["s0_siblings1"][i] = friProof[0]["polQueries"][i][0][1];

        if (friProof[0]["polQueries"][0][1][0].size())
        {
            zkin["s0_vals2"][i] = friProof[0]["polQueries"][i][1][0];
            zkin["s0_siblings2"][i] = friProof[0]["polQueries"][i][1][1];
        }
        if (friProof[0]["polQueries"][0][2][0].size())
        {
            zkin["s0_vals3"][i] = friProof[0]["polQueries"][i][2][0];
            zkin["s0_siblings3"][i] = friProof[0]["polQueries"][i][2][1];
        }

        zkin["s0_vals4"][i] = friProof[0]["polQueries"][i][3][0];
        zkin["s0_siblings4"][i] = friProof[0]["polQueries"][i][3][1];

        zkin["s0_valsC"][i] = friProof[0]["polQueries"][i][4][0];
        zkin["s0_siblingsC"][i] = friProof[0]["polQueries"][i][4][1];
    }

    zkin["finalPol"] = friProof[friProof.size() - 1];

    return zkin;
};