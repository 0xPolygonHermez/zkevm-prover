

#include <string>
#include <iostream>
#include "joinzkinMultichainStark.hpp"
using namespace std;

ordered_json joinzkinmultichain(ordered_json &zkin1, ordered_json &zkin2, ordered_json &verKey, uint64_t steps)
{
    ordered_json zkinOut = ordered_json::object();

    zkinOut["a_nPrevBlocksSha256"] = zkin1["nPrevBlocksSha256"];
    zkinOut["a_prevHash"] = zkin1["prevHash"];
    zkinOut["a_nBlocksSha256"] = zkin1["nBlocksSha256"];
    zkinOut["a_outHash"] = zkin1["outHash"];

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
    for (uint64_t i = 1; i < steps; i++)
    {
        zkinOut["a_s" + std::to_string(i) + "_root"] = zkin1["s" + std::to_string(i) + "_root"];
        zkinOut["a_s" + std::to_string(i) + "_siblings"] = zkin1["s" + std::to_string(i) + "_siblings"];
        zkinOut["a_s" + std::to_string(i) + "_vals"] = zkin1["s" + std::to_string(i) + "_vals"];
    }
    zkinOut["a_finalPol"] = zkin1["finalPol"];

    zkinOut["b_nPrevBlocksSha256"] = zkin2["nPrevBlocksSha256"];
    zkinOut["b_prevHash"] = zkin2["prevHash"];
    zkinOut["b_nBlocksSha256"] = zkin2["nBlocksSha256"];
    zkinOut["b_outHash"] = zkin2["outHash"];

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
    for (uint64_t i = 1; i < steps; i++)
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
