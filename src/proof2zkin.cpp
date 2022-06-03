#include <string>
#include <iostream>
#include "proof2zkin.hpp"
#include "config.hpp"
#include "zkassert.hpp"

using namespace std;

void proof2zkin(const json &p, json &zkin)
{
    // Delete all zkin contents, if any
    zkin.clear();

    zkassert(p.contains("proof"));
    zkassert(p["proof"].is_array());
    zkassert(p["proof"].size() >= 4);

    zkin["s0_rootUp1"] = p["proof"][0];
    //cout << zkin["s0_rootUp1"].dump() << endl;
    zkin["s0_rootUp2"] = p["proof"][1];
    //cout << zkin["s0_rootUp2"].dump() << endl;
    zkin["s0_rootUp3"] = p["proof"][2];
    //cout << zkin["s0_rootUp3"].dump() << endl;
    json friProof = p["proof"][3];
    //cout << "friProof:" << friProof.dump() << endl;

    zkin["s0_valsUp1"] = json::array();
    zkin["s0_valsUp2"] = json::array();
    zkin["s0_valsUp3"] = json::array();
    zkin["s0_valsUpC"] = json::array();
    zkin["s0_valsUp1p"] = json::array();
    zkin["s0_valsUp2p"] = json::array();
    zkin["s0_valsUp3p"] = json::array();
    zkin["s0_valsUpCp"] = json::array();
    zkin["s0_siblingsUp1"] = json::array();
    zkin["s0_siblingsUp2"] = json::array();
    zkin["s0_siblingsUp3"] = json::array();
    zkin["s0_siblingsUpC"] = json::array();
    zkin["s0_siblingsUp1p"] = json::array();
    zkin["s0_siblingsUp2p"] = json::array();
    zkin["s0_siblingsUp3p"] = json::array();
    zkin["s0_siblingsUpCp"] = json::array();
    zkin["s0_valsDown"] = json::array();
    zkin["s0_siblingsDownL"] = json::array();
    zkin["s0_siblingsDownH"] = json::array();

    //cout << "friProof[0]:" << friProof.dump() << endl;

    json stepProof = friProof[0];
    zkin["s0_rootDown"] = stepProof["root2"];
    json polQueries = stepProof["polQueries"];
    zkassert(polQueries.is_array());
    //cout << "polQueries:" << polQueries.dump() << endl;
    for (uint64_t i=0; i<stepProof["polQueries"].size(); i++)
    {
        zkin["s0_valsUp1"][i] = stepProof["polQueries"][i][0][0];
        zkin["s0_valsUp2"][i] = stepProof["polQueries"][i][1][0];
        zkin["s0_valsUp3"][i] = stepProof["polQueries"][i][2][0];
        zkin["s0_valsUpC"][i] = stepProof["polQueries"][i][3][0];
        zkin["s0_valsUp1p"][i] = stepProof["polQueries"][i][4][0];
        zkin["s0_valsUp2p"][i] = stepProof["polQueries"][i][5][0];
        zkin["s0_valsUp3p"][i] = stepProof["polQueries"][i][6][0];
        zkin["s0_valsUpCp"][i] = stepProof["polQueries"][i][7][0];

        zkin["s0_siblingsUp1"][i] = stepProof["polQueries"][i][0][1];
        zkin["s0_siblingsUp2"][i] = stepProof["polQueries"][i][1][1];
        zkin["s0_siblingsUp3"][i] = stepProof["polQueries"][i][2][1];
        zkin["s0_siblingsUpC"][i] = stepProof["polQueries"][i][3][1];
        zkin["s0_siblingsUp1p"][i] = stepProof["polQueries"][i][4][1];
        zkin["s0_siblingsUp2p"][i] = stepProof["polQueries"][i][5][1];
        zkin["s0_siblingsUp3p"][i] = stepProof["polQueries"][i][6][1];
        zkin["s0_siblingsUpCp"][i] = stepProof["polQueries"][i][7][1];

        zkin["s0_valsDown"][i] = stepProof["pol2Queries"][i][0];
        zkin["s0_siblingsDownL"][i] = stepProof["pol2Queries"][i][1][0];
        zkin["s0_siblingsDownH"][i] = stepProof["pol2Queries"][i][1][1];
    }

    for (uint64_t s=1; s<p["proof"][3].size()-1; s++)
    {
        json stepProof = friProof[s];

        string ss = to_string(s);
        string sn_valsUp = "s" + ss + "_valsUp";
        string sn_siblingsUp = "s" + ss + "_siblingsUp";
        string sn_valsDown = "s" + ss + "_valsDown";
        string sn_siblingsDownL = "s" + ss + "_siblingsDownL";
        string sn_siblingsDownH = "s" + ss + "_siblingsDownH";
        string sn_rootDown = "s" + ss + "_rootDown";
        
        zkin[sn_valsUp] = json::array();
        zkin[sn_siblingsUp] = json::array();
        zkin[sn_valsDown] = json::array();
        zkin[sn_siblingsDownL] = json::array();
        zkin[sn_siblingsDownH] = json::array();

        zkin[sn_rootDown] = stepProof["root2"];

        for (uint64_t i=0; i<stepProof["polQueries"].size(); i++)
        {    
            zkin[sn_valsUp][i] = stepProof["polQueries"][i][0];
            zkin[sn_siblingsUp][i] = stepProof["polQueries"][i][1];

            zkin[sn_valsDown][i] = stepProof["pol2Queries"][i][0];
            zkin[sn_siblingsDownL][i] = stepProof["pol2Queries"][i][1][0];
            zkin[sn_siblingsDownH][i] = stepProof["pol2Queries"][i][1][1];
        }
    }

    zkin["lastVals"] = p["proof"][3][p["proof"][3].size()-1];
}