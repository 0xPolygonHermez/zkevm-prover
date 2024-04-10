

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

ordered_json joinzkinBatchRecursive2(ordered_json &zkin1, ordered_json &zkin2, ordered_json &verKey, uint64_t steps)
{
    ordered_json zkinOut = ordered_json::object();

    BatchPublics batchPublics;

    // Define output publics
    for (int i = 0; i < 8; i++)
    {
        zkinOut["publics"][batchPublics.oldStateRootPos + i] = zkin1["publics"][batchPublics.oldStateRootPos + i];
        zkinOut["publics"][batchPublics.oldBatchAccInputHashPos + i] = zkin1["publics"][batchPublics.oldBatchAccInputHashPos + i];
        zkinOut["publics"][batchPublics.previousL1InfoTreeRootPos + i] = zkin1["publics"][batchPublics.previousL1InfoTreeRootPos + i];

        zkinOut["publics"][batchPublics.newStateRootPos + i] = zkin2["publics"][batchPublics.newStateRootPos + i];
        zkinOut["publics"][batchPublics.newBatchAccInputHashPos + i] = zkin2["publics"][batchPublics.newBatchAccInputHashPos + i];
        zkinOut["publics"][batchPublics.currentL1InfoTreeRootPos + i] = zkin2["publics"][batchPublics.currentL1InfoTreeRootPos + i];

        zkinOut["publics"][batchPublics.newLocalExitRootPos + i] = zkin2["publics"][batchPublics.newLocalExitRootPos + i];
    }

    zkinOut["publics"][batchPublics.previousL1InfoTreeIndexPos] = zkin1["publics"][batchPublics.previousL1InfoTreeIndexPos];

    zkinOut["publics"][batchPublics.chainIdPos] = zkin1["publics"][batchPublics.chainIdPos];
    zkinOut["publics"][batchPublics.forkIdPos] = zkin1["publics"][batchPublics.forkIdPos];

    zkinOut["publics"][batchPublics.currentL1InfoTreeIndexPos] = zkin2["publics"][batchPublics.currentL1InfoTreeIndexPos];
    zkinOut["publics"][batchPublics.newLastTimestampPos] = zkin2["publics"][batchPublics.newLastTimestampPos];

    // Add first recursive proof inputs
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
    for (uint64_t i = 1; i < steps; i++)
    {
        zkinOut["a_s" + std::to_string(i) + "_root"] = zkin1["s" + std::to_string(i) + "_root"];
        zkinOut["a_s" + std::to_string(i) + "_siblings"] = zkin1["s" + std::to_string(i) + "_siblings"];
        zkinOut["a_s" + std::to_string(i) + "_vals"] = zkin1["s" + std::to_string(i) + "_vals"];
    }
    zkinOut["a_finalPol"] = zkin1["finalPol"];
    zkinOut["a_isAggregatedCircuit"] = zkin1["isAggregatedCircuit"];

    // Add second recursive proof inputs
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
    for (uint64_t i = 1; i < steps; i++)
    {
        zkinOut["b_s" + std::to_string(i) + "_root"] = zkin2["s" + std::to_string(i) + "_root"];
        zkinOut["b_s" + std::to_string(i) + "_siblings"] = zkin2["s" + std::to_string(i) + "_siblings"];
        zkinOut["b_s" + std::to_string(i) + "_vals"] = zkin2["s" + std::to_string(i) + "_vals"];
    }
    zkinOut["b_finalPol"] = zkin2["finalPol"];
    zkinOut["b_isAggregatedCircuit"] = zkin2["isAggregatedCircuit"];

    // Add rootC Recursive2 to publics
    zkinOut["rootC"] = ordered_json::array();
    for (int i = 0; i < 4; i++)
    {
        zkinOut["rootC"][i] = to_string(verKey["constRoot"][i]);
    }

    return zkinOut;
}

ordered_json joinzkinBlobOuterRecursive2(ordered_json &zkin1, ordered_json &zkin2, ordered_json &verKey, uint64_t steps)
{
    ordered_json zkinOut = ordered_json::object();

    // Blob outer publics
    BlobOuterPublics blobOuterPublics;

    // Define output publics
    for (int i = 0; i < 8; i++)
    {
        zkinOut["publics"][blobOuterPublics.oldStateRootPos + i] = zkin1["publics"][blobOuterPublics.oldStateRootPos + i];
        zkinOut["publics"][blobOuterPublics.oldBlobStateRootPos + i] = zkin1["publics"][blobOuterPublics.oldBlobStateRootPos + i];
        zkinOut["publics"][blobOuterPublics.oldBlobAccInputHashPos + i] = zkin1["publics"][blobOuterPublics.oldBlobAccInputHashPos + i];

        zkinOut["publics"][blobOuterPublics.newStateRootPos + i] = zkin2["publics"][blobOuterPublics.newStateRootPos + i];
        zkinOut["publics"][blobOuterPublics.newBlobStateRootPos + i] = zkin2["publics"][blobOuterPublics.newBlobStateRootPos + i];
        zkinOut["publics"][blobOuterPublics.newBlobAccInputHashPos + i] = zkin2["publics"][blobOuterPublics.newBlobAccInputHashPos + i];

        zkinOut["publics"][blobOuterPublics.newLocalExitRootPos + i] = zkin2["publics"][blobOuterPublics.newLocalExitRootPos + i];
    }

    zkinOut["publics"][blobOuterPublics.oldBlobNumPos] = zkin1["publics"][blobOuterPublics.oldBlobNumPos];

    zkinOut["publics"][blobOuterPublics.chainIdPos] = zkin1["publics"][blobOuterPublics.chainIdPos];
    zkinOut["publics"][blobOuterPublics.forkIdPos] = zkin1["publics"][blobOuterPublics.forkIdPos];

    zkinOut["publics"][blobOuterPublics.newBlobNumPos] = zkin2["publics"][blobOuterPublics.newBlobNumPos];
    
    // Add first recursive proof inputs
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
    for (uint64_t i = 1; i < steps; i++)
    {
        zkinOut["a_s" + std::to_string(i) + "_root"] = zkin1["s" + std::to_string(i) + "_root"];
        zkinOut["a_s" + std::to_string(i) + "_siblings"] = zkin1["s" + std::to_string(i) + "_siblings"];
        zkinOut["a_s" + std::to_string(i) + "_vals"] = zkin1["s" + std::to_string(i) + "_vals"];
    }
    zkinOut["a_finalPol"] = zkin1["finalPol"];

    // Add second recursive proof inputs
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
    for (uint64_t i = 1; i < steps; i++)
    {
        zkinOut["b_s" + std::to_string(i) + "_root"] = zkin2["s" + std::to_string(i) + "_root"];
        zkinOut["b_s" + std::to_string(i) + "_siblings"] = zkin2["s" + std::to_string(i) + "_siblings"];
        zkinOut["b_s" + std::to_string(i) + "_vals"] = zkin2["s" + std::to_string(i) + "_vals"];
    }
    zkinOut["b_finalPol"] = zkin2["finalPol"];

    // Add rootC blobOuterRecursive2 to publics
    zkinOut["rootC"] = ordered_json::array();
    for (int i = 0; i < 4; i++)
    {
        zkinOut["rootC"][i] = to_string(verKey["constRoot"][i]);
    }

    return zkinOut;
}

ordered_json joinzkinBlobOuter(ordered_json &zkinBatch, ordered_json &zkinBlobInnerRecursive1, ordered_json &verKey, std::string chainId, uint64_t steps)
{
    ordered_json zkinOut = ordered_json::object();

    BatchPublics batchPublics;
    BlobInnerPublics blobInnerPublics;
    BlobOuterPublics blobOuterPublics;

    // Define output publics
    for (int i = 0; i < 8; i++)
    {
        zkinOut["publics"][blobOuterPublics.oldBlobStateRootPos + i] = zkinBlobInnerRecursive1["publics"][blobInnerPublics.oldBlobStateRootPos + i]; 
        zkinOut["publics"][blobOuterPublics.oldBlobAccInputHashPos + i] = zkinBlobInnerRecursive1["publics"][blobInnerPublics.oldBlobAccInputHashPos + i];

        zkinOut["publics"][blobOuterPublics.newBlobStateRootPos + i] = zkinBlobInnerRecursive1["publics"][blobInnerPublics.newBlobStateRootPos + i];
        zkinOut["publics"][blobOuterPublics.newBlobAccInputHashPos + i] = zkinBlobInnerRecursive1["publics"][blobInnerPublics.newBlobAccInputHashPos + i];
    }
    
    zkinOut["publics"][blobOuterPublics.oldBlobNumPos] = zkinBlobInnerRecursive1["publics"][blobInnerPublics.oldBlobNumPos];
    zkinOut["publics"][blobOuterPublics.newBlobNumPos] = zkinBlobInnerRecursive1["publics"][blobInnerPublics.newBlobNumPos];
    zkinOut["publics"][blobOuterPublics.forkIdPos] = zkinBatch["publics"][batchPublics.forkIdPos];

    bool isInvalidFinalAccBatchHashData = true;
    for(int i = 0; i < 8; i++) 
    {
        if(zkinBlobInnerRecursive1["publics"][blobInnerPublics.finalAccBatchHashDataPos + i] != to_string(0)) {
            isInvalidFinalAccBatchHashData = false;
            break;
        }
    }

    bool isInvalidBlob = zkinBlobInnerRecursive1["publics"][blobInnerPublics.isInvalidPos] == to_string(1);

    bool isInvalidTimestamp = zkinBatch["publics"][batchPublics.newLastTimestampPos] > zkinBlobInnerRecursive1["publics"][blobInnerPublics.timestampLimitPos];

    bool isInvalidL1InfoTreeIndex = zkinBatch["publics"][batchPublics.currentL1InfoTreeIndexPos] != zkinBlobInnerRecursive1["publics"][blobInnerPublics.lastL1InfoTreeIndexPos];
    
    bool isInvalid = isInvalidFinalAccBatchHashData || isInvalidBlob || isInvalidTimestamp || isInvalidL1InfoTreeIndex;

    for(int i = 0; i < 8; i++) 
    {
        if(isInvalid) {
            zkinOut["publics"][blobOuterPublics.oldStateRootPos + i] = zkinBlobInnerRecursive1["publics"][blobInnerPublics.oldStateRootPos + i];
            zkinOut["publics"][blobOuterPublics.newStateRootPos + i] = zkinBlobInnerRecursive1["publics"][blobInnerPublics.oldStateRootPos + i];
            zkinOut["publics"][blobOuterPublics.newLocalExitRootPos + i] = zkinBlobInnerRecursive1["publics"][blobInnerPublics.localExitRootFromBlobPos + i];
        } else {
            zkinOut["publics"][blobOuterPublics.oldStateRootPos + i] = zkinBatch["publics"][batchPublics.oldStateRootPos + i];
            zkinOut["publics"][blobOuterPublics.newStateRootPos + i] = zkinBatch["publics"][batchPublics.newStateRootPos + i];
            zkinOut["publics"][blobOuterPublics.newLocalExitRootPos + i] = zkinBatch["publics"][batchPublics.newLocalExitRootPos + i];
        }
    }

    if(isInvalid) {
        zkinOut["publics"][blobOuterPublics.chainIdPos] = chainId;
    } else {
        zkinOut["publics"][blobOuterPublics.chainIdPos] = zkinBatch["publics"][batchPublics.chainIdPos];
    }


    // Add batch proof inputs
    zkinOut["batch_publics"] = zkinBatch["publics"];
    zkinOut["batch_root1"] = zkinBatch["root1"];
    zkinOut["batch_root2"] = zkinBatch["root2"];
    zkinOut["batch_root3"] = zkinBatch["root3"];
    zkinOut["batch_root4"] = zkinBatch["root4"];
    zkinOut["batch_evals"] = zkinBatch["evals"];
    zkinOut["batch_s0_vals1"] = zkinBatch["s0_vals1"];
    zkinOut["batch_s0_vals3"] = zkinBatch["s0_vals3"];
    zkinOut["batch_s0_vals4"] = zkinBatch["s0_vals4"];
    zkinOut["batch_s0_valsC"] = zkinBatch["s0_valsC"];
    zkinOut["batch_s0_siblings1"] = zkinBatch["s0_siblings1"];
    zkinOut["batch_s0_siblings3"] = zkinBatch["s0_siblings3"];
    zkinOut["batch_s0_siblings4"] = zkinBatch["s0_siblings4"];
    zkinOut["batch_s0_siblingsC"] = zkinBatch["s0_siblingsC"];
    for (uint64_t i = 1; i < steps; i++)
    {
        zkinOut["batch_s" + std::to_string(i) + "_root"] = zkinBatch["s" + std::to_string(i) + "_root"];
        zkinOut["batch_s" + std::to_string(i) + "_siblings"] = zkinBatch["s" + std::to_string(i) + "_siblings"];
        zkinOut["batch_s" + std::to_string(i) + "_vals"] = zkinBatch["s" + std::to_string(i) + "_vals"];
    }
    zkinOut["batch_finalPol"] = zkinBatch["finalPol"];
    zkinOut["batch_isAggregatedCircuit"] = zkinBatch["isAggregatedCircuit"];

    // Add blob inner proof inputs
    zkinOut["blob_inner_publics"] = zkinBlobInnerRecursive1["publics"];
    zkinOut["blob_inner_root1"] = zkinBlobInnerRecursive1["root1"];
    zkinOut["blob_inner_root2"] = zkinBlobInnerRecursive1["root2"];
    zkinOut["blob_inner_root3"] = zkinBlobInnerRecursive1["root3"];
    zkinOut["blob_inner_root4"] = zkinBlobInnerRecursive1["root4"];
    zkinOut["blob_inner_evals"] = zkinBlobInnerRecursive1["evals"];
    zkinOut["blob_inner_s0_vals1"] = zkinBlobInnerRecursive1["s0_vals1"];
    zkinOut["blob_inner_s0_vals3"] = zkinBlobInnerRecursive1["s0_vals3"];
    zkinOut["blob_inner_s0_vals4"] = zkinBlobInnerRecursive1["s0_vals4"];
    zkinOut["blob_inner_s0_valsC"] = zkinBlobInnerRecursive1["s0_valsC"];
    zkinOut["blob_inner_s0_siblings1"] = zkinBlobInnerRecursive1["s0_siblings1"];
    zkinOut["blob_inner_s0_siblings3"] = zkinBlobInnerRecursive1["s0_siblings3"];
    zkinOut["blob_inner_s0_siblings4"] = zkinBlobInnerRecursive1["s0_siblings4"];
    zkinOut["blob_inner_s0_siblingsC"] = zkinBlobInnerRecursive1["s0_siblingsC"];
    for (uint64_t i = 1; i < steps; i++)
    {
        zkinOut["blob_inner_s" + std::to_string(i) + "_root"] = zkinBlobInnerRecursive1["s" + std::to_string(i) + "_root"];
        zkinOut["blob_inner_s" + std::to_string(i) + "_siblings"] = zkinBlobInnerRecursive1["s" + std::to_string(i) + "_siblings"];
        zkinOut["blob_inner_s" + std::to_string(i) + "_vals"] = zkinBlobInnerRecursive1["s" + std::to_string(i) + "_vals"];
    }
    zkinOut["blob_inner_finalPol"] = zkinBlobInnerRecursive1["finalPol"];

    // Add rootC blobOuter to publics
    zkinOut["rootC"] = ordered_json::array();
    for (int i = 0; i < 4; i++)
    {
        zkinOut["rootC"][i] = to_string(verKey["constRoot"][i]);
    }

    return zkinOut;
}