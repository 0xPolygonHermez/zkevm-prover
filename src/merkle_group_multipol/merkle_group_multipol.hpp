#ifndef MERKLE_GROUP_MULTIPOL
#define MERKLE_GROUP_MULTIPOL

#include "poseidon_opt/poseidon_opt.hpp"
#include "ffiasm/fr.hpp"
#include "merkle/merkle.hpp"
#include <vector>
#include <cassert>
#include <math.h> /* floor */

using namespace std;

// This function is very useful for building fri.
// It a tree of a tree of a tree

// In each group we check all the evaluations of all the polinomials for a
// given root of unity of the next FRI level
// Each group is itself of a tree of the roots of unity
// And for each root of unity there is another tree with the field elements
// of each polynomial in that specific root of unity.

/*
root ---| group_{0}
        | .
        | group_{i} --------| idx_{0}
        |                   | .
        |                   | idx_{j} --- | polEv_{0}
        |                   |             | .
        |                   |             | polEv_{k}
        |                   |             | .
        |                   |             | polEv_{nPols-1}
        |                   | .
        |                   | idx_{groupSize-1}
        | .
        | group_{nGroups-1}

*/

class MerkleGroupMultiPol
{
    typedef RawFr::Element FrElement;

private:
    Poseidon_opt poseidon;
    RawFr field;
    Merkle *M;
    uint32_t nGroups, groupSize, nPols;

public:
    struct MerkleGroupMultiPolTree
    {
        vector<FrElement> mainTree;
        vector<vector<FrElement>> groupTrees;
        vector<vector<vector<FrElement>>> polTrees;
    };
    MerkleGroupMultiPol(Merkle *_M, uint32_t _nGroups, uint32_t _groupSize, uint32_t _nPols);

    void merkelize(MerkleGroupMultiPolTree &tree, vector<vector<RawFr::Element>> pols);
    
    void getGroupProof(MerkleGroupMultiPolTree &tree, uint32_t idx, vector<vector<RawFr::Element>> &v, vector<FrElement> &mp);
    void getElementsProof(MerkleGroupMultiPolTree &tree, uint32_t idx, vector<RawFr::Element> &val, vector<vector<RawFr::Element>> &mp);

    FrElement root(MerkleGroupMultiPolTree &tree);
    FrElement calculateRootFromProof(vector<FrElement> &mp, uint32_t idx, vector<RawFr::Element> groupElements);
    FrElement calculateRootFromGroupProof(vector<FrElement> &mp, uint32_t groupIdx, vector<vector<RawFr::Element>> groupElements);
    FrElement calculateRootFromElementProof(vector<vector<RawFr::Element>> &mp, uint32_t idx, vector<RawFr::Element> val);

    bool verifyGroupProof(FrElement root, vector<FrElement> &mp, uint32_t idx, vector<vector<RawFr::Element>> groupElements);
    bool verifyElementProof(RawFr::Element root, vector<vector<RawFr::Element>> &mp, uint32_t idx, vector<RawFr::Element> val);
};
#endif // MERKLE_GROUP_MULTIPOL