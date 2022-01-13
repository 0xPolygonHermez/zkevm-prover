#ifndef MERKLE_GROUP_MULTIPOL
#define MERKLE_GROUP_MULTIPOL

#include "poseidon_opt.hpp"
#include "ffiasm/fr.hpp"
#include "merkle.hpp"
#include <vector>
#include <cassert>
#include <math.h> /* floor */
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

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
    string fileName;
    uint64_t merkleGroupMultiPolTreeSize;
    uint8_t *merkleGroupMultiPolTreeMappedMemmory;

public:
    struct MerkleGroupMultiPolTree
    {
        vector<FrElement> mainTree;
        vector<vector<FrElement>> groupTrees;
        vector<vector<vector<FrElement>>> polTrees;
    };

    FrElement *mainTree;
    FrElement *groupTrees;
    FrElement *polTrees;

    uint32_t polTreesSize;
    uint32_t groupTreesSize;
    uint32_t mainTreeSize;
    uint32_t polsProofSize;
    uint32_t groupProofSize;
    uint32_t ngroupsProofSize;

    /*
        ===============================================
        PolTree[0][0]               -> PolsProofSize
        ===============================================
        PolTree[0][1]               -> PolsProofSize
        ===============================================
        ...
        ===============================================
        PolTree[0][groupSize]       -> PolsProofSize
        ===============================================
        groupTrees[0]               -> GroupProofSize
        ===============================================
        PolTree[1][0]               -> PolsProofSize
        ===============================================
        PolTree[1][1]               -> PolsProofSize
        ===============================================
        ...
        ===============================================
        PolTree[1][groupSize]       -> PolsProofSize
        ===============================================
        groupTrees[1]               -> GroupProofSize
        ===============================================
        ...
        ===============================================
        PolTree[ngroups][0]         -> PolsProofSize
        ===============================================
        PolTree[ngroups][1]         -> PolsProofSize
        ===============================================
        ...
        ===============================================
        PolTree[ngroups][groupSize] -> PolsProofSize
        ===============================================
        groupTrees[ngroups]         -> GroupProofSize
        ===============================================
        mainTree                    -> nGroupsProofSize
        ===============================================
    */
    FrElement *MerkleGroupMultiPolTreeArray;

    MerkleGroupMultiPol(Merkle *_M, uint32_t _nGroups, uint32_t _groupSize, uint32_t _nPols);

    RawFr::Element *merkelize(MerkleGroupMultiPolTree &tree, vector<vector<RawFr::Element>> pols);
    void merkelize(RawFr::Element *tree, vector<vector<RawFr::Element>> pols);

    RawFr::Element getElement(FrElement *tree, uint32_t polIdx, uint32_t idx);

    void getGroupProof(MerkleGroupMultiPolTree &tree, uint32_t idx, vector<vector<RawFr::Element>> &v, vector<FrElement> &mp);
    void getGroupProof(RawFr::Element *tree, uint32_t idx, RawFr::Element *groupProof);

    void getElementsProof(MerkleGroupMultiPolTree &tree, uint32_t idx, vector<RawFr::Element> &val, vector<vector<RawFr::Element>> &mp);
    void getElementsProof(uint32_t idx, RawFr::Element *val, uint32_t val_size, RawFr::Element *mp, uint32_t mp_size);

    RawFr::Element root(MerkleGroupMultiPolTree &tree);
    RawFr::Element root(RawFr::Element *mainTree);
    RawFr::Element root();

    FrElement calculateRootFromProof(vector<FrElement> &mp, uint32_t idx, vector<RawFr::Element> groupElements);

    RawFr::Element calculateRootFromElementProof(vector<vector<RawFr::Element>> &mp, uint32_t idx, vector<RawFr::Element> val);
    RawFr::Element calculateRootFromElementProof(RawFr::Element *mp, uint32_t mp_size, uint32_t idx, RawFr::Element *val, uint32_t val_size);

    RawFr::Element calculateRootFromGroupProof(vector<FrElement> &mp, uint32_t groupIdx, vector<vector<RawFr::Element>> groupElements);
    RawFr::Element calculateRootFromGroupProof(RawFr::Element *mp, uint32_t mp_size, uint32_t groupIdx, RawFr::Element *groupElements, uint32_t groupElements_size);

    bool verifyGroupProof(FrElement root, vector<FrElement> &mp, uint32_t idx, vector<vector<RawFr::Element>> groupElements);
    bool verifyGroupProof(FrElement root, RawFr::Element *mp, uint32_t mp_size, uint32_t idx, RawFr::Element *groupElements, uint32_t groupElements_size);

    bool verifyElementProof(RawFr::Element root, vector<vector<RawFr::Element>> &mp, uint32_t idx, vector<RawFr::Element> val);
    bool verifyElementProof(RawFr::Element root, RawFr::Element *mp, uint32_t mp_size, uint32_t idx, RawFr::Element *val, uint32_t val_size);

    static uint32_t getTreeSize(Merkle *M, uint32_t nGroups, uint32_t groupSize, uint32_t nPols);
    static uint32_t getGroupProofSize(Merkle *M, uint32_t nGroups, uint32_t groupSize, uint32_t nPols);
    static RawFr::Element *fileToMap(const string &file_name, FrElement *MerkleGroupMultiPolTree, Merkle *M, uint32_t nGroups, uint32_t groupSize, uint32_t nPols);
};
#endif // MERKLE_GROUP_MULTIPOL