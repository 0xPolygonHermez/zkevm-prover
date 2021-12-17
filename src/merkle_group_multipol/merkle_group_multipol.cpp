#include "merkle_group_multipol/merkle_group_multipol.hpp"

MerkleGroupMultiPol::MerkleGroupMultiPol(Merkle *_M, uint32_t _nGroups, uint32_t _groupSize, uint32_t _nPols)
{
    M = _M;
    nGroups = _nGroups;
    groupSize = _groupSize;
    nPols = _nPols;
}

void MerkleGroupMultiPol::merkelize(MerkleGroupMultiPolTree &tree, vector<vector<RawFr::Element>> pols)
{
    assert(pols.size() == nPols);

    for (uint32_t i = 0; i < pols.size(); i++)
    {
        assert(nGroups * groupSize == pols[i].size());
    }
    tree.groupTrees = vector<vector<RawFr::Element>>(nGroups);
    tree.polTrees = vector<vector<vector<RawFr::Element>>>(nGroups);

    vector<RawFr::Element> groupRoots(nGroups);
#pragma omp parallel for
    for (uint32_t i = 0; i < nGroups; i++)
    {
        tree.groupTrees[i] = vector<RawFr::Element>(groupSize);
        tree.polTrees[i] = vector<vector<RawFr::Element>>(groupSize);

        vector<RawFr::Element> polRoots(groupSize);
        for (uint32_t j = 0; j < groupSize; j++)
        {
            tree.polTrees[i][j] = vector<RawFr::Element>(groupSize);
            vector<RawFr::Element> elements = vector<RawFr::Element>(nPols);
            for (uint32_t k = 0; k < nPols; k++)
            {
                elements[k] = pols[k][j * nGroups + i];
            }

            M->merkelize(elements);
            tree.polTrees[i][j] = elements;
            polRoots[j] = M->root(tree.polTrees[i][j]);
        }
        M->merkelize(polRoots);
        tree.groupTrees[i] = polRoots;

        groupRoots[i] = M->root(tree.groupTrees[i]);
    }
    M->merkelize(groupRoots);
    tree.mainTree = groupRoots;
}

void MerkleGroupMultiPol::getGroupProof(MerkleGroupMultiPolTree &tree, uint32_t idx, vector<vector<FrElement>> &v, vector<FrElement> &mp)
{
#pragma omp parallel for
    for (uint32_t j = 0; j < groupSize; j++)
    {
        v[j].insert(v[j].begin(), tree.polTrees[idx][j].begin(), tree.polTrees[idx][j].begin() + nPols);
    }
    mp = M->genMerkleProof(tree.mainTree, idx, 0);
}

MerkleGroupMultiPol::FrElement MerkleGroupMultiPol::root(MerkleGroupMultiPolTree &tree)
{
    return M->root(tree.mainTree);
}

bool MerkleGroupMultiPol::verifyGroupProof(FrElement root, vector<FrElement> &mp, uint32_t idx, vector<vector<RawFr::Element>> groupElements)
{
    FrElement rootC = calculateRootFromGroupProof(mp, idx, groupElements);

    return field.eq(root, rootC);
}
MerkleGroupMultiPol::FrElement MerkleGroupMultiPol::calculateRootFromGroupProof(vector<FrElement> &mp, uint32_t groupIdx, vector<vector<RawFr::Element>> groupElements)
{
    vector<RawFr::Element> polRoots(groupSize);

#pragma omp parallel for
    for (uint32_t j = 0; j < groupSize; j++)
    {
        vector<RawFr::Element> polTree(groupElements[j].size());
        polTree.insert(polTree.begin(), groupElements[j].begin(), groupElements[j].end());
        M->merkelize(polTree);
        polRoots[j] = M->root(polTree);
    }

    M->merkelize(polRoots);
    FrElement rootGroup = M->root(polRoots);
    FrElement rootMain = M->calculateRootFromProof(mp, groupIdx, rootGroup, 0);

    return rootMain;
}

void MerkleGroupMultiPol::getElementsProof(MerkleGroupMultiPolTree &tree, uint32_t idx, vector<RawFr::Element> &val, vector<vector<RawFr::Element>> &mp)
{

    uint32_t group = idx % nGroups;
    uint32_t groupIdx = floor((float)idx / (float)nGroups);

    val.insert(val.begin(), tree.polTrees[group][groupIdx].begin(), tree.polTrees[group][groupIdx].begin() + nPols);
    mp.push_back(M->genMerkleProof(tree.groupTrees[group], groupIdx, 0));
    mp.push_back(M->genMerkleProof(tree.mainTree, group, 0));
}

bool MerkleGroupMultiPol::verifyElementProof(RawFr::Element root, vector<vector<RawFr::Element>> &mp, uint32_t idx, vector<RawFr::Element> val)
{
    FrElement rootC = calculateRootFromElementProof(mp, idx, val);
    return field.eq(root, rootC);
}

MerkleGroupMultiPol::FrElement MerkleGroupMultiPol::calculateRootFromElementProof(vector<vector<RawFr::Element>> &mp, uint32_t idx, vector<RawFr::Element> polTree)
{

    uint32_t group = idx % nGroups;
    uint32_t groupIdx = floor((float)idx / (float)nGroups);

    M->merkelize(polTree);

    FrElement rootPol = M->root(polTree);
    FrElement rootGroup = M->calculateRootFromProof(mp[0], groupIdx, rootPol, 0);
    FrElement rootMain = M->calculateRootFromProof(mp[1], group, rootGroup, 0);
    return rootMain;
}
