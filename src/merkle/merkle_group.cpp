#if 0
#include "merkle_group.hpp"
#include <iostream>
#include <cstring>
#include <assert.h> /* assert */

MerkleGroup::MerkleGroup(Merkle *_M, uint32_t _nGroups, uint32_t _groupSize)
{
    M = _M;
    nGroups = _nGroups;
    groupSize = _groupSize;
    mainTreeSize = M->numHashes(nGroups);
    groupProofSize = M->numHashes(groupSize);
    groupTreesSize = groupProofSize * nGroups;
    mainTreeProofSize = M->MerkleProofSize(groupTreesSize);
    merkleGroupTreeSize = mainTreeSize + groupTreesSize;

    mp_group_size = M->MerkleProofSize(groupSize) * M->arity;
    mp_main_size = M->MerkleProofSize(nGroups) * M->arity;
    elementProofSize = mp_group_size + mp_main_size;
}

void MerkleGroup::merkelize(RawFr::Element *tree, RawFr::Element *pols)
{
    // assert(nGroups * groupSize == pols.size()); TODO
    RawFr::Element groupRoots[mainTreeSize];
#pragma omp parallel for
    for (uint32_t i = 0; i < nGroups; i++)
    {
        RawFr::Element group[groupProofSize];
        for (uint32_t j = 0; j < groupSize; j++)
        {
            group[j] = pols[j * nGroups + i];
        }
        M->merkelize(group, groupSize);
        std::memcpy(&tree[mainTreeSize + i * groupProofSize], &group, groupProofSize * sizeof(RawFr::Element));
        groupRoots[i] = M->root(&tree[mainTreeSize + i * groupProofSize], groupProofSize);
    }
    M->merkelize(groupRoots, nGroups);
    std::memcpy(tree, &groupRoots, mainTreeSize * sizeof(RawFr::Element));
}
void MerkleGroup::merkelize(RawFr::Element *tree, vector<RawFr::Element> pols)
{
    assert(nGroups * groupSize == pols.size());

    RawFr::Element groupRoots[mainTreeSize];
#pragma omp parallel for
    for (uint32_t i = 0; i < nGroups; i++)
    {
        RawFr::Element group[groupProofSize];
        for (uint32_t j = 0; j < groupSize; j++)
        {
            group[j] = pols[j * nGroups + i];
        }

        M->merkelize(group, groupSize);
        std::memcpy(&tree[mainTreeSize + i * groupProofSize], &group, groupProofSize * sizeof(RawFr::Element));

        groupRoots[i] = M->root(&tree[mainTreeSize + i * groupProofSize], groupProofSize);
    }
    M->merkelize(groupRoots, nGroups);
    std::memcpy(tree, &groupRoots, mainTreeSize * sizeof(RawFr::Element));
}

RawFr::Element MerkleGroup::root(RawFr::Element *tree)
{
    return M->root(tree, M->numHashes(nGroups));
}

void MerkleGroup::getGroupProof(RawFr::Element *tree, uint32_t idx, RawFr::Element *groupProof)
{
    uint32_t merkleProofSize = M->MerkleProofSize(nGroups);
    uint32_t mp_size_array = merkleProofSize * M->arity;

    uint32_t tree_cursor = mainTreeSize + idx * groupProofSize;

    std::memcpy(groupProof, &(tree[tree_cursor]), groupSize * sizeof(RawFr::Element));

    M->genMerkleProof(tree, mainTreeSize, idx, 0, &groupProof[groupSize], mp_size_array);
}

RawFr::Element MerkleGroup::calculateRootFromGroupProof(RawFr::Element *mp, uint32_t mp_size, uint32_t groupIdx, RawFr::Element *groupElements, uint32_t groupElements_size)
{
    RawFr::Element tree[groupProofSize];
    std::memcpy(&tree, groupElements, groupSize * sizeof(RawFr::Element));

    M->merkelize(tree, groupSize);

    for (uint32_t i = 0; i < groupProofSize; i++)
    {
        printf("%s\n", field.toString(tree[i], 16).c_str());
    }

    RawFr::Element rootG = M->root(tree, groupProofSize);
    RawFr::Element root = M->calculateRootFromProof(mp, mp_size, groupIdx, rootG, 0);

    return root;
}

bool MerkleGroup::verifyGroupProof(RawFr::Element root, RawFr::Element *mp, uint32_t mp_size, uint32_t idx, RawFr::Element *groupElements, uint32_t groupElements_size)
{
    RawFr::Element rootC = calculateRootFromGroupProof(mp, mp_size, idx, groupElements, groupElements_size);
    return field.eq(root, rootC);
}
void MerkleGroup::getElementsProof(RawFr::Element *tree, uint32_t idx, RawFr::Element *elementsProof)
{
    uint32_t group = idx % nGroups;
    uint32_t groupIdx = floor((float)idx / (float)nGroups);
    std::memcpy(elementsProof, &tree[mainTreeSize + group * groupProofSize + groupIdx], sizeof(RawFr::Element));
    M->genMerkleProof(&tree[mainTreeSize + group * groupProofSize], groupProofSize, groupIdx, 0, &elementsProof[1], mp_group_size);
    M->genMerkleProof(tree, mainTreeSize, group, 0, &elementsProof[mp_group_size + 1], mp_main_size);
}

RawFr::Element MerkleGroup::calculateRootFromElementProof(RawFr::Element *mp, uint32_t mp_size, uint32_t idx, RawFr::Element *val, uint32_t val_size)
{
    uint32_t group = idx % nGroups;
    uint32_t groupIdx = floor((float)idx / (float)nGroups);

    uint32_t mp_group_size = M->MerkleProofSize(groupSize) * M->arity;
    uint32_t mp_main_size = M->MerkleProofSize(mainTreeProofSize) * M->arity;

    printf("val: %s\n", field.toString(val[0], 16).c_str());

    // Merkle::FrElement Merkle::calculateRootFromProof(FrElement *mp, uint32_t size, uint32_t idx, FrElement value, uint32_t offset)
    RawFr::Element rootGroup = M->calculateRootFromProof(mp, mp_group_size, groupIdx, val[0], 0);
    printf("rootGroup: %s\n", field.toString(rootGroup, 16).c_str());

    RawFr::Element root = M->calculateRootFromProof(&mp[mp_group_size], mp_main_size, group, rootGroup, 0);
    return root;
}

bool MerkleGroup::verifyElementProof(RawFr::Element root, RawFr::Element *mp, uint32_t mp_size, uint32_t idx, RawFr::Element *val, uint32_t val_size)
{
    RawFr::Element rootC = calculateRootFromElementProof(mp, mp_size, idx, val, val_size);
    return field.eq(root, rootC);
}

uint32_t MerkleGroup::getTreeMemSize(Merkle *M, uint32_t nGroups, uint32_t groupSize)
{
    uint32_t mainTreeSize = M->numHashes(nGroups);
    uint32_t groupProofSize = M->numHashes(groupSize);
    uint32_t groupTreesSize = groupProofSize * nGroups;
    return (mainTreeSize + groupTreesSize) * sizeof(RawFr::Element);
}
void MerkleGroup::getElementProofSize(Merkle *M, uint32_t nGroups, uint32_t groupSize, uint64_t &memSize, uint64_t &memSizeValue, uint64_t &memSizeMpL, uint64_t &memSizeMpH)
{
    uint32_t mp_group_size = M->MerkleProofSize(groupSize) * M->arity;
    uint32_t mp_main_size = M->MerkleProofSize(nGroups) * M->arity;

    memSizeValue = 1;
    memSizeMpL = mp_group_size;
    memSizeMpH = mp_main_size;
    memSize = (memSizeValue + memSizeMpL + memSizeMpH) * sizeof(RawFr::Element);
}

void MerkleGroup::getGroupProofSize(Merkle *M, uint32_t nGroups, uint32_t groupSize, uint64_t &memSize, uint64_t &memSizeValue, uint64_t &memSizeMp)
{
    memSizeValue = groupSize;
    memSizeMp = M->MerkleProofSize(nGroups) * M->arity;
    memSize = (memSizeValue + memSizeMp) * sizeof(RawFr::Element);
}
#endif