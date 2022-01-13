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

    uint32_t mp_group_size = M->MerkleProofSize(groupSize) * M->arity;
    uint32_t mp_main_size = M->MerkleProofSize(nGroups) * M->arity;
    elementProofSize = mp_group_size + mp_main_size;
}

void MerkleGroup::merkelize(RawFr::Element *tree, vector<RawFr::Element> pols)
{
    assert(nGroups * groupSize == pols.size());

    RawFr::Element groupRoots[mainTreeSize];
    //#pragma omp parallel for
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

    for (int j = 0; j < merkleGroupTreeSize; j++)
    {
        printf("%s\n", field.toString(tree[j], 16).c_str());
    }
    printf("######\n");
}

RawFr::Element MerkleGroup::root(RawFr::Element *tree)
{
    return M->root(tree, M->numHashes(nGroups));
}

void MerkleGroup::getGroupProof(RawFr::Element *tree, uint32_t idx, RawFr::Element *groupElements, uint32_t groupElements_size, RawFr::Element *mp, uint32_t mp_size)
{
    assert(idx < groupSize);
    for (uint32_t i = 0; i < groupSize; i++)
    {
        uint32_t cursor = i * groupSize;
        uint32_t tree_cursor = mainTreeSize + idx * groupProofSize + i * groupProofSize;

        std::memcpy(&(groupElements[cursor]), &(tree[tree_cursor]), groupSize * sizeof(RawFr::Element));
    };
    /*
    for (int j = 0; j < groupElements_size; j++)
    {
        printf("%s\n", field.toString(groupElements[j], 16).c_str());
    }
    */

    M->genMerkleProof(tree, mainTreeSize, idx, 0, mp, mp_size);
}

RawFr::Element MerkleGroup::calculateRootFromGroupProof(RawFr::Element *mp, uint32_t mp_size, uint32_t groupIdx, RawFr::Element *groupElements, uint32_t groupElements_size)
{
    RawFr::Element tree[groupProofSize];
    std::memcpy(&tree, groupElements, groupSize * sizeof(RawFr::Element));

    M->merkelize(tree, groupSize);

    for (int32_t i = 0; i < groupProofSize; i++)
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

void MerkleGroup::getElementsProof(RawFr::Element *tree, uint32_t idx, RawFr::Element *val, uint32_t val_size, RawFr::Element *mp, uint32_t mp_size)
{
    uint32_t group = idx % nGroups;
    uint32_t groupIdx = floor((float)idx / (float)nGroups);

    std::memcpy(val, &tree[mainTreeSize + group * groupProofSize + groupIdx], sizeof(RawFr::Element));
    for (uint32_t k = 0; k < groupTreesSize; k++)
    {
        printf("tree: %s\n", field.toString(tree[mainTreeSize + k], 16).c_str());
    }
    printf("val: %s\n", field.toString(*val, 16).c_str());

    uint32_t mp_group_size = M->MerkleProofSize(groupSize) * M->arity;
    uint32_t mp_main_size = M->MerkleProofSize(mainTreeProofSize) * M->arity;

    M->genMerkleProof(&tree[mainTreeSize + group * groupProofSize], groupProofSize, groupIdx, 0, mp, mp_group_size);

    printf("#########\n");

    for (uint32_t k = 0; k < mp_group_size; k++)
    {
        printf("%s\n", field.toString(mp[k], 16).c_str());
    }
    printf("#########\n");

    M->genMerkleProof(tree, mainTreeSize, group, 0, &mp[mp_group_size], mp_main_size);

    for (uint32_t k = 0; k < mp_main_size; k++)
    {
        printf("%s\n", field.toString(mp[mp_group_size + k], 16).c_str());
    }
    printf("#########\n");
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

uint32_t MerkleGroup::getTreeSize(Merkle *M, uint32_t nGroups, uint32_t groupSize)
{
    uint32_t mainTreeSize = M->numHashes(nGroups);
    uint32_t groupProofSize = M->numHashes(groupSize);
    uint32_t groupTreesSize = groupProofSize * nGroups;
    return mainTreeSize + groupTreesSize;
}
uint32_t MerkleGroup::getElementProofSize(Merkle *M, uint32_t nGroups, uint32_t groupSize)
{
    uint32_t mp_group_size = M->MerkleProofSize(groupSize) * M->arity;
    uint32_t mp_main_size = M->MerkleProofSize(nGroups) * M->arity;
    return mp_group_size + mp_main_size;
}

uint32_t MerkleGroup::getGroupProofSize(Merkle *M, uint32_t nGroups, uint32_t groupSize)
{
    uint32_t merkleProofSize = M->MerkleProofSize(MerkleGroup::getTreeSize(M, nGroups, groupSize));
    uint32_t mp_size_array = merkleProofSize * M->arity;
    return mp_size_array;
}
