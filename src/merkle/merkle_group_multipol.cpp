#include "merkle_group_multipol.hpp"
#include <iostream>
#include <cstring>
#include <assert.h> /* assert */

MerkleGroupMultiPol::MerkleGroupMultiPol(Merkle *_M, uint32_t _nGroups, uint32_t _groupSize, uint32_t _nPols)
{
    M = _M;
    nGroups = _nGroups;
    groupSize = _groupSize;
    nPols = _nPols;

    polsProofSize = M->numHashes(nPols);
    groupProofSize = M->numHashes(groupSize);
    ngroupsProofSize = M->numHashes(nGroups);

    polTreesSize = nGroups * groupSize * polsProofSize * sizeof(RawFr::Element);
    groupTreesSize = nGroups * groupProofSize * sizeof(RawFr::Element);
    mainTreeSize = ngroupsProofSize * sizeof(RawFr::Element);

    mainTree = (FrElement *)malloc(mainTreeSize);
    groupTrees = (FrElement *)malloc(groupTreesSize);
    polTrees = (FrElement *)malloc(polTreesSize);

    MerkleGroupMultiPolTreeArray = (FrElement *)malloc(mainTreeSize + groupTreesSize + polTreesSize);
}

void MerkleGroupMultiPol::merkelize(RawFr::Element *tree, vector<vector<RawFr::Element>> pols)
{
    assert(pols.size() == nPols);
    uint32_t polsProofSize = M->numHashes(nPols);
    uint32_t groupProofSize = M->numHashes(groupSize);
    uint32_t ngroupsProofSize = M->numHashes(nGroups);
    for (uint32_t i = 0; i < pols.size(); i++)
    {
        assert(nGroups * groupSize == pols[i].size());
    }
    RawFr::Element groupRoots[ngroupsProofSize];
#pragma omp parallel for
    for (uint32_t i = 0; i < nGroups; i++)
    {
        RawFr::Element polRoots[groupProofSize];
        for (uint32_t j = 0; j < groupSize; j++)
        {
            RawFr::Element elements[polsProofSize];
            for (uint32_t k = 0; k < nPols; k++)
            {
                elements[k] = pols[k][j * nGroups + i];
            }

            M->merkelize(elements, nPols);

            uint64_t block = (groupProofSize + groupSize * polsProofSize);
            RawFr::Element *cur = &tree[i * block + j * polsProofSize];

            std::memcpy(cur, &elements, polsProofSize * sizeof(RawFr::Element));

            polRoots[j] = M->root((RawFr::Element *)cur, polsProofSize);
        }
        M->merkelize(polRoots, groupSize);

        void *cur = &tree[i * (groupProofSize + groupSize * polsProofSize) + groupSize * polsProofSize];
        std::memcpy(cur, &polRoots, groupProofSize * sizeof(RawFr::Element));
        groupRoots[i] = M->root((RawFr::Element *)cur, groupProofSize);
    }
    M->merkelize(groupRoots, nGroups);

    RawFr::Element *cur = &tree[nGroups * (groupProofSize + groupSize * polsProofSize)];
    std::memcpy(cur, &groupRoots, ngroupsProofSize * sizeof(RawFr::Element));
}

void MerkleGroupMultiPol::getGroupProof(RawFr::Element *tree, uint32_t idx, RawFr::Element *groupProof)
{
    uint32_t polsBatch = groupSize * polsProofSize;
    uint32_t groupSizeBatch = groupProofSize + polsBatch;
    uint32_t cursor = idx * groupSizeBatch;
    uint32_t mp_size_array = (M->MerkleProofSize(ngroupsProofSize) - 1) * M->arity;

    for (uint32_t j = 0; j < groupSize; j++)
    {
        std::memcpy(&(groupProof[j * nPols]), &(tree[cursor + j * polsProofSize]), nPols * sizeof(RawFr::Element));
    }

    M->genMerkleProof(&(tree[nGroups * (groupProofSize + groupSize * polsProofSize)]), ngroupsProofSize, idx, 0, &groupProof[groupSize * nPols], mp_size_array);
}

MerkleGroupMultiPol::FrElement MerkleGroupMultiPol::root(MerkleGroupMultiPolTree &tree)
{
    return M->root(tree.mainTree);
}

RawFr::Element MerkleGroupMultiPol::root(RawFr::Element *mainTree)
{
    return M->root(mainTree, M->numHashes(nGroups));
}

RawFr::Element MerkleGroupMultiPol::root()
{
    return M->root(&MerkleGroupMultiPolTreeArray[nGroups * (groupProofSize + groupSize * polsProofSize)], M->numHashes(nGroups));
}

bool MerkleGroupMultiPol::verifyGroupProof(FrElement root, RawFr::Element *mp, uint32_t mp_size, uint32_t idx, RawFr::Element *groupElements, uint32_t groupElements_size)
{
    FrElement rootC = calculateRootFromGroupProof(mp, mp_size, idx, groupElements, groupElements_size);

    return field.eq(root, rootC);
}

RawFr::Element MerkleGroupMultiPol::calculateRootFromGroupProof(RawFr::Element *mp, uint32_t mp_size, uint32_t groupIdx, RawFr::Element *groupElements, uint32_t groupElements_size)
{

    RawFr::Element polRoots[groupProofSize];
#pragma omp parallel for
    for (uint32_t j = 0; j < groupSize; j++)
    {
        RawFr::Element polTree[polsProofSize];
        std::memcpy(&polTree, &groupElements[j * nPols], nPols * sizeof(RawFr::Element));
        M->merkelize(polTree, nPols);
        polRoots[j] = M->root(polTree, polsProofSize);
    }

    M->merkelize(polRoots, groupSize);

    FrElement rootGroup = M->root(polRoots, groupProofSize);
    FrElement rootMain = M->calculateRootFromProof(mp, mp_size, groupIdx, rootGroup, 0);

    return rootMain;
}

void MerkleGroupMultiPol::getElementsProof(uint32_t idx, RawFr::Element *val, uint32_t val_size, RawFr::Element *mp, uint32_t mp_size)
{
    uint32_t group = idx % nGroups;
    uint32_t groupIdx = floor((float)idx / (float)nGroups);

    std::memcpy(val, &polTrees[group * groupSize * polsProofSize + groupIdx * polsProofSize], nPols * sizeof(RawFr::Element));

    uint32_t mp_group_size = M->MerkleProofSize(groupSize) * M->arity;
    uint32_t mp_main_size = M->MerkleProofSize(nGroups) * M->arity;

    M->genMerkleProof(&groupTrees[group * groupProofSize], groupProofSize, groupIdx, 0, mp, mp_group_size);
    M->genMerkleProof(mainTree, ngroupsProofSize, group, 0, &mp[mp_group_size], mp_main_size);
}

bool MerkleGroupMultiPol::verifyElementProof(RawFr::Element root, RawFr::Element *mp, uint32_t mp_size, uint32_t idx, RawFr::Element *val, uint32_t val_size)
{
    FrElement rootC = calculateRootFromElementProof(mp, mp_size, idx, val, val_size);
    return field.eq(root, rootC);
}

RawFr::Element MerkleGroupMultiPol::calculateRootFromElementProof(RawFr::Element *mp, uint32_t mp_size, uint32_t idx, RawFr::Element *val, uint32_t val_size)
{
    uint32_t group = idx % nGroups;
    uint32_t groupIdx = floor((float)idx / (float)nGroups);

    uint32_t valsMerkle_size = M->numHashes(val_size);
    RawFr::Element valsMerkle[valsMerkle_size] = {field.zero()};
    std::memcpy(&valsMerkle, val, val_size * sizeof(RawFr::Element));

    M->merkelize(valsMerkle, val_size);
    FrElement rootPol = M->root(valsMerkle, valsMerkle_size);

    uint32_t mp_group_size = M->MerkleProofSize(groupSize) * M->arity;
    uint32_t mp_main_size = M->MerkleProofSize(nGroups) * M->arity;

    FrElement rootGroup = M->calculateRootFromProof(mp, mp_group_size, groupIdx, rootPol, 0);
    FrElement rootMain = M->calculateRootFromProof(&mp[mp_group_size], mp_main_size, group, rootGroup, 0);

    return rootMain;
}

RawFr::Element *MerkleGroupMultiPol::fileToMap(const string &fileName, FrElement *MerkleGroupMultiPolTree, Merkle *M, uint32_t nGroups, uint32_t groupSize, uint32_t nPols)
{

    uint32_t merkleGroupMultiPolTreeSize = getTreeMemSize(M, nGroups, groupSize, nPols);
    cout << "MerkleGroupMultiPol::mapToFile() calculated total size=" << merkleGroupMultiPolTreeSize << endl;

    //  Check the file size is the same as the expected
    struct stat sb;
    if (lstat(fileName.c_str(), &sb) == -1)
    {
        cerr << "Error: MerkleGroupMultiPol::mapToFile() failed calling lstat() of file " << fileName << endl;
        exit(-1);
    }
    if ((uint64_t)sb.st_size != merkleGroupMultiPolTreeSize )
    {
        cerr << "Error: MerkleGroupMultiPol::mapToFile() found size of file " << fileName << " to be " << sb.st_size << " B instead of " << merkleGroupMultiPolTreeSize << " B" << endl;
        exit(-1);
    }
    int oflags = O_RDWR;
    int fd = open(fileName.c_str(), oflags, 0666);
    if (fd < 0)
    {
        cerr << "Error: MerkleGroupMultiPol::mapToFile() failed opening input file: " << fileName << endl;
        exit(-1);
    }
    FrElement *merkleGroupMultiPolTree_mmap = (FrElement *)mmap(NULL, merkleGroupMultiPolTreeSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (merkleGroupMultiPolTree_mmap == MAP_FAILED)
    {
        cerr << "Error: MerkleGroupMultiPol::mapToFile() failed calling mmap() of file: " << fileName << endl;
        exit(-1);
    }
    close(fd);

    return merkleGroupMultiPolTree_mmap;
}

uint32_t MerkleGroupMultiPol::getTreeMemSize(Merkle *M, uint32_t nGroups, uint32_t groupSize, uint32_t nPols)
{
    uint32_t polsProofSize = M->numHashes(nPols);
    uint32_t groupProofSize = M->numHashes(groupSize);
    uint32_t ngroupsProofSize = M->numHashes(nGroups);

    uint32_t polTreesSize = nGroups * groupSize * polsProofSize;
    uint32_t groupTreesSize = nGroups * groupProofSize;
    uint32_t mainTreeSize = ngroupsProofSize;

    return (mainTreeSize + groupTreesSize + polTreesSize) * sizeof(RawFr::Element);
}

void MerkleGroupMultiPol::getGroupProofSize(Merkle *M, uint32_t nGroups, uint32_t groupSize, uint32_t nPols, uint64_t &memSize, uint64_t &memSizeValue, uint64_t &memSizeMp)
{
    uint32_t ngroupsProofSize = M->numHashes(nGroups);

    memSizeValue = nPols * groupSize;
    memSizeMp = (M->MerkleProofSize(ngroupsProofSize) - 1) * M->arity;
    memSize = (memSizeValue + memSizeMp) * sizeof(RawFr::Element);
}

// idx is the root of unity
RawFr::Element MerkleGroupMultiPol::getElement(FrElement *tree, uint32_t polIdx, uint32_t idx)
{
    uint32_t group = idx % nGroups;
    uint32_t groupIdx = floor((float)idx / (float)nGroups);

    return M->getElement(&tree[group * (groupProofSize + groupSize * polsProofSize) + groupIdx * polsProofSize], polIdx);
}