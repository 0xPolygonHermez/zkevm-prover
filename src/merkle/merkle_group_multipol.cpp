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
            void *cur = &tree[i * block + j * polsProofSize];

            std::memcpy(cur, &elements, polsProofSize * sizeof(RawFr::Element));

            polRoots[j] = M->root(&polTrees[i * groupSize * polsProofSize + j * polsProofSize], polsProofSize);
        }
        M->merkelize(polRoots, groupSize);
        void *cur = &tree[i * (groupProofSize + groupSize * polsProofSize) + groupSize * polsProofSize];
        std::memcpy(cur, &polRoots, groupProofSize * sizeof(RawFr::Element));
        groupRoots[i] = M->root(&groupTrees[i * groupProofSize], groupProofSize);
    }
    M->merkelize(groupRoots, nGroups);

    void *cur = &tree[nGroups * (groupProofSize + groupSize * polsProofSize)];

    std::memcpy(cur, &groupRoots, ngroupsProofSize * sizeof(RawFr::Element));
}
RawFr::Element *MerkleGroupMultiPol::merkelize(MerkleGroupMultiPolTree &tree, vector<vector<RawFr::Element>> pols)
{
    assert(pols.size() == nPols);

    uint32_t polsProofSize = M->numHashes(nPols);
    uint32_t groupProofSize = M->numHashes(groupSize);
    uint32_t ngroupsProofSize = M->numHashes(nGroups);

    for (uint32_t i = 0; i < pols.size(); i++)
    {
        assert(nGroups * groupSize == pols[i].size());
    }
    tree.groupTrees = vector<vector<RawFr::Element>>(nGroups);
    tree.polTrees = vector<vector<vector<RawFr::Element>>>(nGroups);

    RawFr::Element groupRoots[ngroupsProofSize];

#pragma omp parallel for
    for (uint32_t i = 0; i < nGroups; i++)
    {
        tree.groupTrees[i] = vector<RawFr::Element>(groupSize);
        tree.polTrees[i] = vector<vector<RawFr::Element>>(groupSize);

        RawFr::Element polRoots[groupProofSize];
        for (uint32_t j = 0; j < groupSize; j++)
        {
            tree.polTrees[i][j] = vector<RawFr::Element>(groupSize);

            RawFr::Element elements[polsProofSize];

            for (uint32_t k = 0; k < nPols; k++)
            {
                elements[k] = pols[k][j * nGroups + i];
            }

            M->merkelize(elements, nPols);

            void *cursor = &polTrees[i * groupSize * polsProofSize + j * polsProofSize];
            uint64_t block = (groupProofSize + groupSize * polsProofSize);
            void *cur = &MerkleGroupMultiPolTreeArray[i * block + j * polsProofSize];

            std::memcpy(cursor, &elements, polsProofSize * sizeof(RawFr::Element));
            std::memcpy(cur, &elements, polsProofSize * sizeof(RawFr::Element));

            // To be deleted
            vector<RawFr::Element> tmp(elements, elements + polsProofSize);
            tree.polTrees[i][j] = tmp;

            polRoots[j] = M->root(&polTrees[i * groupSize * polsProofSize + j * polsProofSize], polsProofSize);
        }
        M->merkelize(polRoots, groupSize);
        void *cursor = &groupTrees[i * groupProofSize];
        void *cur = &MerkleGroupMultiPolTreeArray[i * (groupProofSize + groupSize * polsProofSize) + groupSize * polsProofSize];

        std::memcpy(cursor, &polRoots, groupProofSize * sizeof(RawFr::Element));
        std::memcpy(cur, &polRoots, groupProofSize * sizeof(RawFr::Element));

        // To be deleted
        vector<RawFr::Element> tmp(polRoots, polRoots + groupProofSize);
        tree.groupTrees[i] = tmp;

        groupRoots[i] = M->root(&groupTrees[i * groupProofSize], groupProofSize);
    }
    M->merkelize(groupRoots, nGroups);

    void *cur = &MerkleGroupMultiPolTreeArray[nGroups * (groupProofSize + groupSize * polsProofSize)];

    std::memcpy(mainTree, &groupRoots, ngroupsProofSize * sizeof(RawFr::Element));
    std::memcpy(cur, &groupRoots, ngroupsProofSize * sizeof(RawFr::Element));

    // To be deleted
    vector<RawFr::Element> tmp(groupRoots, groupRoots + ngroupsProofSize);
    tree.mainTree = tmp;

    /*
        for (uint32_t k = 0; k < tree.mainTree.size(); k++)
        {
            //printf("%s\n", field.toString(tree.mainTree[k], 16).c_str());
            printf("%s\n", field.toString(mainTree[k], 16).c_str());
        }*/
    return (RawFr::Element *)cur;
}

void MerkleGroupMultiPol::getGroupProof(MerkleGroupMultiPolTree &tree, uint32_t idx, vector<vector<FrElement>> &v, vector<FrElement> &mp)
{
#pragma omp parallel for
    for (uint32_t j = 0; j < groupSize; j++)
    {
        v[j].insert(v[j].begin(), tree.polTrees[idx][j].begin(), tree.polTrees[idx][j].begin() + nPols);
    }

    uint32_t size = tree.mainTree.size();
    mp = M->genMerkleProof(tree.mainTree, idx, 0);
}

void MerkleGroupMultiPol::getGroupProof(uint32_t idx, RawFr::Element *groupElementsArray, uint32_t v_size, RawFr::Element *mp, uint32_t mp_size)
{

    uint32_t polsProofSize = M->numHashes(nPols);
    uint32_t groupProofSize = M->numHashes(groupSize);
    uint32_t ngroupsProofSize = M->numHashes(nGroups);

    uint32_t polsBatch = groupSize * polsProofSize;
    uint32_t groupSizeBatch = groupProofSize + polsBatch;
    uint32_t cursor = idx * groupSizeBatch;

    for (uint32_t j = 0; j < groupSize; j++)
    {
        std::memcpy(&(groupElementsArray[j * nPols]), &(MerkleGroupMultiPolTreeArray[cursor + j * polsProofSize]), nPols * sizeof(RawFr::Element));
    }

    M->genMerkleProof(&(MerkleGroupMultiPolTreeArray[nGroups * (groupProofSize + groupSize * polsProofSize)]), ngroupsProofSize, idx, 0, mp, mp_size);
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

bool MerkleGroupMultiPol::verifyGroupProof(FrElement root, vector<FrElement> &mp, uint32_t idx, vector<vector<RawFr::Element>> groupElements)
{
    FrElement rootC = calculateRootFromGroupProof(mp, idx, groupElements);

    return field.eq(root, rootC);
}

bool MerkleGroupMultiPol::verifyGroupProof(FrElement root, RawFr::Element *mp, uint32_t mp_size, uint32_t idx, RawFr::Element *groupElements, uint32_t groupElements_size)
{
    FrElement rootC = calculateRootFromGroupProof(mp, mp_size, idx, groupElements, groupElements_size);

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
    printf("rootGroup: %s\n", field.toString(rootGroup, 16).c_str());
    printf("rootMain: %s\n", field.toString(rootMain, 16).c_str());

    return rootMain;
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

void MerkleGroupMultiPol::getElementsProof(MerkleGroupMultiPolTree &tree, uint32_t idx, vector<RawFr::Element> &val, vector<vector<RawFr::Element>> &mp)
{
    uint32_t group = idx % nGroups;
    uint32_t groupIdx = floor((float)idx / (float)nGroups);

    val.insert(val.begin(), tree.polTrees[group][groupIdx].begin(), tree.polTrees[group][groupIdx].begin() + nPols);
    mp.push_back(M->genMerkleProof(tree.groupTrees[group], groupIdx, 0));

    /*
    printf("#########\n");

    for (uint32_t k = 0; k < mp[0].size(); k++)
    {
        printf("%s\n", field.toString(mp[0][k], 16).c_str());
    }
    printf("#########\n");
    */
    mp.push_back(M->genMerkleProof(tree.mainTree, group, 0));
}

void MerkleGroupMultiPol::getElementsProof(uint32_t idx, RawFr::Element *val, uint32_t val_size, RawFr::Element *mp, uint32_t mp_size)
{
    uint32_t group = idx % nGroups;
    uint32_t groupIdx = floor((float)idx / (float)nGroups);

    std::memcpy(val, &polTrees[group * groupSize * polsProofSize + groupIdx * polsProofSize], nPols * sizeof(RawFr::Element));

    uint32_t mp_group_size = M->MerkleProofSize(groupSize) * M->arity;
    uint32_t mp_main_size = M->MerkleProofSize(nGroups) * M->arity;

    M->genMerkleProof(&groupTrees[group * groupProofSize], groupProofSize, groupIdx, 0, mp, mp_group_size);
    /*
    printf("#########\n");

    for (uint32_t k = 0; k < mp_group_size; k++)
    {
        printf("%s\n", field.toString(mp[k], 16).c_str());
    }
    printf("#########\n");
    */

    M->genMerkleProof(mainTree, ngroupsProofSize, group, 0, &mp[mp_group_size], mp_main_size);
    /*
    for (uint32_t k = 0; k < mp_main_size; k++)
    {
        printf("%s\n", field.toString(mp[mp_group_size + k], 16).c_str());
    }
    printf("#########\n");
    */
}

bool MerkleGroupMultiPol::verifyElementProof(RawFr::Element root, vector<vector<RawFr::Element>> &mp, uint32_t idx, vector<RawFr::Element> val)
{
    FrElement rootC = calculateRootFromElementProof(mp, idx, val);
    return field.eq(root, rootC);
}

bool MerkleGroupMultiPol::verifyElementProof(RawFr::Element root, RawFr::Element *mp, uint32_t mp_size, uint32_t idx, RawFr::Element *val, uint32_t val_size)
{
    FrElement rootC = calculateRootFromElementProof(mp, mp_size, idx, val, val_size);
    return field.eq(root, rootC);
}

RawFr::Element MerkleGroupMultiPol::calculateRootFromElementProof(vector<vector<RawFr::Element>> &mp, uint32_t idx, vector<RawFr::Element> val)
{

    uint32_t group = idx % nGroups;
    uint32_t groupIdx = floor((float)idx / (float)nGroups);

    M->merkelize(val);
    FrElement rootPol = M->root(val);

    FrElement rootGroup = M->calculateRootFromProof(mp[0], groupIdx, rootPol, 0);
    printf("#########\n");
    for (uint32_t k = 0; k < mp[0].size(); k++)
    {
        printf("%s\n", field.toString(mp[0][k], 16).c_str());
    }
    printf("#########\n");

    FrElement rootMain = M->calculateRootFromProof(mp[1], group, rootGroup, 0);
    printf("rootGroup: %s\n", field.toString(rootGroup, 16).c_str());
    printf("rootMain: %s\n", field.toString(rootMain, 16).c_str());

    return rootMain;
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

    printf("#########\n");
    for (uint32_t k = 0; k < mp_group_size; k++)
    {
        printf("%s\n", field.toString(mp[k], 16).c_str());
    }
    printf("#########\n");
    FrElement rootGroup = M->calculateRootFromProof(mp, mp_group_size, groupIdx, rootPol, 0);

    FrElement rootMain = M->calculateRootFromProof(&mp[mp_group_size], mp_main_size, group, rootGroup, 0);

    printf("rootGroup_: %s\n", field.toString(rootGroup, 16).c_str());
    printf("rootMain_: %s\n", field.toString(rootMain, 16).c_str());

    return rootMain;
}

RawFr::Element *MerkleGroupMultiPol::fileToMap(const string &fileName, FrElement *MerkleGroupMultiPolTree, Merkle *M, uint32_t nGroups, uint32_t groupSize, uint32_t nPols)
{

    uint32_t merkleGroupMultiPolTreeSize = getTreeSize(M, nGroups, groupSize, nPols);
    cout << "MerkleGroupMultiPol::mapToFile() calculated total size=" << merkleGroupMultiPolTreeSize << endl;

    //  Check the file size is the same as the expected
    struct stat sb;
    if (lstat(fileName.c_str(), &sb) == -1)
    {
        cerr << "Error: MerkleGroupMultiPol::mapToFile() failed calling lstat() of file " << fileName << endl;
        exit(-1);
    }
    if ((uint64_t)sb.st_size != merkleGroupMultiPolTreeSize)
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

uint32_t MerkleGroupMultiPol::getTreeSize(Merkle *M, uint32_t nGroups, uint32_t groupSize, uint32_t nPols)
{
    uint32_t polsProofSize = M->numHashes(nPols);
    uint32_t groupProofSize = M->numHashes(groupSize);
    uint32_t ngroupsProofSize = M->numHashes(nGroups);

    uint32_t polTreesSize = nGroups * groupSize * polsProofSize * sizeof(RawFr::Element);
    uint32_t groupTreesSize = nGroups * groupProofSize * sizeof(RawFr::Element);
    uint32_t mainTreeSize = ngroupsProofSize * sizeof(RawFr::Element);

    return mainTreeSize + groupTreesSize + polTreesSize;
}

uint32_t MerkleGroupMultiPol::getGroupProofSize(Merkle *M, uint32_t nGroups, uint32_t groupSize, uint32_t nPols)
{
    uint32_t ngroupsProofSize = M->numHashes(nGroups);
    uint32_t mp_size_array = (M->MerkleProofSize(ngroupsProofSize) - 1) * M->arity;
    return nPols * groupSize + mp_size_array;
}

// idx is the root of unity
RawFr::Element MerkleGroupMultiPol::getElement(FrElement *tree, uint32_t polIdx, uint32_t idx)
{
    uint32_t group = idx % nGroups;
    uint32_t groupIdx = floor((float)idx / (float)nGroups);

    return M->getElement(&tree[group * (groupProofSize + groupSize * polsProofSize) + groupIdx * polsProofSize], polIdx);
}