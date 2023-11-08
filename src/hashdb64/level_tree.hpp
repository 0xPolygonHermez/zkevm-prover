#ifndef LEVEL_TREE_HPP
#define LEVEL_TREE_HPP

#include <iostream>
#include <vector>
#include <cassert>
#include <vector>
#include <cstring>
#include <random>
#include <gmpxx.h>
#include <string>
#include <bitset>

using namespace std;

//
//  LevelTree
//
class LevelTree
{

public:
    LevelTree(){};
    LevelTree(uint64_t nBitsStep_, bool useInsertCounters_ = false);
    void virtual postConstruct(uint64_t nBitsStep_, bool useInsertCounters_ = false);
    ~LevelTree(){};

    // main methods
    uint64_t insert(const uint64_t key[4], int64_t *pileIdx = nullptr);
    bool extract(const uint64_t key[4], int64_t *pileIdx = nullptr);   // pileIdx = -1 if not found
    uint64_t level(const uint64_t key[4], int64_t *pileIdx = nullptr); // pileIdx = -1 if not found

protected:
    const uint64_t nBitsKey = 256;
    uint64_t nBitsStep;
    uint64_t stepMask;
    uint64_t stepsPerKeyWord;
    uint64_t nodeSize;
    uint64_t nSteps;
    uint64_t pileSlotSize;
    bool useInsertCounters;

    vector<int64_t> nodes;
    uint64_t nNodes;

    vector<uint64_t> pile;
    uint64_t nKeys;

    vector<uint64_t> emptyNodes;
    uint64_t nEmptyNodes;

    vector<uint64_t> emptyKeys;
    uint64_t nEmptyKeys;

    // auxiliary methods
    int64_t addNode();
    void removeNode(uint64_t nodeId);
    virtual int64_t addKey(const uint64_t key[4]);
    bool removeKey(uint64_t keyId);
    inline void getKey(uint64_t keyId, uint64_t key[4]);
    inline uint64_t getStepIndex(uint64_t key[4], uint64_t step);
    uint64_t levelInNode(uint64_t nodeId, uint64_t localIdx);
    uint64_t commonBits(const uint64_t key1[4], const uint64_t key2[4], uint64_t checkedBits = 0);
    void mixKey(const uint64_t inKey[4], uint64_t outKey[4]);
};

uint64_t LevelTree::getStepIndex(uint64_t key[4], uint64_t step)
{
    uint64_t indx = (step * nBitsStep) >> 6;
    uint64_t stepInWord = step - indx * stepsPerKeyWord;
    return (key[indx] >> (64 - ((stepInWord + 1) * nBitsStep))) & stepMask;
}
void LevelTree::getKey(uint64_t keyId, uint64_t key[4])
{
    uint64_t offset = pileSlotSize * keyId;
    key[0] = pile[offset];
    key[1] = pile[offset + 1];
    key[2] = pile[offset + 2];
    key[3] = pile[offset + 3];
}

#endif