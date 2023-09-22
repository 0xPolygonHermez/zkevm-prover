#ifndef LEVELTREE_HPP
#define LEVELTREE_HPP
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
    LevelTree(uint64_t nBitsStep_);
    LevelTree(uint64_t nBitsStep_, uint64_t maxNodes_, uint64_t incNodes_, uint64_t maxPile_, uint64_t incPile_, uint64_t maxEmptyNodes_, uint64_t incEmptyNodes_, uint64_t maxEmptyKeys_, uint64_t incEmptyKeys_);
    ~LevelTree(){};
    void reset();

    // main methods
    uint64_t insert(uint64_t key[4]);
    void extract(uint64_t key[4]);
    uint64_t level(uint64_t key[4]);

private:
    const uint64_t nBitsKey = 256;
    const uint64_t nBitsStep;
    const uint16_t nodeSize;
    const uint64_t nSteps;

    uint64_t nNodes = 1;
    uint64_t nKeys = 1;

    vector<int64_t> nodes;
    vector<uint64_t> pile;
    vector<uint64_t> emptyNodes;
    vector<uint64_t> emptyKeys;

    // auxiliary methods
    int64_t addNode();
    int64_t addKey(uint64_t key[4]);
    void removeNode(uint64_t nodeId);
    void removeKey(uint64_t keyId);
    inline uint64_t getStepIndex(uint64_t key[4], uint64_t step);
    uint64_t levelInNode(uint64_t nodeId, uint64_t localIdx);
    uint64_t commonBits(uint64_t key1[4], uint64_t key2[4], uint64_t checkedBits = 0);
    void mixKey(uint64_t inKey[4], uint64_t outKey[4]);

    // memory settings
    uint64_t maxNodes = 1 << 17;
    uint64_t incNodes = 1 << 17;
    uint64_t maxPile = 1 << 16;
    uint64_t incPile = 1 << 16;
    uint64_t maxEmptyNodes = 1 << 15;
    uint64_t incEmptyNodes = 1 << 15;
    uint64_t maxEmptyKeys = 1 << 14;
    uint64_t incEmptyKeys = 1 << 14;

    void allocate();
};

uint64_t LevelTree::getStepIndex(uint64_t key[4], uint64_t step)
{
    uint64_t indx = (step * nBitsStep) >> 6;
    const uint64_t one = 1;
    return (key[indx] >> (nBitsKey - ((step + one) * nBitsStep))) & ((one << nBitsStep) - one);
}

#endif