#include "level_tree.hpp"

using namespace std;

static const uint32_t mask16[] = {0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF};
static const uint64_t mask32[] = {0x5555555555555555, 0x3333333333333333, 0x0F0F0F0F0F0F0F0F, 0x00FF00FF00FF00FF, 0x0000FFFF0000FFFF};
uint32_t interleave16(uint16_t a, uint16_t b);
uint64_t interleave32(uint32_t a, uint32_t b);

LevelTree::LevelTree(uint64_t nBitsStep_, bool useInsertCouters_)
{
    postConstruct(nBitsStep_, useInsertCouters_);
}

void LevelTree::postConstruct(uint64_t nBitsStep_, bool useInsertCounters_)
{
    assert(nBitsKey % nBitsStep_ == 0);
    assert(nBitsStep_ <= 32);
    nBitsStep = nBitsStep_;
    stepMask = (1ULL << nBitsStep) - 1;
    stepsPerKeyWord = 64 / nBitsStep;
    nodeSize = 1 << nBitsStep;
    nSteps = nBitsKey / nBitsStep;
    useInsertCounters = useInsertCounters_;
    if (useInsertCounters)
    {
        pileSlotSize = 5;
    }
    else
    {
        pileSlotSize = 4;
    }
    nodes.resize(4096 * nodeSize, 0);
    nNodes = 1;
    pile.resize(1024 * pileSlotSize, 0);
    nKeys = 1;
    emptyNodes.resize(512, 0);
    nEmptyNodes = 0;
    emptyKeys.resize(512, 0);
    nEmptyKeys = 0;
}

uint64_t LevelTree::insert(const uint64_t key[4], int64_t *pileIdx)
{
    uint64_t currentNodeIdx = 0;
    int64_t nextNodeIdx;
    uint64_t step;
    uint64_t mixedKey[4];
    mixKey(key, mixedKey);

    for (step = 0; step < nSteps; step++)
    {
        uint64_t index = getStepIndex(mixedKey, step);
        nextNodeIdx = nodes[currentNodeIdx * nodeSize + index];
        if (nextNodeIdx == 0)
        {
            int64_t auxPileIdx = addKey(mixedKey);
            nodes[currentNodeIdx * nodeSize + index] = -auxPileIdx;
            if (pileIdx != nullptr)
                *pileIdx = auxPileIdx;
            return step * nBitsStep + levelInNode(currentNodeIdx, index);
        }
        else if (nextNodeIdx > 0)
        {
            currentNodeIdx = nextNodeIdx;
        }
        else
        {
            uint64_t pileKey[4];
            getKey(-nextNodeIdx, pileKey);
            if (pileKey[0] != mixedKey[0] || pileKey[1] != mixedKey[1] || pileKey[2] != mixedKey[2] || pileKey[3] != mixedKey[3])
            {
                int64_t nextPileIdxKey = -addKey(mixedKey);
                if (pileIdx != nullptr)
                    *pileIdx = -nextPileIdxKey;
                int64_t nextPileIdxPKey = nextNodeIdx;
                uint64_t nextIndexKey;
                uint64_t nextIndexPKey;
                do
                {
                    ++step;
                    int64_t addedNodeIdx = addNode();
                    nodes[currentNodeIdx * nodeSize + index] = addedNodeIdx;

                    nextIndexKey = getStepIndex(mixedKey, step);
                    nextIndexPKey = getStepIndex(pileKey, step);
                    if (nextIndexKey != nextIndexPKey)
                    {
                        nodes[addedNodeIdx * nodeSize + nextIndexKey] = nextPileIdxKey;
                        nodes[addedNodeIdx * nodeSize + nextIndexPKey] = nextPileIdxPKey;
                        return step * nBitsStep + levelInNode(addedNodeIdx, nextIndexKey);
                    }
                    else
                    {
                        index = nextIndexKey;
                        currentNodeIdx = addedNodeIdx;
                    }
                } while (step < nSteps);
            }
            else
            {
                if (useInsertCounters)
                {
                    uint64_t offset = pileSlotSize * -nextNodeIdx;
                    pile[offset + 4]++;
                }
                if (pileIdx != nullptr)
                    *pileIdx = -nextNodeIdx;
                return step * nBitsStep + levelInNode(currentNodeIdx, index);
            }
            break;
        }
    }
    assert(0); // should never reach this point
    return 0;
}

uint64_t LevelTree::level(const uint64_t key[4], int64_t *pileIdx)
{
    uint64_t currentNodeIdx = 0;
    int64_t nextNodeIdx;
    uint64_t step;
    uint64_t mixedKey[4];
    mixKey(key, mixedKey);

    for (step = 0; step < nSteps; step++)
    {
        uint64_t index = getStepIndex(mixedKey, step);
        nextNodeIdx = nodes[currentNodeIdx * nodeSize + index];
        if (nextNodeIdx == 0)
        {
            if (pileIdx != nullptr)
            {
                *pileIdx = -1;
            }
            return step * nBitsStep + levelInNode(currentNodeIdx, index);
        }
        else if (nextNodeIdx > 0)
        {
            currentNodeIdx = nextNodeIdx;
        }
        else
        {
            uint64_t pileKey[4];
            getKey(-nextNodeIdx, pileKey);
            uint64_t nbits = step * nBitsStep + levelInNode(currentNodeIdx, index);
            uint64_t common = commonBits(mixedKey, pileKey, nbits);
            if (pileIdx != nullptr)
            {
                if (common == nbits)
                {
                    *pileIdx = -nextNodeIdx;
                }
                else
                {
                    *pileIdx = -1;
                }
            }
            return common;
        }
    }
    assert(0); // should never reach this point
    return 0;
}

bool LevelTree::extract(const uint64_t key[4], int64_t *pileIdx)
{
    int64_t nextId;
    vector<uint64_t> nodeIds(nSteps);
    nodeIds[0] = 0;
    uint64_t mixedKey[4];
    mixKey(key, mixedKey);

    for (uint64_t step = 0; step < nSteps; step++)
    {
        uint64_t index = getStepIndex(mixedKey, step);
        nextId = nodes[nodeIds[step] * nodeSize + index];

        if (nextId == 0)
        {
            if (pileIdx != nullptr)
                *pileIdx = -1;
            return false;
        }
        else if (nextId > 0)
        {
            nodeIds[step + 1] = nextId;
        }
        else if (nextId < 0)
        {
            uint64_t pileKey[4];
            getKey(-nextId, pileKey);
            if (pileKey[0] != mixedKey[0] ||
                pileKey[1] != mixedKey[1] ||
                pileKey[2] != mixedKey[2] ||
                pileKey[3] != mixedKey[3])
            {
                if (pileIdx != nullptr)
                    *pileIdx = -1;
                return false;
            }
            else
            {
                if (pileIdx != nullptr)
                    *pileIdx = -nextId;
                if (removeKey(-nextId))
                {
                    nodes[nodeIds[step] * nodeSize + index] = 0;
                }

                while (step > 0)
                {
                    // Find if there is a unique neighbour at same node and with same level
                    uint64_t nneigh = 0;
                    uint64_t nodeIndexNeighbour = 0;
                    int64_t pileIndexNeighbour = 0;
                    for (uint64_t i = 0; i < nodeSize; ++i)
                    {
                        uint64_t aux = nodes[nodeIds[step] * nodeSize + i];
                        if (aux != 0)
                        {
                            ++nneigh;
                            if (nneigh > 1)
                            {
                                return true;
                            }
                            pileIndexNeighbour = aux;
                            nodeIndexNeighbour = i;
                        }
                    }
                    if (nneigh == 1 && pileIndexNeighbour < 0)
                    {
                        nodes[nodeIds[step] * nodeSize + nodeIndexNeighbour] = 0; // REUSE THIS NODE POSITION LATTER
                        step--;
                        uint64_t keyNeighbour[4];
                        getKey(-pileIndexNeighbour, keyNeighbour);
                        uint64_t indexPrev = getStepIndex(keyNeighbour, step);
                        nodes[nodeIds[step] * nodeSize + indexPrev] = pileIndexNeighbour;
                    }
                    else
                    {
                        return true;
                    }
                }
                return true;
            }
        }
    }
    assert(0); // should never reach this point
    return false;
}

// Returns node index
int64_t LevelTree::addNode()
{
    int64_t nodeId;
    if (nEmptyNodes > 0)
    {
        nodeId = emptyNodes[nEmptyNodes - 1];
        nEmptyNodes--;
    }
    else
    {
        nodeId = nNodes;
        nNodes++;
        if (nodeSize * nNodes == nodes.size())
        {
            nodes.resize(nodes.size() * 2, 0);
        }
    }
    return nodeId;
}

// Returns Key index
int64_t LevelTree::addKey(const uint64_t key[4])
{
    int64_t keyId;
    if (nEmptyKeys > 0)
    {
        keyId = emptyKeys[nEmptyKeys - 1];
        nEmptyKeys--;
    }
    else
    {
        keyId = nKeys;
        nKeys++;
        if (pileSlotSize * nKeys == pile.size())
        {
            pile.resize(pile.size() * 2, 0);
        }
    }
    uint64_t offset = pileSlotSize * keyId;
    pile[offset] = key[0];
    pile[offset + 1] = key[1];
    pile[offset + 2] = key[2];
    pile[offset + 3] = key[3];
    if (useInsertCounters)
        pile[offset + 4]++;

    return keyId;
}

void LevelTree::removeNode(uint64_t nodeId)
{
    assert(nodeId < nNodes);
    memset(&nodes[nodeId * nodeSize], 0, nodeSize * sizeof(int64_t));
    emptyNodes[nEmptyNodes++] = nodeId;
    if (nEmptyNodes == emptyNodes.size())
    {
        emptyNodes.resize(emptyNodes.size() * 2, 0);
    }
}

bool LevelTree::removeKey(uint64_t keyIdx)
{
    assert(keyIdx < nKeys);

    uint64_t offset = pileSlotSize * keyIdx;
    assert((!useInsertCounters) || (pile[offset + 4] > 0));
    if ((!useInsertCounters) || (pile[offset + 4] == 1))
    {
        pile[offset] = 0;
        pile[offset + 1] = 0;
        pile[offset + 2] = 0;
        pile[offset + 3] = 0;
        if (useInsertCounters)
            pile[offset + 4] = 0;
        emptyKeys[nEmptyKeys++] = keyIdx;
        if (nEmptyKeys == emptyKeys.size())
        {
            emptyKeys.resize(emptyKeys.size() * 2, 0);
        }
        return true;
    }
    else
    {
        pile[offset + 4]--;
        return false;
    }
}

uint64_t LevelTree::levelInNode(uint64_t nodeId, uint64_t localIdx)
{
    uint16_t globalIdx = localIdx + nodeId * nodeSize;
    assert(localIdx < nodeSize);
    uint64_t auxIndex = localIdx;
    uint64_t nbits = 0;
    uint64_t start = nodeId * nodeSize;
    for (uint64_t j = 0; j < nBitsStep; ++j)
    {
        uint64_t inc = 1 << (nBitsStep - j - 1);
        if (auxIndex >= inc)
        {
            start = start + inc;
            auxIndex = auxIndex - inc;
        }
        bool found = false;
        for (uint64_t i = start; i < start + inc; ++i)
        {
            if (nodes[i] != 0 && i != globalIdx)
            {
                ++nbits;
                found = true;
                break;
            }
        }
        if (!found)
        {
            return nbits;
        }
    }
    return nbits;
}

uint64_t LevelTree::commonBits(const uint64_t key1[4], const uint64_t key2[4], uint64_t checkedBits)
{
    uint64_t commonWords = checkedBits >> 6;
    for (int i = commonWords; i < 4; ++i)
    {
        if (key1[i] == key2[i])
            commonWords++;
        else
            break;
    }
    if (commonWords == 4)
    {
        return checkedBits;
    }
    uint64_t w1 = key1[commonWords];
    uint64_t w2 = key2[commonWords];

    int count = 0;
    uint64_t mask = 1ULL << 63;

    while (mask > 0)
    {
        if ((w1 & mask) == (w2 & mask))
        {
            count++;
            mask >>= 1;
        }
        else
            break;
    }
    return commonWords * 64 + count;
}

void LevelTree::mixKey(const uint64_t inKey[4], uint64_t outKey[4])
{
    uint16_t k00 = inKey[0] >> 48;
    uint16_t k01 = inKey[0] >> 32;
    uint16_t k02 = inKey[0] >> 16;
    uint16_t k03 = inKey[0];

    uint16_t k10 = inKey[1] >> 48;
    uint16_t k11 = inKey[1] >> 32;
    uint16_t k12 = inKey[1] >> 16;
    uint16_t k13 = inKey[1];

    uint16_t k20 = inKey[2] >> 48;
    uint16_t k21 = inKey[2] >> 32;
    uint16_t k22 = inKey[2] >> 16;
    uint16_t k23 = inKey[2];

    uint16_t k30 = inKey[3] >> 48;
    uint16_t k31 = inKey[3] >> 32;
    uint16_t k32 = inKey[3] >> 16;
    uint16_t k33 = inKey[3];

    // interleave16

    uint32_t k00k20 = interleave16(k00, k20);
    uint32_t k10k30 = interleave16(k10, k30);

    uint32_t k01k21 = interleave16(k01, k21);
    uint32_t k11k31 = interleave16(k11, k31);

    uint32_t k02k22 = interleave16(k02, k22);
    uint32_t k12k32 = interleave16(k12, k32);

    uint32_t k03k23 = interleave16(k03, k23);
    uint32_t k13k33 = interleave16(k13, k33);

    // interleave32
    outKey[0] = interleave32(k00k20, k10k30);
    outKey[1] = interleave32(k01k21, k11k31);
    outKey[2] = interleave32(k02k22, k12k32);
    outKey[3] = interleave32(k03k23, k13k33);
}

uint32_t interleave16(uint16_t x, uint16_t y)
{
    uint32_t a = x;
    uint32_t b = y;

    a = (a | (a << 8)) & mask16[3];
    a = (a | (a << 4)) & mask16[2];
    a = (a | (a << 2)) & mask16[1];
    a = (a | (a << 1)) & mask16[0];

    b = (b | (b << 8)) & mask16[3];
    b = (b | (b << 4)) & mask16[2];
    b = (b | (b << 2)) & mask16[1];
    b = (b | (b << 1)) & mask16[0];

    return b | (a << 1);
}

uint64_t interleave32(uint32_t x, uint32_t y)
{
    uint64_t a = x;
    uint64_t b = y;

    a = (a | (a << 16)) & mask32[4];
    a = (a | (a << 8)) & mask32[3];
    a = (a | (a << 4)) & mask32[2];
    a = (a | (a << 2)) & mask32[1];
    a = (a | (a << 1)) & mask32[0];

    b = (b | (b << 16)) & mask32[4];
    b = (b | (b << 8)) & mask32[3];
    b = (b | (b << 4)) & mask32[2];
    b = (b | (b << 2)) & mask32[1];
    b = (b | (b << 1)) & mask32[0];

    return b | (a << 1);
}
