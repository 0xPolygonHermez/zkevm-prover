#include "level_tree_key_value.hpp"

KVTree::KVTree(uint64_t nBitsStep_)
{
    postConstruct(nBitsStep_);
}

void KVTree::postConstruct(uint64_t nBitsStep_)
{
    assert(nBitsKey % nBitsStep_ == 0);
    assert(nBitsStep_ <= 32);
    nBitsStep = nBitsStep_;
    stepMask = (1ULL << nBitsStep) - 1;
    stepsPerKeyWord = 64 / nBitsStep;
    nodeSize = 1 << nBitsStep;
    nSteps = nBitsKey / nBitsStep;
    useInsertCounters = true;
    pileSlotSize = 6;

    nodes.resize(4096 * nodeSize, 0);
    nNodes = 1;
    pile.resize(1024 * pileSlotSize, 0);
    nKeys = 1;
    emptyNodes.resize(512, 0);
    nEmptyNodes = 0;
    emptyKeys.resize(512, 0);
    nEmptyKeys = 0;
    pileValues.resize(4096, {0, 0});
    nValues = 0;
    emptyValues.resize(1024, 0);
    nEmptyValues = 0;
}

bool KVTree::read(const uint64_t key[4], mpz_class &value, uint64_t &level)
{
    int64_t pileIdx;
    level = LevelTree::level(key, &pileIdx);
    if (pileIdx == -1)
    {
        return false;
    }
    else
    {
        value = pileValues[pile[pileIdx * 6 + 5]].value;
        return true;
    }
}

void KVTree::write(const uint64_t key[4], const mpz_class &value, uint64_t &level)
{
    int64_t pileIdx;
    level = insert(key, &pileIdx);
    addValue(pileIdx, value);
}

bool KVTree::extract(const uint64_t key[4], mpz_class &value)
{
    int64_t pileIdx;
    bool bfound = LevelTree::extract(key, &pileIdx);
    if (bfound)
    {
        removeValue(pileIdx, value);
    }
    return bfound;
}
uint64_t KVTree::addValue(const uint64_t pileIdx, const mpz_class &value)
{
    int idx;
    if (nEmptyValues > 0)
    {
        nEmptyValues--;
        idx = emptyValues[nEmptyValues];
    }
    else
    {
        idx = nValues;
        nValues++;
        if (nValues == pileValues.size())
        {
            pileValues.resize(pileValues.size() * 2, {0, 0});
        }
    }
    pileValues[idx].value = value;
    pileValues[idx].prev = pile[pileIdx * 6 + 5];
    pile[pileIdx * 6 + 5] = idx;
    return idx;
}
void KVTree::removeValue(uint64_t pileIdx, mpz_class &value)
{
    uint64_t valIdx = pile[pileIdx * 6 + 5];
    value = pileValues[valIdx].value;
    pile[pileIdx * 6 + 5] = pileValues[valIdx].prev;
    emptyValues[nEmptyValues] = valIdx;
    nEmptyValues++;
    if (nEmptyValues == emptyValues.size())
    {
        emptyValues.resize(emptyValues.size() * 2, 0);
    }
}
