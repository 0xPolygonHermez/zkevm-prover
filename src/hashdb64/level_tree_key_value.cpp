#include "level_tree_key_value.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"

KVTree::KVTree(uint64_t nBitsStep_)
{
    postConstruct(nBitsStep_);
}

KVTree& KVTree::operator=(const KVTree& other){
    if(this == &other){
        return *this;
    }
    nBitsStep = other.nBitsStep;
    stepMask = other.stepMask;
    stepsPerKeyWord = other.stepsPerKeyWord;
    nodeSize = other.nodeSize;
    nSteps = other.nSteps;
    useInsertCounters = other.useInsertCounters;
    pileSlotSize = other.pileSlotSize;
    nodes = other.nodes;
    nNodes = other.nNodes;
    pile = other.pile;
    nKeys = other.nKeys;
    emptyNodes = other.emptyNodes;
    nEmptyNodes = other.nEmptyNodes;
    emptyKeys = other.emptyKeys;
    nEmptyKeys = other.nEmptyKeys;
    pileValues = other.pileValues;
    nValues = other.nValues;
    emptyValues = other.emptyValues;
    nEmptyValues = other.nEmptyValues;
    return *this;
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

zkresult KVTree::read(const Goldilocks::Element (&key)[4], mpz_class &value, uint64_t &level)
{
    int64_t pileIdx;
    uint64_t key_[4]={key[0].fe,key[1].fe,key[2].fe,key[3].fe}; //avoidable copy
    level = LevelTree::level(key_, &pileIdx);
    if (pileIdx == -1)
    {
        return ZKR_DB_KEY_NOT_FOUND;
    }
    else
    {
        value = pileValues[pile[pileIdx * 6 + 5]].value;
        return ZKR_SUCCESS;
    }
}

zkresult KVTree::write(const Goldilocks::Element (&key)[4], const mpz_class &value, uint64_t &level)
{
    int64_t pileIdx;
    uint64_t key_[4]={key[0].fe,key[1].fe,key[2].fe,key[3].fe}; //avoidable copy
    level = insert(key_, &pileIdx);
    addValue(pileIdx, value);
    return ZKR_SUCCESS;
}

zkresult KVTree::extract(const Goldilocks::Element (&key)[4], const mpz_class &value)
{
    int64_t pileIdx;
    uint64_t key_[4]={key[0].fe,key[1].fe,key[2].fe,key[3].fe}; //avoidable copy
    bool bfound = LevelTree::extract(key_, &pileIdx);
    zkresult result = ZKR_DB_KEY_NOT_FOUND;
    if (bfound)
    {
        mpz_class value_;
        removeValue(pileIdx, value_);
        if(value_ != value){
            zklog.error("KeyValueTree::extract() found stored value=" + value_.get_str(10) + " != provided value=" + value.get_str(10));
            exitProcess();
        }
        result = ZKR_SUCCESS;
    }
    return result;
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
