#include "merkle/merkle.hpp"
#include <algorithm>
#include <iterator>
#include <math.h>   /* floor */
#include <assert.h> /* assert */

#include <omp.h>
Merkle::Merkle(uint8_t _arity)
{
    arity = _arity;
}

void Merkle::merkelize(vector<FrElement> &elements)
{
    uint32_t offset = 0;

    while ((elements.size() - offset) > 1)
    {
        offset = merkelizeLevel(elements, offset);
    }
}

uint32_t Merkle::merkelizeLevel(vector<FrElement> &elements, uint32_t offset)
{
    uint32_t numHashes = (uint32_t)ceil((float)(elements.size() - offset) / (float)arity);

    vector<FrElement> hash(numHashes);
#pragma omp parallel for
    for (uint32_t i = offset; i < elements.size(); i += arity)
    {

        vector<RawFr::Element> buff;
        uint32_t leafSize = std::min((uint32_t)elements.size() - i, (uint32_t)arity);

        copy(elements.begin() + i, elements.begin() + i + leafSize, back_inserter(buff));
        if (leafSize < arity)
        {
            vector<RawFr::Element> zeros(arity - leafSize, field.zero());
            buff.insert(buff.end(), zeros.begin(), zeros.end());
        }
        poseidon.hash(buff, &(hash[(i - offset) / arity]));
    }
    elements.insert(elements.end(), hash.begin(), hash.end());

    return elements.size() - numHashes;
}

Merkle::FrElement Merkle::getElement(vector<FrElement> &elements, uint32_t idx)
{
    return elements[idx];
}

vector<Merkle::FrElement> Merkle::genMerkleProof(vector<Merkle::FrElement> &tree, uint32_t idx, uint32_t offset)
{
    vector<Merkle::FrElement> proof;
    uint32_t n = _nElements(tree.size() - offset);
    if (n <= 1)
        return proof;
    uint32_t nextIdx = floor(idx / arity);
    uint32_t nc = std::min((int)arity, (int)(n - nextIdx * arity));

    vector<Merkle::FrElement> a;
    a.insert(a.begin(), tree.begin() + offset + nextIdx * arity, tree.begin() + offset + nextIdx * arity + nc);

    if (nc < arity)
    {
        vector<RawFr::Element> zeros(arity - nc, field.zero());
        a.insert(a.end(), zeros.begin(), zeros.end());
    }

    vector<Merkle::FrElement> nextProof = genMerkleProof(tree, nextIdx, offset + n);

    proof.insert(proof.end(), a.begin(), a.end());
    proof.insert(proof.end(), nextProof.begin(), nextProof.end());

    return proof;
}

uint32_t Merkle::_nElements(uint32_t n)
{
    if (n <= 1)
        return 0;

    uint32_t l = 0;
    vector<uint32_t> treeSize = {1};

    while (n > treeSize[l])
    {
        l++;
        treeSize.push_back(treeSize[l - 1] + std::pow(arity, l));
    }

    uint32_t acc = 0;
    uint32_t rem = n;
    while ((l > 0) && (rem > 0))
    {
        rem--;
        acc += floor(rem / treeSize[l - 1]) * std::pow(arity, (l - 1));
        rem = rem % treeSize[l - 1];
        l--;
    }

    assert(rem != 1);

    return acc;
}

Merkle::FrElement Merkle::calculateRootFromProof(vector<FrElement> &mp, uint32_t idx, FrElement value, uint32_t offset)
{
    if ((mp.size() - offset) == 0)
    {
        return value;
    }

    uint32_t curIdx = idx % arity;
    uint32_t nextIdx = floor((float)idx / (float)arity);

    vector<Merkle::FrElement> a;
    a.insert(a.begin(), mp.begin() + offset, mp.begin() + offset + arity);
    FrElement nextValue;
    a[curIdx] = value;
    poseidon.hash(a, &nextValue);

    return calculateRootFromProof(mp, nextIdx, nextValue, offset + arity);
}

bool Merkle::verifyMerkleProof(FrElement root, vector<FrElement> &mp, uint32_t idx, FrElement value, uint32_t offset)
{
    FrElement rootC = calculateRootFromProof(mp, idx, value, offset);
    return field.eq(root, rootC);
}

Merkle::FrElement Merkle::root(vector<Merkle::FrElement> &tree)
{
    return tree[tree.size() - 1];
}