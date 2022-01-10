#include "merkle.hpp"
#include <algorithm>
#include <iterator>
#include <math.h>   /* floor */
#include <assert.h> /* assert */
#include <cstring>
#include <math.h> /* log10 */

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

void Merkle::merkelize(Merkle::FrElement *elements, uint32_t size)
{
    uint32_t prevSize = 0;
    uint32_t offset = 0;
    while (size > prevSize + 1)
    {
        prevSize = size;
        size = merkelizeLevel(elements, size, offset);
        offset = prevSize;
    }
}
uint32_t Merkle::merkelizeLevel(Merkle::FrElement *elements, uint32_t size, uint32_t offset)
{
    uint32_t numHashes = (uint32_t)ceil((float)(size - offset) / (float)arity);
    FrElement hash[numHashes];
#pragma omp parallel for
    for (uint32_t i = offset; i < size; i += arity)
    {
        uint32_t leafSize = std::min(size - i, (uint32_t)arity);
        std::vector<RawFr::Element> buff(elements + i, elements + i + leafSize);

        if (leafSize < arity)
        {
            vector<RawFr::Element> zeros(arity - leafSize, field.zero());
            buff.insert(buff.end(), zeros.begin(), zeros.end());
        }
        poseidon.hash(buff, &(hash[(i - offset) / arity]));
    }
    uint32_t sizeElement = sizeof(RawFr::Element);
    uint64_t pos = (uint64_t)elements + size * sizeElement;
    std::memcpy((void *)pos, &hash, numHashes * sizeElement);
    return size + numHashes;
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

uint32_t Merkle::numHashes(uint32_t n)
{
    uint32_t prevSize = 0;
    uint32_t offset = 0;
    while (n > prevSize + 1)
    {
        prevSize = n;
        uint32_t numHashes = (uint32_t)ceil((float)(n - offset) / (float)arity);
        n += numHashes;
        offset = prevSize;
    }

    return n;
}

Merkle::FrElement Merkle::getElement(vector<FrElement> &elements, uint32_t idx)
{
    return elements[idx];
}

RawFr::Element Merkle::getElement(Merkle::FrElement *elements, uint32_t idx)
{
    return elements[idx];
}

uint32_t Merkle::MerkleProofSize(uint32_t size)
{
    if (size > 1)
    {
        return (uint32_t)ceil(log10(size) / log10(arity));
    }
    return 1;
}

uint32_t Merkle::MerkleProofSizeBytes(uint32_t size)
{
    return (uint32_t)ceil(log10(size) / log10(arity)) * arity * sizeof(RawFr::Element);
}

vector<Merkle::FrElement> Merkle::genMerkleProof(vector<Merkle::FrElement> &tree, uint32_t idx, uint32_t offset)
{
    vector<Merkle::FrElement> proof;
    uint32_t n = _nElements(tree.size() - offset);
    if (n <= 1)
        return proof;
    uint32_t nextIdx = floor((float)idx / (float)arity);
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

void Merkle::genMerkleProof(Merkle::FrElement *elements, uint32_t size, uint32_t idx, uint32_t offset, Merkle::FrElement *proof, uint32_t pSize)
{
    uint32_t n = _nElements(size - offset);
    uint32_t pos = MerkleProofSize(n) - 1;
    uint32_t proofSize = MerkleProofSize(_nElements(size));
    if (n <= 1)
        return;
    uint32_t nextIdx = floor((float)idx / (float)arity);
    uint8_t nc = std::min((int)arity, (int)(n - nextIdx * arity));
    Merkle::FrElement a[arity] = {field.zero()};
    // std::memcpy(&a, elements + (offset + nextIdx * arity), nc * sizeof(RawFr::Element));

    std::memcpy(&a, &(elements[offset + nextIdx * arity]), nc * sizeof(RawFr::Element));

    genMerkleProof(elements, size, nextIdx, offset + n, proof, proofSize);

    std::memcpy(&proof[(proofSize - pos - 1) * arity], &a, arity * sizeof(RawFr::Element));

    return;
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
    assert(field.eq(a[curIdx], value) == 1);
    // a[curIdx] = value;

    poseidon.hash(a, &nextValue);

    return calculateRootFromProof(mp, nextIdx, nextValue, offset + arity);
}

Merkle::FrElement Merkle::calculateRootFromProof(FrElement *mp, uint32_t size, uint32_t idx, FrElement value, uint32_t offset)
{
    if (size - offset == 0)
    {
        return value;
    }
    uint32_t curIdx = idx % arity;
    uint32_t nextIdx = floor((float)idx / (float)arity);

    vector<Merkle::FrElement> a(&mp[offset], &mp[offset] + (uint64_t)arity);
    FrElement nextValue;
    assert(field.eq(a[curIdx], value) == 1);
    // a[curIdx] = value;
    poseidon.hash(a, &nextValue);

    return calculateRootFromProof(mp, size, nextIdx, nextValue, offset + arity);
}

bool Merkle::verifyMerkleProof(FrElement root, vector<FrElement> &mp, uint32_t idx, FrElement value, uint32_t offset)
{
    FrElement rootC = calculateRootFromProof(mp, idx, value, offset);
    return field.eq(root, rootC);
}

bool Merkle::verifyMerkleProof(FrElement root, Merkle::FrElement *mp, uint32_t mp_size, uint32_t idx, FrElement value, uint32_t offset)
{
    FrElement rootC = calculateRootFromProof(mp, mp_size, idx, value, offset);
    return field.eq(root, rootC);
}

RawFr::Element Merkle::root(vector<Merkle::FrElement> &tree)
{
    return tree[tree.size() - 1];
}

RawFr::Element Merkle::root(Merkle::FrElement *tree, uint32_t size)
{
    return tree[size - 1];
}