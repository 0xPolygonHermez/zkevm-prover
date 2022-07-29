#ifndef MERKLEHASH_GOLDILOCKS
#define MERKLEHASH_GOLDILOCKS

#include <cassert>
#include <math.h> /* floor */
#include "goldilocks_base_field.hpp"
#include "goldilocks_cubic_extension.hpp"

#define MERKLEHASHGOLDILOCKS_HEADER_SIZE 2
#define MERKLEHASHGOLDILOCKS_ARITY 2
class MerklehashGoldilocks
{
public:
    static void getElement(Goldilocks::Element &element, Goldilocks::Element *tree, uint64_t idx, uint64_t subIdx)
    {
        uint64_t elementsInLinear = Goldilocks::toU64(tree[0]);
        uint64_t nLinears = Goldilocks::toU64(tree[1]);

        assert((idx > 0) || (idx < nLinears));

        element = tree[MERKLEHASHGOLDILOCKS_HEADER_SIZE + idx * elementsInLinear + subIdx];
    };

    static void root(Goldilocks::Element *root, Goldilocks::Element *tree, uint64_t numElementsTree)
    {
        std::memcpy(root, &tree[numElementsTree - HASH_SIZE], HASH_SIZE * sizeof(Goldilocks::Element));
    }

    static void root(Goldilocks::Element (&root)[HASH_SIZE], Goldilocks::Element *tree, uint64_t numElementsTree)
    {
        std::memcpy(root, &tree[numElementsTree - HASH_SIZE], HASH_SIZE * sizeof(Goldilocks::Element));
    }

    static void getGroupProof(Goldilocks::Element *proof, Goldilocks::Element *tree, uint64_t idx)
    {
        uint64_t elementsInLinear = Goldilocks::toU64(tree[0]);
        uint64_t nLinears = Goldilocks::toU64(tree[1]);

//#pragma omp parallel for
        for (uint64_t i = 0; i < elementsInLinear; i++)
        {

            getElement(proof[i], tree, idx, i);
        }

        genMerkleProof(&proof[elementsInLinear], tree, idx, MERKLEHASHGOLDILOCKS_HEADER_SIZE + elementsInLinear * nLinears, nLinears * HASH_SIZE);
    }

    static void genMerkleProof(Goldilocks::Element *proof, Goldilocks::Element *tree, uint32_t idx, uint32_t offset, uint32_t n)
    {

        if (n <= HASH_SIZE)
            return;
        uint64_t nextIdx = idx >> 1;
        uint64_t si = (idx ^ 1) * HASH_SIZE;

        std::memcpy(proof, &tree[offset + si], HASH_SIZE * sizeof(Goldilocks::Element));

        uint64_t nextN = (std::floor((n - 1) / 8) + 1) * HASH_SIZE;
        genMerkleProof(&proof[HASH_SIZE], tree, nextIdx, offset + nextN * 2, nextN);
    }

    static uint32_t _nElements(uint32_t n)
    {
        if (n <= 1)
            return 0;

        uint32_t l = 0;
        std::vector<uint32_t> treeSize = {1};

        while (n > treeSize[l])
        {
            l++;
            treeSize.push_back(treeSize[l - 1] + std::pow(MERKLEHASHGOLDILOCKS_ARITY, l));
        }

        uint32_t acc = 0;
        uint32_t rem = n;
        while ((l > 0) && (rem > 0))
        {
            rem--;
            acc += floor(rem / treeSize[l - 1]) * std::pow(MERKLEHASHGOLDILOCKS_ARITY, (l - 1));
            rem = rem % treeSize[l - 1];
            l--;
        }

        assert(rem != 1);

        return acc;
    }

    static uint32_t MerkleProofSize(uint32_t size)
    {
        if (size > 1)
        {
            return (uint32_t)ceil(log10(size) / log10(MERKLEHASHGOLDILOCKS_ARITY));
        }
        return 0;
    }

    static uint64_t getTreeNumElements(uint64_t numCols, uint64_t degree)
    {
        return numCols * degree + degree * HASH_SIZE + (degree - 1) * HASH_SIZE + MERKLEHASHGOLDILOCKS_HEADER_SIZE;
    };
};

#endif