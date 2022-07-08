#ifndef MERKLEHASH_GOLDILOCKS
#define MERKLEHASH_GOLDILOCKS

#include <cassert>
#include "goldilocks_base_field.hpp"
#include "goldilocks_cubic_extension.hpp"

#define MERKLEHASHGOLDILOCKS_HEADER_SIZE 3

class MerklehashGoldilocks
{
public:
    static void getElement(Goldilocks::Element &element, Goldilocks::Element *tree, uint64_t idx, uint64_t subIdx)
    {
        uint64_t elementSize = Goldilocks::toU64(tree[0]);
        uint64_t elementsInLinear = Goldilocks::toU64(tree[1]);
        uint64_t nLinears = Goldilocks::toU64(tree[2]);

        assert((idx > 0) || (idx < nLinears));
        assert(elementSize == 1);

        element = tree[MERKLEHASHGOLDILOCKS_HEADER_SIZE + idx * elementsInLinear + subIdx];
    };
    static void getElement(Goldilocks3::Element &element, Goldilocks::Element *tree, uint64_t idx, uint64_t subIdx)
    {
        uint64_t elementSize = Goldilocks::toU64(tree[0]);
        uint64_t elementsInLinear = Goldilocks::toU64(tree[1]);
        uint64_t nLinears = Goldilocks::toU64(tree[2]);

        assert((idx > 0) || (idx < nLinears));
        assert(elementSize == 3);

        std::memcpy(element, &tree[MERKLEHASHGOLDILOCKS_HEADER_SIZE + (idx * elementsInLinear + subIdx) * elementSize], sizeof(Goldilocks3::Element));
    };

    static void root(Goldilocks::Element (&root)[HASH_SIZE], Goldilocks::Element *tree, uint64_t numElementsTree)
    {
        std::memcpy(root, &tree[numElementsTree - HASH_SIZE], HASH_SIZE * sizeof(Goldilocks::Element));
    }
};

#endif