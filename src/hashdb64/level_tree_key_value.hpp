#ifndef KV_TREE_HPP
#define KV_TREE_HPP
#include <iostream>
#include <vector>
#include <cassert>
#include <vector>
#include <cstring>
#include <random>
#include <gmpxx.h>
#include <string>
#include <bitset>
#include "level_tree.hpp"
#include "zkglobals.hpp"
#include "zkresult.hpp"

using namespace std;

struct
{
    mpz_class value;
    uint32_t prev;
} typedef ListedValue;

//
//  KVTree
//
class KVTree : public LevelTree
{
public:
    KVTree(){};
    KVTree(uint64_t nBitsStep_);
    void postConstruct(uint64_t nBitsStep_);
    KVTree& operator=(const KVTree& other);
    ~KVTree(){};

    zkresult write(const Goldilocks::Element (&key)[4], const mpz_class &value, uint64_t &level);
    zkresult read(const Goldilocks::Element (&key)[4],       mpz_class &value, uint64_t &level);
    zkresult extract(const Goldilocks::Element (&key)[4], const mpz_class &value); // returns ZKR_DB_KEY_NOT_FOUND if key was not found; value is used to check that it matches the latest value
    uint64_t level(const Goldilocks::Element (&key)[4]); // returns level; key might or might not exist in the tree
    

protected:
    uint64_t addValue(const uint64_t pileIdx, const mpz_class &value);
    void removeValue(const uint64_t pileIdx, mpz_class &value);

    vector<ListedValue> pileValues;
    uint64_t nValues = 0;

    vector<uint64_t> emptyValues;
    uint64_t nEmptyValues = 0;
};

inline uint64_t KVTree::level(const Goldilocks::Element (&key)[4])
{
    uint64_t key_[4]={key[0].fe,key[1].fe,key[2].fe,key[3].fe}; //avoidable copy
    return LevelTree::level(key_);
}
#endif