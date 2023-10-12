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
    ~KVTree(){};

    bool read(const uint64_t key[4], mpz_class &value, uint64_t &level);
    void write(const uint64_t key[4], const mpz_class &value, uint64_t &level);
    bool extract(const uint64_t key[4], mpz_class &value);

protected:
    uint64_t addValue(const uint64_t pileIdx, const mpz_class &value);
    void removeValue(const uint64_t pileIdx, mpz_class &value);

    vector<ListedValue> pileValues;
    uint64_t nValues = 0;

    vector<uint64_t> emptyValues;
    uint64_t nEmptyValues = 0;
};

#endif