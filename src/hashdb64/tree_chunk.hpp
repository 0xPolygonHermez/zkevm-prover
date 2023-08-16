#ifndef TREE_CHUNK_HPP
#define TREE_CHUNK_HPP

#include <string>
#include "goldilocks_base_field.hpp"
#include "zkresult.hpp"
#include "database_64.hpp"

using namespace std;

/*
A Tree 64 is an SMT data model based on SMT chunks of 1 hash --> 64 childs, i.e. 6 levels of the SMT tree
Every child an be zero, a leaf (key, value), or another Tree 64
*/

#define TREE_CHUNK_HEIGHT 6
#define TREE_CHUNK_WIDTH 64
#define TREE_CHUNK_MAX_DATA_SIZE (TREE_CHUNK_WIDTH*(32+32) + 16 + 16) // All leaves = 64*(key+value) + isZero + isLeaf

class Leaf
{
public:
    Goldilocks::Element key[4];
    mpz_class value; // 256 bits
};

class Intermediate
{
public:
    Goldilocks::Element hash[4];
};

enum ChildType
{
    UNSPECIFIED  = 0,
    ZERO         = 1,
    LEAF         = 2,
    INTERMEDIATE = 3
};

class Child
{
public:
    ChildType    type;
    Leaf         leaf;
    Intermediate intermediate;
    Child() : type(UNSPECIFIED) {};
};

class TreeChunk
{
private:
    Database64          &db;
public:
    uint8_t             key; // 6 bits portion of the total key at this level of the SMT tree
    Goldilocks::Element hash[4];
    Child               children[TREE_CHUNK_WIDTH];

    // Encoded data
    string              data;

    // Constructor
    TreeChunk(Database64 &db) : db(db) {};

    // Encode/decode data functions
    zkresult data2children(void);
    zkresult children2data(void);
};

#endif