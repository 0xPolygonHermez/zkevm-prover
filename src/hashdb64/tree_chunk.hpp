#ifndef TREE_CHUNK_HPP
#define TREE_CHUNK_HPP

#include <string>
#include "goldilocks_base_field.hpp"
#include "zkresult.hpp"
#include "database_64.hpp"
#include "leaf_node.hpp"
#include "intermediate_node.hpp"
#include "child.hpp"

using namespace std;

/*
A Tree 64 is an SMT data model based on SMT chunks of 1 hash --> 64 childs, i.e. 6 levels of the SMT tree
Every child an be zero, a leaf node (key, value), or an intermediate node (the hash of another Tree 64)

A TreeChunk contains 6 levels of children:

child1:                                                                      *    ^                   level
children2:                                  *                                     |                   level+1
children4:                  *                               *                     |                   level+2
children8:          *               *               *              *              |  calculateHash()  level+3
children16:     *       *       *       *       *       *       *      * ...      |                   level+4
children32:   *   *   *   *   *   *   *   *   *   *   *   *   *   *  *  * ...     |                   level+5
children64:  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ...    |                   
                ^                         |
                |  data2children()        |  children2data()
                |                         \/
data:        xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

During a process batch, we need to:
- read from the database the required TreeChunks (SMT get)
- modify them (SMT set)
- and finally recalculate all the hashes and save the result in the database

*/

#define TREE_CHUNK_HEIGHT 6
#define TREE_CHUNK_WIDTH 64
#define TREE_CHUNK_MAX_DATA_SIZE (TREE_CHUNK_WIDTH*(32+32) + 16 + 16) // All leaves = 64*(key+value) + isZero + isLeaf

class TreeChunk
{
private:
    Database64          &db;
    Goldilocks          &fr;
public:
    uint64_t            level; // Level of the top hash of the chunk: 0, 6, 12, 18, 24, etc.
    uint8_t             key; // 6 bits portion of the total key at this level of the SMT tree
    Goldilocks::Element hash[4];
    Child               child1;
    Child               children2[2];
    Child               children4[4];
    Child               children8[8];
    Child               children16[16];
    Child               children32[32];
    Child               children64[TREE_CHUNK_WIDTH];

    // Encoded data
    string              data;

    // Constructor
    TreeChunk(Database64 &db) : db(db), fr(db.fr) {};

    // Encode/decode data functions
    zkresult data2children (void); // Decodde data and store result into children64
    zkresult children2data (void); // Encode children64 into data

    // Calculate hash functions
    zkresult calculateHash (void); // Calculate the hash of the chunk based on the (new) values of children64
    zkresult calculateChildren (const uint64_t level, Child * inputChildren, Child * outputChildren, uint64_t outputSize); // Calculates outputSize output children, as a result of combining outputSize*2 input children
    zkresult calculateChild (const uint64_t level, Child &leftChild, Child &rightChild, Child &outputChild);
};

#endif