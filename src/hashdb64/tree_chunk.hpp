#ifndef TREE_CHUNK_HPP
#define TREE_CHUNK_HPP

#include <string>
#include <vector>
#include "goldilocks_base_field.hpp"
#include "poseidon_goldilocks.hpp"
#include "zkresult.hpp"
#include "database_64.hpp"
#include "leaf_node.hpp"
#include "intermediate_node.hpp"
#include "child.hpp"
#include "key_value.hpp"
#include "hash_value_gl.hpp"
#include "zkglobals.hpp"

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
    //Database64          &db;
    //Goldilocks          &fr;
    PoseidonGoldilocks  &poseidon;
private:
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

    // Flags
    bool bHashValid;
    bool bChildrenRestValid;
    bool bChildren64Valid;
    bool bDataValid;

public:
    // Encoded data
    string              data;
    vector<uint64_t>    list; // List of IDs to KeyValue instances to write or to read

public:

    // Constructor
    TreeChunk(/*Database64 &db,*/ PoseidonGoldilocks &poseidon) :
        //db(db),
        //fr(db.fr),
        poseidon(poseidon),
        bHashValid(false),
        bChildrenRestValid(false),
        bChildren64Valid(false),
        bDataValid(false)
    {
        hash[0] = fr.zero();
        hash[1] = fr.zero();
        hash[2] = fr.zero();
        hash[3] = fr.zero();
    };

    // Read from database
    //zkresult readDataFromDb (const Goldilocks::Element (&hash)[4]);

    // Encode/decode data functions
    zkresult data2children (void); // Decodde data and store result into children64
    zkresult children2data (void); // Encode children64 into data
    uint64_t numberOfNonZeroChildren (void);

    // Calculate hash functions
    zkresult calculateHash (vector<HashValueGL> *hashValues); // Calculate the hash of the chunk based on the (new) values of children64
    zkresult calculateChildren (const uint64_t level, Child * inputChildren, Child * outputChildren, uint64_t outputSize, vector<HashValueGL> *hashValues); // Calculates outputSize output children, as a result of combining outputSize*2 input children
    zkresult calculateChild (const uint64_t level, Child &leftChild, Child &rightChild, Child &outputChild, vector<HashValueGL> *hashValues);

    // Children access
    const Child & getChild (uint64_t position)
    {
        return children64[position];
    };
    const Child & getChild1 (void)
    {
        return child1;
    };

    void setChild (uint64_t position, const Child & child)
    {
        children64[position] = child;
        bChildrenRestValid = false;
        bHashValid = false;
        bDataValid = false;
    };
    void setLeafChild (uint64_t position, const Goldilocks::Element (&key)[4], const mpz_class &value)
    {
        children64[position].type = LEAF;
        children64[position].leaf.key[0] = key[0];
        children64[position].leaf.key[1] = key[1];
        children64[position].leaf.key[2] = key[2];
        children64[position].leaf.key[3] = key[3];
        children64[position].leaf.value = value;
        bChildrenRestValid = false;
        bHashValid = false;
        bDataValid = false;
    }
    void setZeroChild (uint64_t position)
    {
        children64[position].type = ZERO;
        bChildrenRestValid = false;
        bHashValid = false;
        bDataValid = false;
    }
    void setTreeChunkChild (uint64_t position, uint64_t id)
    {
        children64[position].type = TREE_CHUNK;
        children64[position].treeChunkId = id;
        bChildrenRestValid = false;
        bHashValid = false;
        bDataValid = false;
    }
    bool getDataValid (void)
    {
        return bDataValid;
    }

    void setLevel (uint64_t _level)
    {
        level = _level;
    }

    void resetToZero (uint64_t _level)
    {
        level = _level;
        hash[0] = fr.zero();
        hash[1] = fr.zero();
        hash[2] = fr.zero();
        hash[3] = fr.zero();
        for (uint64_t i=0; i<64; i++)
        {
            children64[i].type = ZERO;
        }
        bDataValid = false;
        bChildren64Valid = true;
        bChildrenRestValid = false;
        bHashValid = true;
    }

    void getHash (Goldilocks::Element (&result)[4])
    {
        if (!bHashValid)
        {
            zklog.error("TreeChunk::getHash() called with bHashValid=false");
            exitProcess();
        }
        result[0] = hash[0];
        result[1] = hash[1];
        result[2] = hash[2];
        result[3] = hash[3];
    }

    void getLeafHash(const uint64_t position, Goldilocks::Element (&result)[4]);

    void print (void) const;
};

#endif