#ifndef FOREST_HPP
#define FOREST_HPP

#include "tree_chunk.hpp"

/*

A Tree (state) is made of a set of TreeChunks:

      /\
     /__\
        /\
       /__\
          /\
         /__\
        /\ 
       /__\

When we call SMT.get(root, key, value):
    - we want to read [key, value]
    - we call db.read(treeChunk.hash, treeChunk.data) starting from the root until we reach the [key, value] leaf node

When we call SMT.set(oldStateRoot, key, newValue, newStateRoot)
    - we want to write a new leaf node [key, newValue] and get the resulting newStateRoot
    - we calculate the new position of [key, newValue], creating new chunks if needed
    - we recalculate the hashes of all the modified and new chunks
    - we call db.write(treeChunk.hash, treeChunk.data) of all the modified and new chunks

Every time we call SMT.set(), we are potentially creating a new Tree = SUM(TreeChunks)
Every new Tree is a newer version of the state
Many Trees (states) coexist in the same Forest (state history)
Every executor.processBatch() can potentially create several new Trees (states)
The Forest takes note of the latest Tree hash to keep track of the current state:

     SR1      SR2      SR3      SR4
     /\       /\       /\       /\
    /__\     /__\     /__\     /__\
                /\       /\       /\
               /__\     /__\     /__\
                           /\       /\
                          /__\     /__\
                                  /\ 
                                 /__\

*/

class Forest
{
public:
    string root;

};

#endif