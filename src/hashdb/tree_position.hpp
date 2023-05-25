#ifndef TREE_POSITION_HPP
#define TREE_POSITION_HPP

#include <vector>
#include <cstdint>

using namespace std;

class TreePosition
{
public:
    vector<uint64_t> keys; // List of key bits, where 0 = left child, 1 = right child
    int64_t level; // Level at which the node is in the branch, i.e. limit of the remaining key
    TreePosition() : level(-1) {}; // Level = -1 means no valid data

    bool operator==(const TreePosition &other);

    TreePosition & operator=(const TreePosition &other);
};

#endif