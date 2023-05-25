#ifndef FLUSH_DATA_HPP
#define FLUSH_DATA_HPP

#include <string>
#include "goldilocks_base_field.hpp"
#include "tree_position.hpp"

using namespace std;

class FlushData
{
public:
    string key;
    string value;
    TreePosition treePosition;
};

#endif