#ifndef CHECK_TREE_64_HPP
#define CHECK_TREE_64_HPP

#include "zkresult.hpp"
#include "database_64.hpp"

using namespace std;

class CheckTreeCounters64
{
public:
    uint64_t intermediateNodes;
    uint64_t leafNodes;
    uint64_t values;
    uint64_t maxLevel;
    CheckTreeCounters64() : intermediateNodes(0), leafNodes(0), values(0), maxLevel(0) {};
};

zkresult CheckTree64 (Database64 &db, const string &key, uint64_t level, CheckTreeCounters64 &checkTreeCounters);

#endif