#ifndef CHECK_TREE_HPP
#define CHECK_TREE_HPP

#include "zkresult.hpp"
#include "database.hpp"

using namespace std;

class CheckTreeCounters
{
public:
    uint64_t intermediateNodes;
    uint64_t leafNodes;
    uint64_t values;
    uint64_t maxLevel;
    CheckTreeCounters() : intermediateNodes(0), leafNodes(0), values(0), maxLevel(0) {};
};

zkresult CheckTree (Database &db, const string &key, uint64_t level, CheckTreeCounters &checkTreeCounters);

#endif