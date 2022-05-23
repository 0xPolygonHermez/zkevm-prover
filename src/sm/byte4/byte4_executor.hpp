#ifndef BYTE4_EXECUTOR_HPP
#define BYTE4_EXECUTOR_HPP

#include <map>
#include "commit_pols.hpp"

using namespace std;

class Byte4Executor
{
public:
    void execute (map<uint32_t, bool> &input, Byte4CommitPols & pols);
};

#endif