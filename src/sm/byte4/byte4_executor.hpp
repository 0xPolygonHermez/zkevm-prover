#ifndef BYTE4_EXECUTOR_HPP
#define BYTE4_EXECUTOR_HPP

#include <map>
#include "commit_pols.hpp"
#include "goldilocks/goldilocks_base_field.hpp"

using namespace std;

class Byte4Executor
{
public:
    Goldilocks &fr;
    const uint64_t N;
    Byte4Executor(Goldilocks &fr) : fr(fr), N(Byte4CommitPols::pilDegree()) {}
    void execute (map<uint32_t, bool> &input, Byte4CommitPols & pols);
};

#endif