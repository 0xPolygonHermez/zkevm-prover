#ifndef PROGRAM_HPP
#define PROGRAM_HPP

#include <string>
#include <vector>

using namespace std;

class Program
{
public:
    string op; // Mandatory // TODO: Map to an enum to save time while executing
    string msg;
    string value;
    string w;
    string shiftInv;
    uint64_t result;
    uint64_t tree;
    uint64_t nGroups;
    uint64_t groupSize;
    uint64_t nPols;
    uint64_t polIdx;
    uint64_t extendBits;
    uint64_t N;
    uint64_t f;
    uint64_t t;
    uint64_t resultH1;
    uint64_t resultH2;
    uint64_t constant;
    uint64_t shift;
    uint64_t idx;
    uint64_t pos;
    uint64_t idxArray;
    uint64_t add;
    uint64_t mod;
    uint64_t specialX;
    vector<uint64_t> values;
    vector<uint64_t> pols;
    vector<uint64_t> fields;
};

#endif