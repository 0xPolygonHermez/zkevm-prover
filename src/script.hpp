#ifndef SCRIPT_HPP
#define SCRIPT_HPP

#include <string>
#include <iostream>
#include <vector>
#include <nlohmann/json.hpp>
#include "pol_types.hpp"

using namespace std;
using json = nlohmann::json;

enum eReferenceType {
    rt_unknown = 0,
    rt_pol = 1,
    rt_field = 2,
    rt_treeGroup = 3,
    rt_treeGroupMultipol = 4,
    rt_treeGroup_elementProof = 5,
    rt_treeGroup_groupProof = 6,
    rt_treeGroupMultipol_groupProof = 7,
    rt_idxArray = 8,
    rt_int = 9
};

class Reference
{
public:
    uint64_t id; // Mandatory
    eReferenceType type; // Mandatory
    uint64_t N;
    eElementType elementType;
    uint64_t nGroups;
    uint64_t groupSize;
    uint64_t nPols;
};

class Program
{
public:
    string op; // Mandatory
    string msg;
    uint64_t result;
    uint64_t tree;
    uint64_t nGroups;
    uint64_t groupSize;
    uint64_t nPols;
    uint64_t polIdx;
    string value;
    vector<uint64_t> values;
    uint64_t extendBits;
    uint64_t N;
    vector<uint64_t> pols;
    uint64_t f;
    uint64_t t;
    uint64_t resultH1;
    uint64_t resultH2;
    uint64_t constant; // const
    uint64_t shift;
    uint64_t idx;
    uint64_t pos;
    uint64_t idxArray;
    string w;
    vector<uint64_t> fields;
    uint64_t add;
    uint64_t mod;

};

class Script
{
private:
    bool bParsed;

public:

    vector<Reference> refs;
    vector<Program> program;

    Script(): bParsed(false) {};
    void parse (json &scriptJson);

private:
    void parseReferences (json &scriptJson);
    void parseProgram (json &scriptJson);
};

#endif