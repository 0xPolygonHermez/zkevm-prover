#ifndef PROGRAM_HPP
#define PROGRAM_HPP

#include <string>
#include <vector>

using namespace std;

enum eOperation
{
    op_unknown = 0,
    op_field_set = 1,
    op_field_add = 2,
    op_field_sub = 3,
    op_field_neg = 4,
    op_field_mul = 5,
    op_pol_add = 6,
    op_pol_sub = 7,
    op_pol_neg = 8,
    op_pol_mul = 9,
    op_pol_addc = 10,
    op_pol_mulc = 11,
    op_pol_grandProduct = 12,
    op_pol_batchInverse = 13,
    op_pol_rotate = 14,
    op_pol_extend = 15,
    op_pol_getEvaluation = 16,
    op_treeGroupMultipol_extractPol = 17,
    op_treeGroupMultipol_merkelize = 18,
    op_treeGroupMultipol_root = 19,
    op_treeGroupMultipol_getGroupProof = 20,
    op_treeGroup_merkelize = 21,
    op_treeGroup_root = 22,
    op_treeGroup_getElementProof = 23,
    op_treeGroup_getGroupProof = 24,
    op_idxArrayFromFields = 25,
    op_idxArray_get = 26,
    op_idx_addMod = 27,
    op_calculateH1H2 = 28,
    op_friReduce = 29,
    op_hash = 30,
    op_log = 31
};

eOperation string2op (const string s);
string op2string (eOperation op);

class Program
{
public:
    eOperation op; // Mandatory
    string msg;
    string value; // TODO: Parse once, use many
    string w;
    string shiftInv;
    uint64_t result;
    uint64_t tree;
    uint64_t nGroups;
    uint64_t groupSize;
    uint64_t nPols;
    uint64_t polIdx;
    uint64_t extendBits;
    uint64_t reduceBits;
    uint64_t pol;
    uint64_t N;
    uint64_t f;
    uint64_t t;
    uint64_t resultH1;
    uint64_t resultH2;
    uint64_t constant;
    uint64_t shift;
    uint64_t idx;
    uint64_t p;
    uint64_t pos;
    uint64_t idxArray;
    uint64_t add;
    uint64_t mod;
    uint64_t specialX;
    uint64_t n;
    uint64_t nBits;
    vector<uint64_t> values;
    vector<uint64_t> pols;
    vector<uint64_t> fields;
};

#endif