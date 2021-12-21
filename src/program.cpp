#include <iostream>
#include "program.hpp"

using namespace std;

eOperation string2op (const string s)
{
    if (s == "field_set") return op_field_set;
    if (s == "field_add") return op_field_add;
    if (s == "field_sub") return op_field_sub;
    if (s == "field_neg") return op_field_neg;
    if (s == "field_mul") return op_field_mul;
    if (s == "pol_add") return op_pol_add;
    if (s == "pol_sub") return op_pol_sub;
    if (s == "pol_neg") return op_pol_neg;
    if (s == "pol_mul") return op_pol_mul;
    if (s == "pol_addc") return op_pol_addc;
    if (s == "pol_mulc") return op_pol_mulc;
    if (s == "pol_grandProduct") return op_pol_grandProduct;
    if (s == "pol_batchInverse") return op_pol_batchInverse;
    if (s == "pol_rotate") return op_pol_rotate;
    if (s == "pol_extend") return op_pol_extend;
    if (s == "pol_getEvaluation") return op_pol_getEvaluation;
    if (s == "treeGroupMultipol_extractPol") return op_treeGroupMultipol_extractPol;
    if (s == "treeGroupMultipol_merkelize") return op_treeGroupMultipol_merkelize;
    if (s == "treeGroupMultipol_root") return op_treeGroupMultipol_root;
    if (s == "treeGroupMultipol_getGroupProof") return op_treeGroupMultipol_getGroupProof;
    if (s == "treeGroup_merkelize") return op_treeGroup_merkelize;
    if (s == "treeGroup_root") return op_treeGroup_root;
    if (s == "treeGroup_getElementProof") return op_treeGroup_getElementProof;
    if (s == "treeGroup_getGroupProof") return op_treeGroup_getGroupProof;
    if (s == "idxArrayFromFields") return op_idxArrayFromFields;
    if (s == "idxArray_get") return op_idxArray_get;
    if (s == "idx_addMod") return op_idx_addMod;
    if (s == "calculateH1H2") return op_calculateH1H2;
    if (s == "friReduce") return op_friReduce;
    if (s == "hash") return op_hash;
    if (s == "log") return op_log;
    cerr << "Error: string2op() unrecognized string op: " << s << endl;
    exit(-1);
}

string op2string (eOperation op)
{
    switch (op)
    {
        case op_field_set: return "field_set";
        case op_field_add: return "field_add";
        case op_field_sub: return "field_sub";
        case op_field_neg: return "field_neg";
        case op_field_mul: return "field_mul";
        case op_pol_add: return "pol_add";
        case op_pol_sub: return "pol_sub";
        case op_pol_neg: return "pol_neg";
        case op_pol_mul: return "pol_mul";
        case op_pol_addc: return "pol_addc";
        case op_pol_mulc: return "pol_mulc";
        case op_pol_grandProduct: return "pol_grandProduct";
        case op_pol_batchInverse: return "pol_batchInverse";
        case op_pol_rotate: return "pol_rotate";
        case op_pol_extend: return "pol_extend";
        case op_pol_getEvaluation: return "pol_getEvaluation";
        case op_treeGroupMultipol_extractPol: return "treeGroupMultipol_extractPol";
        case op_treeGroupMultipol_merkelize: return "treeGroupMultipol_merkelize";
        case op_treeGroupMultipol_root: return "treeGroupMultipol_root";
        case op_treeGroupMultipol_getGroupProof: return "treeGroupMultipol_getGroupProof";
        case op_treeGroup_merkelize: return "treeGroup_merkelize";
        case op_treeGroup_root: return "treeGroup_root";
        case op_treeGroup_getElementProof: return "treeGroup_getElementProof";
        case op_treeGroup_getGroupProof: return "treeGroup_getGroupProof";
        case op_idxArrayFromFields: return "idxArrayFromFields";
        case op_idxArray_get: return "idxArray_get";
        case op_idx_addMod: return "idx_addMod";
        case op_calculateH1H2: return "calculateH1H2";
        case op_friReduce: return "friReduce";
        case op_hash: return "hash";
        case op_log: return "log";
        case op_unknown:
        default:
            cerr << "Error: op2string() unrecognized op: " << op << endl;
            exit(-1);
    }
}