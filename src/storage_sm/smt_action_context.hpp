#ifndef SMT_ACTION_CONTEXT_HPP
#define SMT_ACTION_CONTEXT_HPP

#include <map>
#include <vector>
#include "ff/ff.hpp"
#include "smt_action.hpp"

class SmtActionContext
{
public:
    uint64_t level;
    int64_t currentLevel;
    FieldElement key[4];
    FieldElement rKey[4];
    vector<uint64_t> bits;
    map< uint64_t, vector<FieldElement> > siblings;
    void init (const SmtAction &action);
};

#endif