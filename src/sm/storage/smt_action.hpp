#ifndef SMT_ACTION_HPP
#define SMT_ACTION_HPP

#include "smt.hpp"
#include "poseidon_goldilocks.hpp"

class SmtAction
{
public:
    bool bIsSet;
    SmtGetResult getResult;
    SmtSetResult setResult;
    string toString (Goldilocks &fr);
};

#endif