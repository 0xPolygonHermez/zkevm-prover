#ifndef SMT_ACTION_LIST_HPP
#define SMT_ACTION_LIST_HPP

#include "smt.hpp"

class SmtAction
{
public:
    bool bIsSet;
    SmtGetResult getResult;
    SmtSetResult setResult;
};

#endif