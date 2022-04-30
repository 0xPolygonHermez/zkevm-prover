#ifndef SMT_ACTION_HPP
#define SMT_ACTION_HPP

#include "smt.hpp"

class SmtAction
{
public:
    bool bIsSet;
    SmtGetResult getResult;
    SmtSetResult setResult;
};

#endif