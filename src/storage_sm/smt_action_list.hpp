#ifndef SMT_ACTION_LIST_HPP
#define SMT_ACTION_LIST_HPP

#include "smt_action.hpp"

class SmtActionList
{
public:
    vector<SmtAction> action;
    void addGetAction (SmtGetResult &getResult)
    {
        SmtAction a;
        a.bIsSet = false;
        a.getResult = getResult;
        action.push_back(a);
    }
    void addSetAction (SmtSetResult &setResult)
    {
        SmtAction a;
        a.bIsSet = true;
        a.setResult = setResult;
        action.push_back(a);
    }
};

#endif