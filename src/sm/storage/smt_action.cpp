#include "smt_action.hpp"

string SmtAction::toString (Goldilocks &fr)
{
    string result;
    if (bIsSet)
    {
        result += "SmtAction.setResult=\n" + setResult.toString(fr);
    }
    else
    {
        result += "SmtAction.getResult=\n" + getResult.toString(fr);
    }
    return result;
}