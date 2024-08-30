#ifndef SETUP_CTX_HPP
#define SETUP_CTX_HPP

#include "stark_info.hpp"
#include "const_pols.hpp"
#include "expressions_bin.hpp"

class SetupCtx {
public:

    StarkInfo &starkInfo;
    ExpressionsBin &expressionsBin;
    ConstPols &constPols;

    SetupCtx(StarkInfo &_starkInfo, ExpressionsBin& _expressionsBin, ConstPols& _constPols) : starkInfo(_starkInfo), expressionsBin(_expressionsBin), constPols(_constPols)  {};

    ~SetupCtx() {};
};

#endif