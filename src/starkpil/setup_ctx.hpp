#ifndef SETUP_CTX_HPP
#define SETUP_CTX_HPP

#include "stark_info.hpp"
#include "const_pols.hpp"
#include "expressions_bin.hpp"

class SetupCtx {
public:

    StarkInfo *starkInfo;
    ExpressionsBin *expressionsBin;
    ConstPols *constPols;

    SetupCtx(string starkInfoFile, string expressionsBinFile, string constPolsFile) {
        starkInfo = new StarkInfo(starkInfoFile);
        expressionsBin = new ExpressionsBin(expressionsBinFile);
        constPols = new ConstPols(*starkInfo, constPolsFile);
    }

    SetupCtx(string starkInfoFile, string expressionsBinFile, string constPolsFile, string constTreeFile) {
        starkInfo = new StarkInfo(starkInfoFile);
        expressionsBin = new ExpressionsBin(expressionsBinFile);
        constPols = new ConstPols(*starkInfo, constPolsFile, constTreeFile);
    }

    ~SetupCtx() {
        delete starkInfo;
        delete expressionsBin;
        delete constPols;
    };
};

#endif