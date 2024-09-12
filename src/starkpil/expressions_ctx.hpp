#ifndef EXPRESSIONS_CTX_HPP
#define EXPRESSIONS_CTX_HPP
#include "expressions_bin.hpp"
#include "const_pols.hpp"
#include "stark_info.hpp"
#include "steps.hpp"
#include "setup_ctx.hpp"

class ExpressionsCtx {
public:

    SetupCtx setupCtx;

    ExpressionsCtx(SetupCtx& _setupCtx) : setupCtx(_setupCtx) {};

    virtual ~ExpressionsCtx() {};
    
    virtual void calculateExpressions(StepsParams& params, Goldilocks::Element *dest, ParserArgs &parserArgs, ParserParams &parserParams, bool domainExtended, bool inverse = false, bool imPols = false) {};
 
    void calculateExpression(StepsParams& params, Goldilocks::Element* dest, uint64_t expressionId, bool inverse = false, bool imPols = false) {
        bool domainExtended = expressionId == setupCtx.starkInfo.cExpId || expressionId == setupCtx.starkInfo.friExpId;
        calculateExpressions(params, dest, setupCtx.expressionsBin.expressionsBinArgsExpressions, setupCtx.expressionsBin.expressionsInfo[expressionId], domainExtended, inverse, imPols);
    }
};

#endif