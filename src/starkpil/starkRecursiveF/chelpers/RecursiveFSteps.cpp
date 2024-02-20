#include "RecursiveFSteps.hpp"

void RecursiveFSteps::calculateExpressions(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams, bool useGeneric) {
    uint32_t nrowsBatch = 4;
        bool domainExtended = parserParams.stage > 3 ? true : false;
        RecursiveFSteps::parser_avx(starkInfo, params, parserArgs, parserParams, nrowsBatch, domainExtended);
}