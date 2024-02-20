#include "chelpers_steps.hpp"


class AllSteps : public CHelpersSteps {
    public:
        void calculateExpressions(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams, bool useGeneric);
    private:
        void parser_avx(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams, uint32_t nrowsBatch, bool domainExtended);
};