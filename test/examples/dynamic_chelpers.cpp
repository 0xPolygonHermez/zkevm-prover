#include <type_traits>

#include "chelpers_dyn_interface.hpp"
#include "AllSteps.hpp"

static AllSteps allSteps;

extern "C" void calculateExpressions(StarkInfo *starkInfo, StepsParams *params, ParserArgs *parserArgs, ParserParams *parserParams)
{
    allSteps.calculateExpressions(*starkInfo, *params, *parserArgs, *parserParams);
}

static_assert(std::is_same<decltype(&calculateExpressions), CalculateExpressionsFnPtr>::value, "Function does not match the required signature");
