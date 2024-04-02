#ifndef CHELPERS_HPP
#define CHELPERS_HPP

#include <string>
#include <map>
#include "binfile_utils.hpp"
#include "polinomial.hpp"
#include "goldilocks_base_field.hpp"
#include "goldilocks_base_field_avx.hpp"
#include "goldilocks_cubic_extension.hpp"
#include "constant_pols_starks.hpp"
#include "stark_info.hpp"
#include <immintrin.h>
#include <cassert>

const int CHELPERS_STAGES_SECTION = 2;
const int CHELPERS_EXPRESSIONS_SECTION = 3;
const int CHELPERS_CONSTRAINTS_SECTION = 4;
const int CHELPERS_HINTS_SECTION = 5;

class HintField
{
public:
    opType operand;
    uint64_t id;
    uint64_t value;    
};


class Hint 
{
public:
    std::string name;
    std::map<string,HintField> fields;    
};

struct ParserParams
{
    uint32_t stage;
    uint32_t executeBefore;
    uint32_t expId;
    uint32_t nTemp1;
    uint32_t nTemp3;
    uint32_t nOps;
    uint32_t opsOffset;
    uint32_t nArgs;
    uint32_t argsOffset;
    uint32_t nNumbers;
    uint64_t numbersOffset;
    uint32_t nConstPolsUsed;
    uint32_t constPolsOffset;
    uint32_t nCmPolsUsed;
    uint32_t cmPolsOffset;
    uint32_t nChallengesUsed;
    uint32_t challengesOffset;
    uint32_t nPublicsUsed;
    uint32_t publicsOffset;
    uint32_t nSubproofValuesUsed;
    uint32_t subproofValuesOffset;
    uint32_t nCmPolsCalculated;
    uint32_t cmPolsCalculatedOffset;
    uint32_t destDim;
    uint32_t destId;
};

struct ParserArgs 
{
    uint8_t* ops;
    uint16_t* args;
    uint64_t* numbers;
    uint16_t* constPolsIds;
    uint16_t* cmPolsIds;
    uint16_t* challengesIds;
    uint16_t* publicsIds;
    uint16_t* subproofValuesIds;
    uint16_t* cmPolsCalculatedIds;
};

class CHelpers
{
public:
    std::vector<ParserParams> stagesInfo;

    std::vector<ParserParams> expressionsInfo;

    std::vector<ParserParams> constraintsInfoDebug;

    std::vector<Hint> hints;
    
    ParserArgs cHelpersArgs;

    ParserArgs cHelpersArgsDebug;

    ParserArgs cHelpersArgsExpressions;

    ~CHelpers() {
        delete[] cHelpersArgs.ops;
        delete[] cHelpersArgs.args;
        delete[] cHelpersArgs.numbers;
        delete[] cHelpersArgs.constPolsIds;
        delete[] cHelpersArgs.cmPolsIds;
        delete[] cHelpersArgs.challengesIds;
        delete[] cHelpersArgs.publicsIds;
        delete[] cHelpersArgs.subproofValuesIds;
        delete[] cHelpersArgs.cmPolsCalculatedIds;

        delete[] cHelpersArgsExpressions.ops;
        delete[] cHelpersArgsExpressions.args;
        delete[] cHelpersArgsExpressions.numbers;
        delete[] cHelpersArgsExpressions.constPolsIds;
        delete[] cHelpersArgsExpressions.cmPolsIds;
        delete[] cHelpersArgsExpressions.challengesIds;
        delete[] cHelpersArgsExpressions.publicsIds;
        delete[] cHelpersArgsExpressions.subproofValuesIds;
        delete[] cHelpersArgsExpressions.cmPolsCalculatedIds;

        delete[] cHelpersArgsDebug.ops;
        delete[] cHelpersArgsDebug.args;
        delete[] cHelpersArgsDebug.numbers;
        delete[] cHelpersArgsDebug.constPolsIds;
        delete[] cHelpersArgsDebug.cmPolsIds;
        delete[] cHelpersArgsDebug.challengesIds;
        delete[] cHelpersArgsDebug.publicsIds;
        delete[] cHelpersArgsDebug.subproofValuesIds;        
    };

    void loadCHelpers(BinFileUtils::BinFile *cHelpersBin);
};


#endif
