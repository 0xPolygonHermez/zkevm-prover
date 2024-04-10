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
        if (cHelpersArgs.ops) delete[] cHelpersArgs.ops;
        if (cHelpersArgs.args) delete[] cHelpersArgs.args;
        if (cHelpersArgs.numbers) delete[] cHelpersArgs.numbers;
        if (cHelpersArgs.constPolsIds) delete[] cHelpersArgs.constPolsIds;
        if (cHelpersArgs.cmPolsIds) delete[] cHelpersArgs.cmPolsIds;
        if (cHelpersArgs.challengesIds) delete[] cHelpersArgs.challengesIds;
        if (cHelpersArgs.publicsIds) delete[] cHelpersArgs.publicsIds;
        if (cHelpersArgs.subproofValuesIds) delete[] cHelpersArgs.subproofValuesIds;
        if (cHelpersArgs.cmPolsCalculatedIds) delete[] cHelpersArgs.cmPolsCalculatedIds;

        if (cHelpersArgsExpressions.ops) delete[] cHelpersArgsExpressions.ops;
        if (cHelpersArgsExpressions.args) delete[] cHelpersArgsExpressions.args;
        if (cHelpersArgsExpressions.numbers) delete[] cHelpersArgsExpressions.numbers;
        if (cHelpersArgsExpressions.constPolsIds) delete[] cHelpersArgsExpressions.constPolsIds;
        if (cHelpersArgsExpressions.cmPolsIds) delete[] cHelpersArgsExpressions.cmPolsIds;
        if (cHelpersArgsExpressions.challengesIds) delete[] cHelpersArgsExpressions.challengesIds;
        if (cHelpersArgsExpressions.publicsIds) delete[] cHelpersArgsExpressions.publicsIds;
        if (cHelpersArgsExpressions.subproofValuesIds) delete[] cHelpersArgsExpressions.subproofValuesIds;
        if (cHelpersArgsExpressions.cmPolsCalculatedIds) delete[] cHelpersArgsExpressions.cmPolsCalculatedIds;

        if (cHelpersArgsDebug.ops) delete[] cHelpersArgsDebug.ops;
        if (cHelpersArgsDebug.args) delete[] cHelpersArgsDebug.args;
        if (cHelpersArgsDebug.numbers) delete[] cHelpersArgsDebug.numbers;
        if (cHelpersArgsDebug.constPolsIds) delete[] cHelpersArgsDebug.constPolsIds;
        if (cHelpersArgsDebug.cmPolsIds) delete[] cHelpersArgsDebug.cmPolsIds;
        if (cHelpersArgsDebug.challengesIds) delete[] cHelpersArgsDebug.challengesIds;
        if (cHelpersArgsDebug.publicsIds) delete[] cHelpersArgsDebug.publicsIds;
        if (cHelpersArgsDebug.subproofValuesIds) delete[] cHelpersArgsDebug.subproofValuesIds;
    };

    /* Constructor */
    CHelpers(string file);

    void loadCHelpers(BinFileUtils::BinFile *cHelpersBin);
};


#endif
