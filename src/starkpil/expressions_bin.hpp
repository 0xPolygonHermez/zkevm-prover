#ifndef BINARY_HPP
#define BINARY_HPP

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

const int BINARY_IMPOLS_SECTION = 2;
const int BINARY_EXPRESSIONS_SECTION = 3;
const int BINARY_CONSTRAINTS_SECTION = 4;
const int BINARY_HINTS_SECTION = 5;

struct ParserParams
{
    uint32_t stage;
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
    uint32_t firstRow;
    uint32_t lastRow;
    uint32_t destDim;
    uint32_t destId;
    string line;
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
};

class ExpressionsBin
{
public:
    std::map<uint64_t, ParserParams> expressionsInfo;

    std::vector<ParserParams> constraintsInfoDebug;

    std::vector<ParserParams> imPolsInfo;

    std::vector<Hint> hints;

    ParserArgs expressionsBinArgsConstraints;

    ParserArgs expressionsBinArgsImPols;
    
    ParserArgs expressionsBinArgsExpressions;

    ~ExpressionsBin() {
        if (expressionsBinArgsImPols.ops) delete[] expressionsBinArgsImPols.ops;
        if (expressionsBinArgsImPols.args) delete[] expressionsBinArgsImPols.args;
        if (expressionsBinArgsImPols.numbers) delete[] expressionsBinArgsImPols.numbers;
        if (expressionsBinArgsImPols.constPolsIds) delete[] expressionsBinArgsImPols.constPolsIds;
        if (expressionsBinArgsImPols.cmPolsIds) delete[] expressionsBinArgsImPols.cmPolsIds;
        if (expressionsBinArgsImPols.challengesIds) delete[] expressionsBinArgsImPols.challengesIds;
        if (expressionsBinArgsImPols.publicsIds) delete[] expressionsBinArgsImPols.publicsIds;
        if (expressionsBinArgsImPols.subproofValuesIds) delete[] expressionsBinArgsImPols.subproofValuesIds;

        if (expressionsBinArgsExpressions.ops) delete[] expressionsBinArgsExpressions.ops;
        if (expressionsBinArgsExpressions.args) delete[] expressionsBinArgsExpressions.args;
        if (expressionsBinArgsExpressions.numbers) delete[] expressionsBinArgsExpressions.numbers;
        if (expressionsBinArgsExpressions.constPolsIds) delete[] expressionsBinArgsExpressions.constPolsIds;
        if (expressionsBinArgsExpressions.cmPolsIds) delete[] expressionsBinArgsExpressions.cmPolsIds;
        if (expressionsBinArgsExpressions.challengesIds) delete[] expressionsBinArgsExpressions.challengesIds;
        if (expressionsBinArgsExpressions.publicsIds) delete[] expressionsBinArgsExpressions.publicsIds;
        if (expressionsBinArgsExpressions.subproofValuesIds) delete[] expressionsBinArgsExpressions.subproofValuesIds;

        if (expressionsBinArgsConstraints.ops) delete[] expressionsBinArgsConstraints.ops;
        if (expressionsBinArgsConstraints.args) delete[] expressionsBinArgsConstraints.args;
        if (expressionsBinArgsConstraints.numbers) delete[] expressionsBinArgsConstraints.numbers;
        if (expressionsBinArgsConstraints.constPolsIds) delete[] expressionsBinArgsConstraints.constPolsIds;
        if (expressionsBinArgsConstraints.cmPolsIds) delete[] expressionsBinArgsConstraints.cmPolsIds;
        if (expressionsBinArgsConstraints.challengesIds) delete[] expressionsBinArgsConstraints.challengesIds;
        if (expressionsBinArgsConstraints.publicsIds) delete[] expressionsBinArgsConstraints.publicsIds;
        if (expressionsBinArgsConstraints.subproofValuesIds) delete[] expressionsBinArgsConstraints.subproofValuesIds;
    };

    /* Constructor */
    ExpressionsBin(string file);

    void loadExpressionsBin(BinFileUtils::BinFile *expressionsBin);
};


#endif
