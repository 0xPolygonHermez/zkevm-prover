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


const int CHELPERS_HEADER_SECTION = 2;
const int CHELPERS_STAGES_SECTION = 3;

const int CHELPERS_BUFFERS_PIL1_SECTION = 4;

const int CHELPERS_EXPRESSIONS_PIL2_SECTION = 4;
const int CHELPERS_BUFFERS_PIL2_SECTION = 5;
const int CHELPERS_SYMBOLS_PIL2_SECTION = 6;
const int CHELPERS_CONSTRAINTS_DEBUG_PIL2_SECTION = 7;


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
};

class CHelpers
{
public:
    std::map<string, ParserParams> stagesInfo;
    std::map<uint64_t, ParserParams> expressionsInfo;

    std::vector<std::vector<ParserParams>> constraintsInfoDebug;
    
    ParserArgs cHelpersArgs;

    ParserArgs cHelpersArgsDebug;

    ~CHelpers() {
        delete[] cHelpersArgs.ops;
        delete[] cHelpersArgs.args;
        delete[] cHelpersArgs.numbers;
        delete[] cHelpersArgs.constPolsIds;
        delete[] cHelpersArgs.cmPolsIds;

        delete[] cHelpersArgsDebug.ops;
        delete[] cHelpersArgsDebug.args;
        delete[] cHelpersArgsDebug.numbers;
        delete[] cHelpersArgsDebug.constPolsIds;
        delete[] cHelpersArgsDebug.cmPolsIds;

        stagesInfo.clear();
    };

    void loadCHelpers(BinFileUtils::BinFile *cHelpersBin, bool pil2);
};


#endif
