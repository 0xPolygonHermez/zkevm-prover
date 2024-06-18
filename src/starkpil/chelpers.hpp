#ifndef CHELPERS_HPP
#define CHELPERS_HPP

#include <string>
#include <map>
#include "binfile_utils.hpp"
#include "polinomial.hpp"
#include "goldilocks_base_field.hpp"
#include "goldilocks_base_field_avx.hpp"
#include "goldilocks_base_field_avx512.hpp"
#include "goldilocks_base_field_pack.hpp"
#include "goldilocks_cubic_extension.hpp"
#include "goldilocks_cubic_extension_pack.hpp"
#include "goldilocks_cubic_extension_avx.hpp"
#include "goldilocks_cubic_extension_avx512.hpp"

#include "constant_pols_starks.hpp"
#include "stark_info.hpp"
#include <immintrin.h>
#include <cassert>


const int CHELPERS_HEADER_SECTION = 2;
const int CHELPERS_STAGES_SECTION = 3;
const int CHELPERS_BUFFERS_SECTION = 4;

struct ParserParams 
{
    uint32_t stage;
    uint32_t executeBefore;
    uint32_t nTemp1;
    uint32_t nTemp3;
    uint32_t nOps;
    uint32_t opsOffset;
    uint32_t nArgs;
    uint32_t argsOffset;
    uint32_t nNumbers;
    uint64_t numbersOffset;
    uint32_t nStorePols;
    uint32_t storePolsOffset;
};

struct ParserArgs 
{
    uint8_t* ops;
    uint16_t* args;
    uint64_t* numbers;
    uint8_t* storePols;
};

class CHelpers
{
public:
    std::map<std::string, ParserParams> stagesInfo;
    
    ParserArgs cHelpersArgs;

    ~CHelpers() {
        delete[] cHelpersArgs.ops;
        delete[] cHelpersArgs.args;
        delete[] cHelpersArgs.storePols;
        delete[] cHelpersArgs.numbers;

        stagesInfo.clear();
    };

    void loadCHelpers(BinFileUtils::BinFile *cHelpersBin);
};


#endif
