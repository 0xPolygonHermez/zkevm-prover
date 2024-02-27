#include "chelpers.hpp"

void CHelpers::loadCHelpers(BinFileUtils::BinFile *cHelpersBin, bool pil2) {

    cHelpersBin->startReadSection(CHELPERS_HEADER_SECTION);

    uint32_t nOps = cHelpersBin->readU32LE();
    uint32_t nArgs = cHelpersBin->readU32LE();
    uint32_t nNumbers = cHelpersBin->readU32LE();

    cHelpersArgs.ops = new uint8_t[nOps];
    cHelpersArgs.args = new uint16_t[nArgs];
    cHelpersArgs.numbers = new uint64_t[nNumbers];

    cHelpersBin->endReadSection();


    cHelpersBin->startReadSection(CHELPERS_STAGES_SECTION);

    uint64_t nStages = cHelpersBin->readU32LE();

    for(uint64_t i = 0; i < nStages; ++i) {
        ParserParams parserParamsStage;

        uint32_t stage = cHelpersBin->readU32LE();
        parserParamsStage.stage = stage;
        
        std::string stageName = "step" + std::to_string(stage);

        if(!pil2) {
            parserParamsStage.executeBefore = cHelpersBin->readU32LE();
            if(parserParamsStage.executeBefore == 0) stageName += "_after";
        }
      
        parserParamsStage.nTemp1 = cHelpersBin->readU32LE();
        parserParamsStage.nTemp3 = cHelpersBin->readU32LE();

        parserParamsStage.nOps = cHelpersBin->readU32LE();
        parserParamsStage.opsOffset = cHelpersBin->readU32LE();

        parserParamsStage.nArgs = cHelpersBin->readU32LE();
        parserParamsStage.argsOffset = cHelpersBin->readU32LE();

        parserParamsStage.nNumbers = cHelpersBin->readU32LE();
        parserParamsStage.numbersOffset = cHelpersBin->readU32LE();

        stagesInfo[stageName] = parserParamsStage;
    }
    
    cHelpersBin->endReadSection();

    if(pil2) {
        cHelpersBin->startReadSection(CHELPERS_EXPRESSIONS_PIL2_SECTION);

        uint64_t nExpressions = cHelpersBin->readU32LE();

        for(uint64_t i = 0; i < nExpressions; ++i) {
            ParserParams parserParamsExpression;

            uint32_t expId = cHelpersBin->readU32LE();
            parserParamsExpression.expId = expId;
            parserParamsExpression.stage = cHelpersBin->readU32LE();

            parserParamsExpression.nTemp1 = cHelpersBin->readU32LE();
            parserParamsExpression.nTemp3 = cHelpersBin->readU32LE();

            parserParamsExpression.nOps = cHelpersBin->readU32LE();
            parserParamsExpression.opsOffset = cHelpersBin->readU32LE();

            parserParamsExpression.nArgs = cHelpersBin->readU32LE();
            parserParamsExpression.argsOffset = cHelpersBin->readU32LE();

            parserParamsExpression.nNumbers = cHelpersBin->readU32LE();
            parserParamsExpression.numbersOffset = cHelpersBin->readU32LE();

            expressionsInfo[expId] = parserParamsExpression;
        }

        cHelpersBin->endReadSection();
    }

    if(pil2) {
        cHelpersBin->startReadSection(CHELPERS_BUFFERS_PIL2_SECTION);
    } else {
        cHelpersBin->startReadSection(CHELPERS_BUFFERS_PIL1_SECTION);
    }

    for(uint64_t j = 0; j < nOps; ++j) {
        cHelpersArgs.ops[j] = cHelpersBin->readU8LE();
    }
    for(uint64_t j = 0; j < nArgs; ++j) {
        cHelpersArgs.args[j] = cHelpersBin->readU16LE();
    }
    for(uint64_t j = 0; j < nNumbers; ++j) {
        cHelpersArgs.numbers[j] = cHelpersBin->readU64LE();
    }

    cHelpersBin->endReadSection();
};