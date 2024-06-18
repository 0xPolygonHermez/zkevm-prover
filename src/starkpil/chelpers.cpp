#include "chelpers.hpp"

void CHelpers::loadCHelpers(BinFileUtils::BinFile *cHelpersBin) {

    cHelpersBin->startReadSection(CHELPERS_HEADER_SECTION);

    uint32_t nOps = cHelpersBin->readU32LE();
    uint32_t nArgs = cHelpersBin->readU32LE();
    uint32_t nNumbers = cHelpersBin->readU32LE();
    uint32_t nStorePols = cHelpersBin->readU32LE();

    cHelpersArgs.ops = new uint8_t[nOps];
    cHelpersArgs.args = new uint16_t[nArgs];
    cHelpersArgs.numbers = new uint64_t[nNumbers];
    cHelpersArgs.storePols = new uint8_t[nStorePols];

    cHelpersBin->endReadSection();


    cHelpersBin->startReadSection(CHELPERS_STAGES_SECTION);

    uint64_t nStages = cHelpersBin->readU32LE();

    for(uint64_t i = 0; i < nStages; ++i) {
        ParserParams parserParamsStage;

        uint32_t stage = cHelpersBin->readU32LE();
        uint32_t executeBefore = cHelpersBin->readU32LE();

        parserParamsStage.stage = stage;
        parserParamsStage.executeBefore = executeBefore;
        
        std::string stageName = "step" + std::to_string(stage);
        if(executeBefore == 0) stageName += "_after";

        parserParamsStage.nTemp1 = cHelpersBin->readU32LE();
        parserParamsStage.nTemp3 = cHelpersBin->readU32LE();

        parserParamsStage.nOps = cHelpersBin->readU32LE();
        parserParamsStage.opsOffset = cHelpersBin->readU32LE();

        parserParamsStage.nArgs = cHelpersBin->readU32LE();
        parserParamsStage.argsOffset = cHelpersBin->readU32LE();

        parserParamsStage.nNumbers = cHelpersBin->readU32LE();
        parserParamsStage.numbersOffset = cHelpersBin->readU32LE();

        parserParamsStage.nStorePols = cHelpersBin->readU32LE();
        parserParamsStage.storePolsOffset = cHelpersBin->readU32LE();

        stagesInfo[stageName] = parserParamsStage;
    }
    
    cHelpersBin->endReadSection();

    cHelpersBin->startReadSection(CHELPERS_BUFFERS_SECTION);

    for(uint64_t j = 0; j < nOps; ++j) {
        cHelpersArgs.ops[j] = cHelpersBin->readU8LE();
    }
    for(uint64_t j = 0; j < nArgs; ++j) {
        cHelpersArgs.args[j] = cHelpersBin->readU16LE();
    }
    for(uint64_t j = 0; j < nNumbers; ++j) {
        cHelpersArgs.numbers[j] = cHelpersBin->readU64LE();
    }

    for(uint64_t j = 0; j < nStorePols; ++j) {
        cHelpersArgs.storePols[j] = cHelpersBin->readU8LE();
    }

    cHelpersBin->endReadSection();
};