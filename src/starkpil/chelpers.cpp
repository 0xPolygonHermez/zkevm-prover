#include "chelpers.hpp"

void CHelpers::loadCHelpersPil1(BinFileUtils::BinFile *cHelpersBin) {
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

        parserParamsStage.executeBefore = cHelpersBin->readU32LE();
        if(parserParamsStage.executeBefore == 0) stageName += "_after";
      
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

    cHelpersBin->endReadSection();
}


void CHelpers::loadCHelpersPil2(BinFileUtils::BinFile *cHelpersBin) {

    cHelpersBin->startReadSection(CHELPERS_STAGES_PIL2_SECTION);

    uint32_t nOps = cHelpersBin->readU32LE();
    uint32_t nArgs = cHelpersBin->readU32LE();
    uint32_t nNumbers = cHelpersBin->readU32LE();
    uint32_t nConstPolsIds = cHelpersBin->readU32LE();
    uint32_t nCmPolsIds = cHelpersBin->readU32LE();
    uint32_t nChallengesIds = cHelpersBin->readU32LE();
    uint32_t nPublicsIds = cHelpersBin->readU32LE();
    uint32_t nSubproofValuesIds = cHelpersBin->readU32LE();

    cHelpersArgs.ops = new uint8_t[nOps];
    cHelpersArgs.args = new uint16_t[nArgs];
    cHelpersArgs.numbers = new uint64_t[nNumbers];
    cHelpersArgs.constPolsIds = new uint16_t[nConstPolsIds];
    cHelpersArgs.cmPolsIds = new uint16_t[nCmPolsIds];
    cHelpersArgs.challengesIds = new uint16_t[nChallengesIds];
    cHelpersArgs.publicsIds = new uint16_t[nPublicsIds];
    cHelpersArgs.subproofValuesIds = new uint16_t[nSubproofValuesIds];

    uint64_t nStages = cHelpersBin->readU32LE();

    for(uint64_t i = 0; i < nStages; ++i) {
        ParserParams parserParamsStage;

        uint32_t stage = cHelpersBin->readU32LE();
        parserParamsStage.stage = stage;
        
        std::string stageName = "step" + std::to_string(stage);
   
        parserParamsStage.nTemp1 = cHelpersBin->readU32LE();
        parserParamsStage.nTemp3 = cHelpersBin->readU32LE();

        parserParamsStage.nOps = cHelpersBin->readU32LE();
        parserParamsStage.opsOffset = cHelpersBin->readU32LE();

        parserParamsStage.nArgs = cHelpersBin->readU32LE();
        parserParamsStage.argsOffset = cHelpersBin->readU32LE();

        parserParamsStage.nNumbers = cHelpersBin->readU32LE();
        parserParamsStage.numbersOffset = cHelpersBin->readU32LE();
        
        parserParamsStage.nConstPolsUsed = cHelpersBin->readU32LE();
        parserParamsStage.constPolsOffset = cHelpersBin->readU32LE();
        
        parserParamsStage.nCmPolsUsed = cHelpersBin->readU32LE();
        parserParamsStage.cmPolsOffset = cHelpersBin->readU32LE();

        parserParamsStage.nChallengesUsed = cHelpersBin->readU32LE();
        parserParamsStage.challengesOffset = cHelpersBin->readU32LE();

        parserParamsStage.nPublicsUsed = cHelpersBin->readU32LE();
        parserParamsStage.publicsOffset = cHelpersBin->readU32LE();

        parserParamsStage.nSubproofValuesUsed = cHelpersBin->readU32LE();
        parserParamsStage.subproofValuesOffset = cHelpersBin->readU32LE();

        stagesInfo[stageName] = parserParamsStage;
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

    for(uint64_t j = 0; j < nConstPolsIds; ++j) {
        cHelpersArgs.constPolsIds[j] = cHelpersBin->readU16LE();
    }

    for(uint64_t j = 0; j < nCmPolsIds; ++j) {
        cHelpersArgs.cmPolsIds[j] = cHelpersBin->readU16LE();
    }

    for(uint64_t j = 0; j < nChallengesIds; ++j) {
        cHelpersArgs.challengesIds[j] = cHelpersBin->readU16LE();
    }

    for(uint64_t j = 0; j < nPublicsIds; ++j) {
        cHelpersArgs.publicsIds[j] = cHelpersBin->readU16LE();
    }

    for(uint64_t j = 0; j < nSubproofValuesIds; ++j) {
        cHelpersArgs.subproofValuesIds[j] = cHelpersBin->readU16LE();
    }
    
    cHelpersBin->endReadSection();
     
    cHelpersBin->startReadSection(CHELPERS_EXPRESSIONS_PIL2_SECTION);

    uint32_t nOpsExpressions = cHelpersBin->readU32LE();
    uint32_t nArgsExpressions = cHelpersBin->readU32LE();
    uint32_t nNumbersExpressions = cHelpersBin->readU32LE();
    uint32_t nConstPolsIdsExpressions = cHelpersBin->readU32LE();
    uint32_t nCmPolsIdsExpressions = cHelpersBin->readU32LE();
    uint32_t nChallengesIdsExpressions = cHelpersBin->readU32LE();
    uint32_t nPublicsIdsExpressions = cHelpersBin->readU32LE();
    uint32_t nSubproofValuesIdsExpressions = cHelpersBin->readU32LE();

    cHelpersArgsExpressions.ops = new uint8_t[nOpsExpressions];
    cHelpersArgsExpressions.args = new uint16_t[nArgsExpressions];
    cHelpersArgsExpressions.numbers = new uint64_t[nNumbersExpressions];
    cHelpersArgsExpressions.constPolsIds = new uint16_t[nConstPolsIdsExpressions];
    cHelpersArgsExpressions.cmPolsIds = new uint16_t[nCmPolsIdsExpressions];
    cHelpersArgsExpressions.challengesIds = new uint16_t[nChallengesIdsExpressions];
    cHelpersArgsExpressions.publicsIds = new uint16_t[nPublicsIdsExpressions];
    cHelpersArgsExpressions.subproofValuesIds = new uint16_t[nSubproofValuesIdsExpressions];

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

        parserParamsExpression.nConstPolsUsed = cHelpersBin->readU32LE();
        parserParamsExpression.constPolsOffset = cHelpersBin->readU32LE();

        parserParamsExpression.nCmPolsUsed = cHelpersBin->readU32LE();
        parserParamsExpression.cmPolsOffset = cHelpersBin->readU32LE();

        parserParamsExpression.nChallengesUsed = cHelpersBin->readU32LE();
        parserParamsExpression.challengesOffset = cHelpersBin->readU32LE();

        parserParamsExpression.nPublicsUsed = cHelpersBin->readU32LE();
        parserParamsExpression.publicsOffset = cHelpersBin->readU32LE();

        parserParamsExpression.nSubproofValuesUsed = cHelpersBin->readU32LE();
        parserParamsExpression.subproofValuesOffset = cHelpersBin->readU32LE();

        expressionsInfo[expId] = parserParamsExpression;
    }

    for(uint64_t j = 0; j < nOpsExpressions; ++j) {
        cHelpersArgsExpressions.ops[j] = cHelpersBin->readU8LE();
    }
    for(uint64_t j = 0; j < nArgsExpressions; ++j) {
        cHelpersArgsExpressions.args[j] = cHelpersBin->readU16LE();
    }
    for(uint64_t j = 0; j < nNumbersExpressions; ++j) {
        cHelpersArgsExpressions.numbers[j] = cHelpersBin->readU64LE();
    }

    for(uint64_t j = 0; j < nConstPolsIdsExpressions; ++j) {
        cHelpersArgsExpressions.constPolsIds[j] = cHelpersBin->readU16LE();
    }

    for(uint64_t j = 0; j < nCmPolsIdsExpressions; ++j) {
        cHelpersArgsExpressions.cmPolsIds[j] = cHelpersBin->readU16LE();
    }

    for(uint64_t j = 0; j < nChallengesIdsExpressions; ++j) {
        cHelpersArgsExpressions.challengesIds[j] = cHelpersBin->readU16LE();
    }

    for(uint64_t j = 0; j < nPublicsIdsExpressions; ++j) {
        cHelpersArgsExpressions.publicsIds[j] = cHelpersBin->readU16LE();
    }

    for(uint64_t j = 0; j < nSubproofValuesIdsExpressions; ++j) {
        cHelpersArgsExpressions.subproofValuesIds[j] = cHelpersBin->readU16LE();
    }

    cHelpersBin->endReadSection();

    cHelpersBin->startReadSection(CHELPERS_CONSTRAINTS_PIL2_SECTION);

    uint32_t nOpsDebug = cHelpersBin->readU32LE();
    uint32_t nArgsDebug = cHelpersBin->readU32LE();
    uint32_t nNumbersDebug = cHelpersBin->readU32LE();
    uint32_t nConstPolsIdsDebug = cHelpersBin->readU32LE();
    uint32_t nCmPolsIdsDebug = cHelpersBin->readU32LE();
    uint32_t nChallengesIdsDebug = cHelpersBin->readU32LE();
    uint32_t nPublicsIdsDebug = cHelpersBin->readU32LE();
    uint32_t nSubproofValuesIdsDebug = cHelpersBin->readU32LE();

    cHelpersArgsDebug.ops = new uint8_t[nOpsDebug];
    cHelpersArgsDebug.args = new uint16_t[nArgsDebug];
    cHelpersArgsDebug.numbers = new uint64_t[nNumbersDebug];
    cHelpersArgsDebug.constPolsIds = new uint16_t[nConstPolsIdsDebug];
    cHelpersArgsDebug.cmPolsIds = new uint16_t[nCmPolsIdsDebug];
    cHelpersArgsDebug.challengesIds = new uint16_t[nChallengesIdsDebug];
    cHelpersArgsDebug.publicsIds = new uint16_t[nPublicsIdsDebug];
    cHelpersArgsDebug.subproofValuesIds = new uint16_t[nSubproofValuesIdsDebug];

    constraintsInfoDebug.resize(nStages);
    
    uint32_t nConstraints = cHelpersBin->readU32LE();

    for(uint64_t i = 0; i < nConstraints; ++i) {
        ParserParams parserParamsConstraint;

        uint32_t stage = cHelpersBin->readU32LE();
        parserParamsConstraint.stage = stage;

        parserParamsConstraint.destDim = cHelpersBin->readU32LE();
        parserParamsConstraint.destId = cHelpersBin->readU32LE();

        parserParamsConstraint.nTemp1 = cHelpersBin->readU32LE();
        parserParamsConstraint.nTemp3 = cHelpersBin->readU32LE();

        parserParamsConstraint.nOps = cHelpersBin->readU32LE();
        parserParamsConstraint.opsOffset = cHelpersBin->readU32LE();

        parserParamsConstraint.nArgs = cHelpersBin->readU32LE();
        parserParamsConstraint.argsOffset = cHelpersBin->readU32LE();

        parserParamsConstraint.nNumbers = cHelpersBin->readU32LE();
        parserParamsConstraint.numbersOffset = cHelpersBin->readU32LE();

        parserParamsConstraint.nConstPolsUsed = cHelpersBin->readU32LE();
        parserParamsConstraint.constPolsOffset = cHelpersBin->readU32LE();

        parserParamsConstraint.nCmPolsUsed = cHelpersBin->readU32LE();
        parserParamsConstraint.cmPolsOffset = cHelpersBin->readU32LE();

        parserParamsConstraint.nChallengesUsed = cHelpersBin->readU32LE();
        parserParamsConstraint.challengesOffset = cHelpersBin->readU32LE();

        parserParamsConstraint.nPublicsUsed = cHelpersBin->readU32LE();
        parserParamsConstraint.publicsOffset = cHelpersBin->readU32LE();

        parserParamsConstraint.nSubproofValuesUsed = cHelpersBin->readU32LE();
        parserParamsConstraint.subproofValuesOffset = cHelpersBin->readU32LE();

        if(constraintsInfoDebug[stage].empty()) {
            constraintsInfoDebug[stage].emplace_back();
        }

        constraintsInfoDebug[stage].push_back(parserParamsConstraint);
    }


    for(uint64_t j = 0; j < nOpsDebug; ++j) {
        cHelpersArgsDebug.ops[j] = cHelpersBin->readU8LE();
    }
    for(uint64_t j = 0; j < nArgsDebug; ++j) {
        cHelpersArgsDebug.args[j] = cHelpersBin->readU16LE();
    }
    for(uint64_t j = 0; j < nNumbersDebug; ++j) {
        cHelpersArgsDebug.numbers[j] = cHelpersBin->readU64LE();
    }

    for(uint64_t j = 0; j < nConstPolsIdsDebug; ++j) {
        cHelpersArgsDebug.constPolsIds[j] = cHelpersBin->readU16LE();
    }

    for(uint64_t j = 0; j < nCmPolsIdsDebug; ++j) {
        cHelpersArgsDebug.cmPolsIds[j] = cHelpersBin->readU16LE();
    }

    for(uint64_t j = 0; j < nChallengesIdsDebug; ++j) {
        cHelpersArgsDebug.challengesIds[j] = cHelpersBin->readU16LE();
    }

    for(uint64_t j = 0; j < nPublicsIdsDebug; ++j) {
        cHelpersArgsDebug.publicsIds[j] = cHelpersBin->readU16LE();
    }

    for(uint64_t j = 0; j < nSubproofValuesIdsDebug; ++j) {
        cHelpersArgsDebug.subproofValuesIds[j] = cHelpersBin->readU16LE();
    }

    cHelpersBin->endReadSection();
}

void CHelpers::loadCHelpers(BinFileUtils::BinFile *cHelpersBin, bool pil2_) {
    pil2 = pil2_;
    if(pil2) {
        loadCHelpersPil2(cHelpersBin);
    } else {
        loadCHelpersPil1(cHelpersBin);
    }
};