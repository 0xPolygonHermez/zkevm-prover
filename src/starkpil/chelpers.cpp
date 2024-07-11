#include "chelpers.hpp"

CHelpers::CHelpers(string file) {
    std::unique_ptr<BinFileUtils::BinFile> cHelpersBinFile = BinFileUtils::openExisting(file, "chps", 1);
    loadCHelpers(cHelpersBinFile.get());
}

void CHelpers::loadCHelpers(BinFileUtils::BinFile *cHelpersBin) {

    cHelpersBin->startReadSection(CHELPERS_STAGES_SECTION);

    uint32_t nOps = cHelpersBin->readU32LE();
    uint32_t nArgs = cHelpersBin->readU32LE();
    uint32_t nNumbers = cHelpersBin->readU32LE();
    uint32_t nConstPolsIds = cHelpersBin->readU32LE();
    uint32_t nCmPolsIds = cHelpersBin->readU32LE();
    uint32_t nChallengesIds = cHelpersBin->readU32LE();
    uint32_t nPublicsIds = cHelpersBin->readU32LE();
    uint32_t nSubproofValuesIds = cHelpersBin->readU32LE();
    uint32_t nCmPolsCalculatedIds = cHelpersBin->readU32LE();

    cHelpersArgs.ops = new uint8_t[nOps];
    cHelpersArgs.args = new uint16_t[nArgs];
    cHelpersArgs.numbers = new uint64_t[nNumbers];
    cHelpersArgs.constPolsIds = new uint16_t[nConstPolsIds];
    cHelpersArgs.cmPolsIds = new uint16_t[nCmPolsIds];
    cHelpersArgs.challengesIds = new uint16_t[nChallengesIds];
    cHelpersArgs.publicsIds = new uint16_t[nPublicsIds];
    cHelpersArgs.subproofValuesIds = new uint16_t[nSubproofValuesIds];
    cHelpersArgs.cmPolsCalculatedIds = new uint16_t[nCmPolsCalculatedIds];

    uint64_t nStages = cHelpersBin->readU32LE();

    for(uint64_t i = 0; i < nStages; ++i) {
        ParserParams parserParamsStage;

        uint32_t stage = cHelpersBin->readU32LE();
        parserParamsStage.stage = stage;
           
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

        parserParamsStage.nCmPolsCalculated = cHelpersBin->readU32LE();
        parserParamsStage.cmPolsCalculatedOffset = cHelpersBin->readU32LE();

        stagesInfo.push_back(parserParamsStage);
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

    for(uint64_t j = 0; j < nCmPolsCalculatedIds; ++j) {
        cHelpersArgs.cmPolsCalculatedIds[j] = cHelpersBin->readU16LE();
    }
    
    cHelpersBin->endReadSection();
     
    cHelpersBin->startReadSection(CHELPERS_IMPOLS_SECTION);

    uint32_t nOpsImPols = cHelpersBin->readU32LE();
    uint32_t nArgsImPols = cHelpersBin->readU32LE();
    uint32_t nNumbersImPols = cHelpersBin->readU32LE();
    uint32_t nConstPolsIdsImPols = cHelpersBin->readU32LE();
    uint32_t nCmPolsIdsImPols = cHelpersBin->readU32LE();
    uint32_t nChallengesIdsImPols = cHelpersBin->readU32LE();
    uint32_t nPublicsIdsImPols = cHelpersBin->readU32LE();
    uint32_t nSubproofValuesIdsImPols = cHelpersBin->readU32LE();
    uint32_t nCmPolsCalculatedIdsImPols = cHelpersBin->readU32LE();

    cHelpersArgsImPols.ops = new uint8_t[nOpsImPols];
    cHelpersArgsImPols.args = new uint16_t[nArgsImPols];
    cHelpersArgsImPols.numbers = new uint64_t[nNumbersImPols];
    cHelpersArgsImPols.constPolsIds = new uint16_t[nConstPolsIdsImPols];
    cHelpersArgsImPols.cmPolsIds = new uint16_t[nCmPolsIdsImPols];
    cHelpersArgsImPols.challengesIds = new uint16_t[nChallengesIdsImPols];
    cHelpersArgsImPols.publicsIds = new uint16_t[nPublicsIdsImPols];
    cHelpersArgsImPols.subproofValuesIds = new uint16_t[nSubproofValuesIdsImPols];
    cHelpersArgsImPols.cmPolsCalculatedIds = new uint16_t[nCmPolsCalculatedIdsImPols];
           
    imPolsInfo.destDim = 0;
    imPolsInfo.nTemp1 = cHelpersBin->readU32LE();
    imPolsInfo.nTemp3 = cHelpersBin->readU32LE();

    imPolsInfo.nOps = cHelpersBin->readU32LE();
    imPolsInfo.opsOffset = cHelpersBin->readU32LE();

    imPolsInfo.nArgs = cHelpersBin->readU32LE();
    imPolsInfo.argsOffset = cHelpersBin->readU32LE();

    imPolsInfo.nNumbers = cHelpersBin->readU32LE();
    imPolsInfo.numbersOffset = cHelpersBin->readU32LE();
    
    imPolsInfo.nConstPolsUsed = cHelpersBin->readU32LE();
    imPolsInfo.constPolsOffset = cHelpersBin->readU32LE();
    
    imPolsInfo.nCmPolsUsed = cHelpersBin->readU32LE();
    imPolsInfo.cmPolsOffset = cHelpersBin->readU32LE();

    imPolsInfo.nChallengesUsed = cHelpersBin->readU32LE();
    imPolsInfo.challengesOffset = cHelpersBin->readU32LE();

    imPolsInfo.nPublicsUsed = cHelpersBin->readU32LE();
    imPolsInfo.publicsOffset = cHelpersBin->readU32LE();

    imPolsInfo.nSubproofValuesUsed = cHelpersBin->readU32LE();
    imPolsInfo.subproofValuesOffset = cHelpersBin->readU32LE();

    imPolsInfo.nCmPolsCalculated = cHelpersBin->readU32LE();
    imPolsInfo.cmPolsCalculatedOffset = cHelpersBin->readU32LE();

    for(uint64_t j = 0; j < nOpsImPols; ++j) {
        cHelpersArgsImPols.ops[j] = cHelpersBin->readU8LE();
    }
    for(uint64_t j = 0; j < nArgsImPols; ++j) {
        cHelpersArgsImPols.args[j] = cHelpersBin->readU16LE();
    }
    for(uint64_t j = 0; j < nNumbersImPols; ++j) {
        cHelpersArgsImPols.numbers[j] = cHelpersBin->readU64LE();
    }

    for(uint64_t j = 0; j < nConstPolsIdsImPols; ++j) {
        cHelpersArgsImPols.constPolsIds[j] = cHelpersBin->readU16LE();
    }

    for(uint64_t j = 0; j < nCmPolsIdsImPols; ++j) {
        cHelpersArgsImPols.cmPolsIds[j] = cHelpersBin->readU16LE();
    }

    for(uint64_t j = 0; j < nChallengesIdsImPols; ++j) {
        cHelpersArgsImPols.challengesIds[j] = cHelpersBin->readU16LE();
    }

    for(uint64_t j = 0; j < nPublicsIdsImPols; ++j) {
        cHelpersArgsImPols.publicsIds[j] = cHelpersBin->readU16LE();
    }

    for(uint64_t j = 0; j < nSubproofValuesIdsImPols; ++j) {
        cHelpersArgsImPols.subproofValuesIds[j] = cHelpersBin->readU16LE();
    }

    for(uint64_t j = 0; j < nCmPolsCalculatedIdsImPols; ++j) {
        cHelpersArgsImPols.cmPolsCalculatedIds[j] = cHelpersBin->readU16LE();
    }
    
    cHelpersBin->endReadSection();

    cHelpersBin->startReadSection(CHELPERS_EXPRESSIONS_SECTION);

    uint32_t nOpsExpressions = cHelpersBin->readU32LE();
    uint32_t nArgsExpressions = cHelpersBin->readU32LE();
    uint32_t nNumbersExpressions = cHelpersBin->readU32LE();
    uint32_t nConstPolsIdsExpressions = cHelpersBin->readU32LE();
    uint32_t nCmPolsIdsExpressions = cHelpersBin->readU32LE();
    uint32_t nChallengesIdsExpressions = cHelpersBin->readU32LE();
    uint32_t nPublicsIdsExpressions = cHelpersBin->readU32LE();
    uint32_t nSubproofValuesIdsExpressions = cHelpersBin->readU32LE();
    uint32_t nCmPolsCalculatedIdsExpressions = cHelpersBin->readU32LE();

    cHelpersArgsExpressions.ops = new uint8_t[nOpsExpressions];
    cHelpersArgsExpressions.args = new uint16_t[nArgsExpressions];
    cHelpersArgsExpressions.numbers = new uint64_t[nNumbersExpressions];
    cHelpersArgsExpressions.constPolsIds = new uint16_t[nConstPolsIdsExpressions];
    cHelpersArgsExpressions.cmPolsIds = new uint16_t[nCmPolsIdsExpressions];
    cHelpersArgsExpressions.challengesIds = new uint16_t[nChallengesIdsExpressions];
    cHelpersArgsExpressions.publicsIds = new uint16_t[nPublicsIdsExpressions];
    cHelpersArgsExpressions.subproofValuesIds = new uint16_t[nSubproofValuesIdsExpressions];
    cHelpersArgsExpressions.cmPolsCalculatedIds = new uint16_t[nCmPolsCalculatedIdsExpressions];

    uint64_t nExpressions = cHelpersBin->readU32LE();

    for(uint64_t i = 0; i < nExpressions; ++i) {
        ParserParams parserParamsExpression;

        uint32_t expId = cHelpersBin->readU32LE();
        
        parserParamsExpression.expId = expId;
        parserParamsExpression.destDim = cHelpersBin->readU32LE();
        parserParamsExpression.destId = cHelpersBin->readU32LE();
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

        parserParamsExpression.nCmPolsCalculated = cHelpersBin->readU32LE();
        parserParamsExpression.cmPolsCalculatedOffset = cHelpersBin->readU32LE();

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

    for(uint64_t j = 0; j < nCmPolsCalculatedIdsExpressions; ++j) {
        cHelpersArgsExpressions.cmPolsCalculatedIds[j] = cHelpersBin->readU16LE();
    }

    cHelpersBin->endReadSection();

    cHelpersBin->startReadSection(CHELPERS_CONSTRAINTS_SECTION);

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
    
    uint32_t nConstraints = cHelpersBin->readU32LE();

    for(uint64_t i = 0; i < nConstraints; ++i) {
        ParserParams parserParamsConstraint;

        uint32_t stage = cHelpersBin->readU32LE();
        parserParamsConstraint.stage = stage;

        parserParamsConstraint.destDim = cHelpersBin->readU32LE();
        parserParamsConstraint.destId = cHelpersBin->readU32LE();

        parserParamsConstraint.firstRow = cHelpersBin->readU32LE();
        parserParamsConstraint.lastRow = cHelpersBin->readU32LE();

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

        parserParamsConstraint.nCmPolsCalculated = 0;
    
        constraintsInfoDebug.push_back(parserParamsConstraint);
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

    cHelpersBin->startReadSection(CHELPERS_HINTS_SECTION);

    uint32_t nHints = cHelpersBin->readU32LE();

    for(uint64_t h = 0; h < nHints; h++) {
        Hint hint;
        hint.name = cHelpersBin->readString();

        uint32_t nFields = cHelpersBin->readU32LE();

        for(uint64_t f = 0; f < nFields; f++) {
            HintField hintField;
            std::string name = cHelpersBin->readString();
            std::string operand = cHelpersBin->readString();
            hintField.operand = string2opType(operand);
            if(hintField.operand == opType::number) {
                hintField.value = cHelpersBin->readU32LE();
            } else {
                hintField.id = cHelpersBin->readU32LE();
            }
            if(hintField.operand == opType::tmp) {
                hintField.dim = cHelpersBin->readU32LE();
            }
            hint.fields[name] = hintField;
        }

        hints.push_back(hint);
    }

    cHelpersBin->endReadSection();
}
