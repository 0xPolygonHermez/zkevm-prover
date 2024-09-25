#include "expressions_bin.hpp"

ExpressionsBin::ExpressionsBin(string file) {
    std::unique_ptr<BinFileUtils::BinFile> expressionsBinFile = BinFileUtils::openExisting(file, "chps", 1);
    loadExpressionsBin(expressionsBinFile.get());
}

void ExpressionsBin::loadExpressionsBin(BinFileUtils::BinFile *expressionsBin) {
    expressionsBin->startReadSection(BINARY_IMPOLS_SECTION);

    uint32_t nOpsImPols = expressionsBin->readU32LE();
    uint32_t nArgsImPols = expressionsBin->readU32LE();
    uint32_t nNumbersImPols = expressionsBin->readU32LE();
    uint32_t nConstPolsIdsImPols = expressionsBin->readU32LE();
    uint32_t nCmPolsIdsImPols = expressionsBin->readU32LE();
    uint32_t nChallengesIdsImPols = expressionsBin->readU32LE();
    uint32_t nPublicsIdsImPols = expressionsBin->readU32LE();
    uint32_t nSubproofValuesIdsImPols = expressionsBin->readU32LE();

    expressionsBinArgsImPols.ops = new uint8_t[nOpsImPols];
    expressionsBinArgsImPols.args = new uint16_t[nArgsImPols];
    expressionsBinArgsImPols.numbers = new uint64_t[nNumbersImPols];
    expressionsBinArgsImPols.constPolsIds = new uint16_t[nConstPolsIdsImPols];
    expressionsBinArgsImPols.cmPolsIds = new uint16_t[nCmPolsIdsImPols];
    expressionsBinArgsImPols.challengesIds = new uint16_t[nChallengesIdsImPols];
    expressionsBinArgsImPols.publicsIds = new uint16_t[nPublicsIdsImPols];
    expressionsBinArgsImPols.subproofValuesIds = new uint16_t[nSubproofValuesIdsImPols];
           
    uint32_t nStages = expressionsBin->readU32LE();

    for(uint64_t i = 0; i < nStages; ++i) {
        ParserParams imPolsStageInfo;
        imPolsStageInfo.destDim = 0;
        imPolsStageInfo.nTemp1 = expressionsBin->readU32LE();
        imPolsStageInfo.nTemp3 = expressionsBin->readU32LE();

        imPolsStageInfo.nOps = expressionsBin->readU32LE();
        imPolsStageInfo.opsOffset = expressionsBin->readU32LE();

        imPolsStageInfo.nArgs = expressionsBin->readU32LE();
        imPolsStageInfo.argsOffset = expressionsBin->readU32LE();

        imPolsStageInfo.nNumbers = expressionsBin->readU32LE();
        imPolsStageInfo.numbersOffset = expressionsBin->readU32LE();
        
        imPolsStageInfo.nConstPolsUsed = expressionsBin->readU32LE();
        imPolsStageInfo.constPolsOffset = expressionsBin->readU32LE();
        
        imPolsStageInfo.nCmPolsUsed = expressionsBin->readU32LE();
        imPolsStageInfo.cmPolsOffset = expressionsBin->readU32LE();

        imPolsStageInfo.nChallengesUsed = expressionsBin->readU32LE();
        imPolsStageInfo.challengesOffset = expressionsBin->readU32LE();

        imPolsStageInfo.nPublicsUsed = expressionsBin->readU32LE();
        imPolsStageInfo.publicsOffset = expressionsBin->readU32LE();

        imPolsStageInfo.nSubproofValuesUsed = expressionsBin->readU32LE();
        imPolsStageInfo.subproofValuesOffset = expressionsBin->readU32LE();

        imPolsInfo.push_back(imPolsStageInfo);
    }

    for(uint64_t j = 0; j < nOpsImPols; ++j) {
        expressionsBinArgsImPols.ops[j] = expressionsBin->readU8LE();
    }
    for(uint64_t j = 0; j < nArgsImPols; ++j) {
        expressionsBinArgsImPols.args[j] = expressionsBin->readU16LE();
    }
    for(uint64_t j = 0; j < nNumbersImPols; ++j) {
        expressionsBinArgsImPols.numbers[j] = expressionsBin->readU64LE();
    }

    for(uint64_t j = 0; j < nConstPolsIdsImPols; ++j) {
        expressionsBinArgsImPols.constPolsIds[j] = expressionsBin->readU16LE();
    }

    for(uint64_t j = 0; j < nCmPolsIdsImPols; ++j) {
        expressionsBinArgsImPols.cmPolsIds[j] = expressionsBin->readU16LE();
    }

    for(uint64_t j = 0; j < nChallengesIdsImPols; ++j) {
        expressionsBinArgsImPols.challengesIds[j] = expressionsBin->readU16LE();
    }

    for(uint64_t j = 0; j < nPublicsIdsImPols; ++j) {
        expressionsBinArgsImPols.publicsIds[j] = expressionsBin->readU16LE();
    }

    for(uint64_t j = 0; j < nSubproofValuesIdsImPols; ++j) {
        expressionsBinArgsImPols.subproofValuesIds[j] = expressionsBin->readU16LE();
    }
    
    expressionsBin->endReadSection();

    expressionsBin->startReadSection(BINARY_EXPRESSIONS_SECTION);

    uint32_t nOpsExpressions = expressionsBin->readU32LE();
    uint32_t nArgsExpressions = expressionsBin->readU32LE();
    uint32_t nNumbersExpressions = expressionsBin->readU32LE();
    uint32_t nConstPolsIdsExpressions = expressionsBin->readU32LE();
    uint32_t nCmPolsIdsExpressions = expressionsBin->readU32LE();
    uint32_t nChallengesIdsExpressions = expressionsBin->readU32LE();
    uint32_t nPublicsIdsExpressions = expressionsBin->readU32LE();
    uint32_t nSubproofValuesIdsExpressions = expressionsBin->readU32LE();

    expressionsBinArgsExpressions.ops = new uint8_t[nOpsExpressions];
    expressionsBinArgsExpressions.args = new uint16_t[nArgsExpressions];
    expressionsBinArgsExpressions.numbers = new uint64_t[nNumbersExpressions];
    expressionsBinArgsExpressions.constPolsIds = new uint16_t[nConstPolsIdsExpressions];
    expressionsBinArgsExpressions.cmPolsIds = new uint16_t[nCmPolsIdsExpressions];
    expressionsBinArgsExpressions.challengesIds = new uint16_t[nChallengesIdsExpressions];
    expressionsBinArgsExpressions.publicsIds = new uint16_t[nPublicsIdsExpressions];
    expressionsBinArgsExpressions.subproofValuesIds = new uint16_t[nSubproofValuesIdsExpressions];

    uint64_t nExpressions = expressionsBin->readU32LE();

    for(uint64_t i = 0; i < nExpressions; ++i) {
        ParserParams parserParamsExpression;

        uint32_t expId = expressionsBin->readU32LE();
        
        parserParamsExpression.expId = expId;
        parserParamsExpression.destDim = expressionsBin->readU32LE();
        parserParamsExpression.destId = expressionsBin->readU32LE();
        parserParamsExpression.stage = expressionsBin->readU32LE();

        parserParamsExpression.nTemp1 = expressionsBin->readU32LE();
        parserParamsExpression.nTemp3 = expressionsBin->readU32LE();

        parserParamsExpression.nOps = expressionsBin->readU32LE();
        parserParamsExpression.opsOffset = expressionsBin->readU32LE();

        parserParamsExpression.nArgs = expressionsBin->readU32LE();
        parserParamsExpression.argsOffset = expressionsBin->readU32LE();

        parserParamsExpression.nNumbers = expressionsBin->readU32LE();
        parserParamsExpression.numbersOffset = expressionsBin->readU32LE();

        parserParamsExpression.nConstPolsUsed = expressionsBin->readU32LE();
        parserParamsExpression.constPolsOffset = expressionsBin->readU32LE();

        parserParamsExpression.nCmPolsUsed = expressionsBin->readU32LE();
        parserParamsExpression.cmPolsOffset = expressionsBin->readU32LE();

        parserParamsExpression.nChallengesUsed = expressionsBin->readU32LE();
        parserParamsExpression.challengesOffset = expressionsBin->readU32LE();

        parserParamsExpression.nPublicsUsed = expressionsBin->readU32LE();
        parserParamsExpression.publicsOffset = expressionsBin->readU32LE();

        parserParamsExpression.nSubproofValuesUsed = expressionsBin->readU32LE();
        parserParamsExpression.subproofValuesOffset = expressionsBin->readU32LE();

        parserParamsExpression.line = expressionsBin->readString();

        expressionsInfo[expId] = parserParamsExpression;
    }

    for(uint64_t j = 0; j < nOpsExpressions; ++j) {
        expressionsBinArgsExpressions.ops[j] = expressionsBin->readU8LE();
    }
    for(uint64_t j = 0; j < nArgsExpressions; ++j) {
        expressionsBinArgsExpressions.args[j] = expressionsBin->readU16LE();
    }
    for(uint64_t j = 0; j < nNumbersExpressions; ++j) {
        expressionsBinArgsExpressions.numbers[j] = expressionsBin->readU64LE();
    }

    for(uint64_t j = 0; j < nConstPolsIdsExpressions; ++j) {
        expressionsBinArgsExpressions.constPolsIds[j] = expressionsBin->readU16LE();
    }

    for(uint64_t j = 0; j < nCmPolsIdsExpressions; ++j) {
        expressionsBinArgsExpressions.cmPolsIds[j] = expressionsBin->readU16LE();
    }

    for(uint64_t j = 0; j < nChallengesIdsExpressions; ++j) {
        expressionsBinArgsExpressions.challengesIds[j] = expressionsBin->readU16LE();
    }

    for(uint64_t j = 0; j < nPublicsIdsExpressions; ++j) {
        expressionsBinArgsExpressions.publicsIds[j] = expressionsBin->readU16LE();
    }

    for(uint64_t j = 0; j < nSubproofValuesIdsExpressions; ++j) {
        expressionsBinArgsExpressions.subproofValuesIds[j] = expressionsBin->readU16LE();
    }

    expressionsBin->endReadSection();

    expressionsBin->startReadSection(BINARY_CONSTRAINTS_SECTION);

    uint32_t nOpsDebug = expressionsBin->readU32LE();
    uint32_t nArgsDebug = expressionsBin->readU32LE();
    uint32_t nNumbersDebug = expressionsBin->readU32LE();
    uint32_t nConstPolsIdsDebug = expressionsBin->readU32LE();
    uint32_t nCmPolsIdsDebug = expressionsBin->readU32LE();
    uint32_t nChallengesIdsDebug = expressionsBin->readU32LE();
    uint32_t nPublicsIdsDebug = expressionsBin->readU32LE();
    uint32_t nSubproofValuesIdsDebug = expressionsBin->readU32LE();

    expressionsBinArgsConstraints.ops = new uint8_t[nOpsDebug];
    expressionsBinArgsConstraints.args = new uint16_t[nArgsDebug];
    expressionsBinArgsConstraints.numbers = new uint64_t[nNumbersDebug];
    expressionsBinArgsConstraints.constPolsIds = new uint16_t[nConstPolsIdsDebug];
    expressionsBinArgsConstraints.cmPolsIds = new uint16_t[nCmPolsIdsDebug];
    expressionsBinArgsConstraints.challengesIds = new uint16_t[nChallengesIdsDebug];
    expressionsBinArgsConstraints.publicsIds = new uint16_t[nPublicsIdsDebug];
    expressionsBinArgsConstraints.subproofValuesIds = new uint16_t[nSubproofValuesIdsDebug];
    
    uint32_t nConstraints = expressionsBin->readU32LE();

    for(uint64_t i = 0; i < nConstraints; ++i) {
        ParserParams parserParamsConstraint;

        uint32_t stage = expressionsBin->readU32LE();
        parserParamsConstraint.stage = stage;
        parserParamsConstraint.expId = 0;
        
        parserParamsConstraint.destDim = expressionsBin->readU32LE();
        parserParamsConstraint.destId = expressionsBin->readU32LE();

        parserParamsConstraint.firstRow = expressionsBin->readU32LE();
        parserParamsConstraint.lastRow = expressionsBin->readU32LE();

        parserParamsConstraint.nTemp1 = expressionsBin->readU32LE();
        parserParamsConstraint.nTemp3 = expressionsBin->readU32LE();

        parserParamsConstraint.nOps = expressionsBin->readU32LE();
        parserParamsConstraint.opsOffset = expressionsBin->readU32LE();

        parserParamsConstraint.nArgs = expressionsBin->readU32LE();
        parserParamsConstraint.argsOffset = expressionsBin->readU32LE();

        parserParamsConstraint.nNumbers = expressionsBin->readU32LE();
        parserParamsConstraint.numbersOffset = expressionsBin->readU32LE();

        parserParamsConstraint.nConstPolsUsed = expressionsBin->readU32LE();
        parserParamsConstraint.constPolsOffset = expressionsBin->readU32LE();

        parserParamsConstraint.nCmPolsUsed = expressionsBin->readU32LE();
        parserParamsConstraint.cmPolsOffset = expressionsBin->readU32LE();

        parserParamsConstraint.nChallengesUsed = expressionsBin->readU32LE();
        parserParamsConstraint.challengesOffset = expressionsBin->readU32LE();

        parserParamsConstraint.nPublicsUsed = expressionsBin->readU32LE();
        parserParamsConstraint.publicsOffset = expressionsBin->readU32LE();

        parserParamsConstraint.nSubproofValuesUsed = expressionsBin->readU32LE();
        parserParamsConstraint.subproofValuesOffset = expressionsBin->readU32LE();

        parserParamsConstraint.imPol = bool(expressionsBin->readU32LE());
        parserParamsConstraint.line = expressionsBin->readString();

        constraintsInfoDebug.push_back(parserParamsConstraint);
    }


    for(uint64_t j = 0; j < nOpsDebug; ++j) {
        expressionsBinArgsConstraints.ops[j] = expressionsBin->readU8LE();
    }
    for(uint64_t j = 0; j < nArgsDebug; ++j) {
        expressionsBinArgsConstraints.args[j] = expressionsBin->readU16LE();
    }
    for(uint64_t j = 0; j < nNumbersDebug; ++j) {
        expressionsBinArgsConstraints.numbers[j] = expressionsBin->readU64LE();
    }

    for(uint64_t j = 0; j < nConstPolsIdsDebug; ++j) {
        expressionsBinArgsConstraints.constPolsIds[j] = expressionsBin->readU16LE();
    }

    for(uint64_t j = 0; j < nCmPolsIdsDebug; ++j) {
        expressionsBinArgsConstraints.cmPolsIds[j] = expressionsBin->readU16LE();
    }

    for(uint64_t j = 0; j < nChallengesIdsDebug; ++j) {
        expressionsBinArgsConstraints.challengesIds[j] = expressionsBin->readU16LE();
    }

    for(uint64_t j = 0; j < nPublicsIdsDebug; ++j) {
        expressionsBinArgsConstraints.publicsIds[j] = expressionsBin->readU16LE();
    }

    for(uint64_t j = 0; j < nSubproofValuesIdsDebug; ++j) {
        expressionsBinArgsConstraints.subproofValuesIds[j] = expressionsBin->readU16LE();
    }

    expressionsBin->endReadSection();

    expressionsBin->startReadSection(BINARY_HINTS_SECTION);

    uint32_t nHints = expressionsBin->readU32LE();

    for(uint64_t h = 0; h < nHints; h++) {
        Hint hint;
        hint.name = expressionsBin->readString();

        uint32_t nFields = expressionsBin->readU32LE();

        for(uint64_t f = 0; f < nFields; f++) {
            HintField hintField;
            std::string name = expressionsBin->readString();
            std::string operand = expressionsBin->readString();
            hintField.name = name;
            hintField.operand = string2opType(operand);
            if(hintField.operand == opType::number) {
                hintField.value = expressionsBin->readU64LE();
            } else if(hintField.operand == opType::string_) {
                hintField.stringValue = expressionsBin->readString();
            } else {
                hintField.id = expressionsBin->readU32LE();
            }
            if(hintField.operand == opType::tmp) {
                hintField.dim = expressionsBin->readU32LE();
            }
            hint.fields.push_back(hintField);
        }

        hints.push_back(hint);
    }

    expressionsBin->endReadSection();
}

VecU64Result ExpressionsBin::getHintIdsByName(std::string name) {
    VecU64Result hintIds;

    hintIds.nElements = 0;
    for (uint64_t i = 0; i < hints.size(); ++i) {
        if (hints[i].name == name) {
            hintIds.nElements++;
        }
    }

    uint64_t c = 0;
    hintIds.ids = new uint64_t[hintIds.nElements];
    for (uint64_t i = 0; i < hints.size(); ++i) {
        if (hints[i].name == name) {
            hintIds.ids[c++] = i;
        }
    }

    return hintIds;
}