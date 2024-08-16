#ifndef EXPRESSIONS_BUILDER_HPP
#define EXPRESSIONS_BUILDER_HPP
#include "chelpers.hpp"
#include "stark_info.hpp"
#include "steps.hpp"
#include "hint_handler.hpp"

class ExpressionsBuilder {
public:

    StarkInfo& starkInfo;
    CHelpers& cHelpers;
    StepsParams& params;

    vector<bool> subProofValuesCalculated;
    vector<bool> commitsCalculated;

    ExpressionsBuilder(StarkInfo& _starkInfo, CHelpers& _cHelpers, StepsParams& _params) : starkInfo(_starkInfo),  cHelpers(_cHelpers), params(_params) {
        commitsCalculated.resize(starkInfo.cmPolsMap.size(), false);
        subProofValuesCalculated.resize(starkInfo.nSubProofValues, false);
    };

    virtual ~ExpressionsBuilder() {};

    std::tuple<Goldilocks::Element*, hintFieldType> getHintField(uint64_t hintId, std::string hintFieldName) {
        uint64_t deg = 1 << starkInfo.starkStruct.nBits;
        Hint hint = cHelpers.hints[hintId];
        if(hint.fields.count(hintFieldName) == 0) {
            zklog.error("Hint field " + hintFieldName + " not found in hint " + hint.name + ".");
            exitProcess();
            exit(-1);
        }

        Goldilocks::Element* dest;
        hintFieldType type;

        HintField hintField = hint.fields[hintFieldName];
        if(hintField.operand == opType::cm) {
            uint64_t dim = starkInfo.cmPolsMap[hintField.id].dim;
            dest = new Goldilocks::Element[deg*dim];
            getPolynomial(dest, true, hintField.id, false);
            type = dim == 1 ? hintFieldType::Column : hintFieldType::Extended_Column;
        } else if(hintField.operand == opType::const_) {
            uint64_t dim = starkInfo.constPolsMap[hintField.id].dim;
            dest = new Goldilocks::Element[deg*dim];
            getPolynomial(dest, false, hintField.id, false);
            type = dim == 1 ? hintFieldType::Column : hintFieldType::Extended_Column;
        } else if (hintField.operand == opType::tmp) {
            uint64_t dim = cHelpers.expressionsInfo[hintField.id].destDim;
            dest = new Goldilocks::Element[deg*dim];
            calculateExpression(dest, hintField.id);
            type = dim == 1 ? hintFieldType::Column : hintFieldType::Extended_Column;
        } else if (hintField.operand == opType::public_) {
            dest = new Goldilocks::Element[1];
            dest[0] = params.publicInputs[hintField.id];
            type = hintFieldType::Field;
        } else if (hintField.operand == opType::number) {
            dest = new Goldilocks::Element[1];
            dest[0] = Goldilocks::fromU64(hintField.value);
            type = hintFieldType::Field;
        } else if (hintField.operand == opType::subproofvalue) {
            dest = new Goldilocks::Element[FIELD_EXTENSION];
            std::memcpy(dest, &params.subproofValues[FIELD_EXTENSION*hintField.id], FIELD_EXTENSION * sizeof(Goldilocks::Element));
            type = hintFieldType::Extended_Field;
        } else if (hintField.operand == opType::challenge) {
            dest = new Goldilocks::Element[FIELD_EXTENSION];
            std::memcpy(dest, &params.challenges[FIELD_EXTENSION*hintField.id], FIELD_EXTENSION * sizeof(Goldilocks::Element));
            type = hintFieldType::Extended_Field;
        } else {
            zklog.error("Unknown hintFieldType");
            exitProcess();
            exit(-1);
        }

        return std::make_tuple(dest, type);
    }

    void setHintField(Goldilocks::Element* values, uint64_t hintId, std::string hintFieldName) {
        
        Hint hint = cHelpers.hints[hintId];
        HintField hintField = hint.fields[hintFieldName];

        if(hintField.operand == opType::cm) {
            setPolynomial(values, hintField.id, false);
            commitsCalculated[hintField.id] = true;
        } else if(hintField.operand == opType::subproofvalue) {
            std::memcpy(&params.subproofValues[FIELD_EXTENSION*hintField.id], values, FIELD_EXTENSION * sizeof(Goldilocks::Element));
            subProofValuesCalculated[hintField.id] = true;
        } else {
            zklog.error("Only committed pols and subproofvalues can be set");
            exitProcess();
            exit(-1);  
        }
    }

    virtual void calculateExpressions(Goldilocks::Element *dest, ParserArgs &parserArgs, ParserParams &parserParams, bool domainExtended, bool inverse) {};

    bool verifyConstraint(Goldilocks::Element* dest, ParserParams& parserParams, uint64_t row) {
        if(row < parserParams.firstRow || row > parserParams.lastRow) return true;
        if(parserParams.destDim == 1) {
            return Goldilocks::isZero(dest[row]);
        } else if(parserParams.destDim == FIELD_EXTENSION) {
            for(uint64_t i = 0; i < FIELD_EXTENSION; ++i) {
                if(!Goldilocks::isZero(dest[FIELD_EXTENSION*row + i])) return false;
            }
            return true;
        } else {
            exitProcess();
            exit(-1);
        }
    }
    
    virtual void verifyConstraints(uint64_t stage) {
        for (uint64_t i = 0; i < cHelpers.constraintsInfoDebug.size(); i++) {
            if(cHelpers.constraintsInfoDebug[i].stage == stage) {
                Goldilocks::Element* pAddr = &params.pols[starkInfo.mapOffsets[std::make_pair("q", true)]];
                calculateConstraint(pAddr, i);
            }
        }
    }

    virtual void calculateConstraint(Goldilocks::Element* dest, uint64_t constraintId) {
        uint64_t domainSize = 1 << starkInfo.starkStruct.nBits;
        Goldilocks::Element* validConstraint = new Goldilocks::Element[domainSize];
    #pragma omp parallel for
        for(uint64_t i = 0; i < domainSize; ++i) {
            validConstraint[i] = Goldilocks::one();
        }  
        
        calculateExpressions(dest, cHelpers.cHelpersArgsDebug, cHelpers.constraintsInfoDebug[constraintId], false, false);


        bool isValidConstraint = true;
        uint64_t nInvalidRows = 0;
        uint64_t maxInvalidRowsDisplay = 100;
        for(uint64_t i = 0; i < domainSize; ++i) {
            if(!verifyConstraint(dest, cHelpers.constraintsInfoDebug[constraintId], i)) {
                isValidConstraint = false;
                if(nInvalidRows < maxInvalidRowsDisplay) {
                    cout << "Constraint check failed at " << i << endl;
                    nInvalidRows++;
                } else {
                    cout << "There are more than " << maxInvalidRowsDisplay << " invalid rows" << endl;
                    break;
                }
            }
        }
        if(isValidConstraint) {
            TimerLog(CONSTRAINT_CHECKS_PASSED);
        } else {
            TimerLog(CONSTRAINT_CHECKS_FAILED);
        }
    }

    void getPolynomial(Polinomial &pol, bool committed, uint64_t idPol, bool domainExtended) {
        PolMap polInfo = committed ? starkInfo.cmPolsMap[idPol] : starkInfo.constPolsMap[idPol];
        uint64_t deg = domainExtended ? 1 << starkInfo.starkStruct.nBitsExt : 1 << starkInfo.starkStruct.nBits;
        uint64_t dim = polInfo.dim;
        std::string stage = committed ? "cm" + to_string(polInfo.stage) : "const";
        uint64_t nCols = starkInfo.mapSectionsN[stage];
        uint64_t offset = starkInfo.mapOffsets[std::make_pair(stage, domainExtended)];
        offset += polInfo.stagePos;
        Goldilocks::Element *pols = committed ? params.pols : domainExtended ? params.constPolsExtended : params.constPols;
        pol = Polinomial(&pols[offset], deg, dim, nCols, std::to_string(idPol));
    }
    
    void getPolynomial(Goldilocks::Element *dest, bool committed, uint64_t idPol, bool domainExtended) {
        PolMap polInfo = committed ? starkInfo.cmPolsMap[idPol] : starkInfo.constPolsMap[idPol];
        uint64_t deg = domainExtended ? 1 << starkInfo.starkStruct.nBitsExt : 1 << starkInfo.starkStruct.nBits;
        uint64_t dim = polInfo.dim;
        std::string stage = committed ? "cm" + to_string(polInfo.stage) : "const";
        uint64_t nCols = starkInfo.mapSectionsN[stage];
        uint64_t offset = starkInfo.mapOffsets[std::make_pair(stage, domainExtended)];
        offset += polInfo.stagePos;
        Goldilocks::Element *pols = committed ? params.pols : domainExtended ? params.constPolsExtended : params.constPols;
        Polinomial pol = Polinomial(&pols[offset], deg, dim, nCols, std::to_string(idPol));

        for(uint64_t j = 0; j < deg; ++j) {
            std::memcpy(&dest[j*FIELD_EXTENSION], pol[j], dim * sizeof(Goldilocks::Element));
        }
    }

    void setPolynomial(Goldilocks::Element *values, uint64_t idPol, bool domainExtended) {
        PolMap polInfo = starkInfo.cmPolsMap[idPol];
        uint64_t deg = domainExtended ? 1 << starkInfo.starkStruct.nBitsExt : 1 << starkInfo.starkStruct.nBits;
        uint64_t dim = polInfo.dim;
        std::string stage = "cm" + to_string(polInfo.stage);
        uint64_t nCols = starkInfo.mapSectionsN[stage];
        uint64_t offset = starkInfo.mapOffsets[std::make_pair(stage, domainExtended)];
        offset += polInfo.stagePos;
        Polinomial pol = Polinomial(&params.pols[offset], deg, dim, nCols, std::to_string(idPol));

        for(uint64_t j = 0; j < deg; ++j) {
            std::memcpy(pol[j], &values[j*FIELD_EXTENSION], dim * sizeof(Goldilocks::Element));
        }
    }

    virtual void calculateExpression(Goldilocks::Element* dest, uint64_t expressionId, bool inverse = false) {
        bool domainExtended = expressionId == starkInfo.cExpId || expressionId == starkInfo.friExpId;
        calculateExpressions(dest, cHelpers.cHelpersArgsExpressions, cHelpers.expressionsInfo[expressionId], domainExtended, inverse);
    }
};

#endif