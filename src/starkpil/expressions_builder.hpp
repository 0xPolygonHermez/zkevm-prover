#ifndef EXPRESSIONS_BUILDER_HPP
#define EXPRESSIONS_BUILDER_HPP
#include "chelpers.hpp"
#include "stark_info.hpp"
#include "steps.hpp"
#include "hint_handler.hpp"

typedef enum {
    Field = 0,
    FieldExtended = 1,
    Column = 2,
    ColumnExtended = 3,
} HintFieldType;

struct HintFieldInfo {
    uint64_t size; // Destination size (in Goldilocks elements)
    uint8_t offset;
    HintFieldType fieldType;
    Goldilocks::Element* values;
};


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

    void setCommitCalculated(uint64_t id) {
        commitsCalculated[id] = true;
    };

    void setSubproofValueCalculated(uint64_t id) {
        subProofValuesCalculated[id] = true;
    };

    void canImPolsBeCalculated(uint64_t step) {
        for(uint64_t i = 0; i < starkInfo.cmPolsMap.size(); ++i) {
            PolMap cmPol = starkInfo.cmPolsMap[i];
            if((cmPol.stage < step || (cmPol.stage == step && !cmPol.imPol)) && !commitsCalculated[i]) {
                zklog.info("Witness polynomial " + starkInfo.cmPolsMap[i].name + " is not calculated");
                exitProcess();
                exit(-1);
            }
        }
    }

    void canStageBeCalculated(uint64_t step) {
        if(step == starkInfo.nStages) {
            for(uint64_t i = 0; i < starkInfo.nSubProofValues; i++) {
                if(!subProofValuesCalculated[i]) {
                    zklog.info("Subproofvalue " + to_string(i) + " is not calculated");
                    exitProcess();
                    exit(-1);
                }
            }
        }

        if(step <= starkInfo.nStages) {
            for(uint64_t i = 0; i < starkInfo.cmPolsMap.size(); i++) {
                if(starkInfo.cmPolsMap[i].stage == step && !commitsCalculated[i]) {
                    zklog.info("Witness polynomial " + starkInfo.cmPolsMap[i].name + " is not calculated");
                    exitProcess();
                    exit(-1);
                }
            }
        }
    }

    HintFieldInfo getHintField(uint64_t hintId, std::string hintFieldName, bool dest, bool firstStage) {
        uint64_t deg = 1 << starkInfo.starkStruct.nBits;

        if(cHelpers.hints.size() == 0) {
            zklog.error("No hints were found.");
            exitProcess();
            exit(-1);
        }

        Hint hint = cHelpers.hints[hintId];
        if(hint.fields.count(hintFieldName) == 0) {
            zklog.error("Hint field " + hintFieldName + " not found in hint " + hint.name + ".");
            exitProcess();
            exit(-1);
        }

        HintField hintField = hint.fields[hintFieldName];

        if(dest && hintField.operand != opType::cm && hintField.operand == opType::subproofvalue) {
            zklog.error("Invalid destination.");
            exitProcess();
            exit(-1);
        }

        HintFieldInfo hintFieldInfo;

        if(hintField.operand == opType::cm) {
            uint64_t dim = starkInfo.cmPolsMap[hintField.id].dim;
            hintFieldInfo.size = deg*dim;
            hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
            hintFieldInfo.fieldType = dim == 1 ? HintFieldType::Column : HintFieldType::ColumnExtended;
            hintFieldInfo.offset = dim;
            if(!dest) getPolynomial(hintFieldInfo.values, true, hintField.id, false);
        } else if(hintField.operand == opType::const_) {
            uint64_t dim = starkInfo.constPolsMap[hintField.id].dim;
            hintFieldInfo.size = deg*dim;
            hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
            hintFieldInfo.fieldType = dim == 1 ? HintFieldType::Column : HintFieldType::ColumnExtended;
            hintFieldInfo.offset = dim;
            getPolynomial(hintFieldInfo.values, false, hintField.id, false);
        } else if (hintField.operand == opType::tmp) {
            uint64_t dim = cHelpers.expressionsInfo[hintField.id].destDim;
            hintFieldInfo.size = deg*dim;
            hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
            hintFieldInfo.fieldType = dim == 1 ? HintFieldType::Column : HintFieldType::ColumnExtended;
            hintFieldInfo.offset = dim;
            calculateExpression(hintFieldInfo.values, hintField.id);
        } else if (hintField.operand == opType::public_) {
            hintFieldInfo.size = 1;
            hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
            hintFieldInfo.values[0] = params.publicInputs[hintField.id];
            hintFieldInfo.fieldType = HintFieldType::Field;
            hintFieldInfo.offset = 1;
        } else if (hintField.operand == opType::number) {
            hintFieldInfo.size = 1;
            hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
            hintFieldInfo.values[0] = Goldilocks::fromU64(hintField.value);
            hintFieldInfo.fieldType = HintFieldType::Field;
            hintFieldInfo.offset = 1;
            cout << Goldilocks::toString(hintFieldInfo.values[0]) << endl;
        } else if (hintField.operand == opType::subproofvalue) {
            hintFieldInfo.size = FIELD_EXTENSION;
            hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
            hintFieldInfo.fieldType = HintFieldType::FieldExtended;
            hintFieldInfo.offset = FIELD_EXTENSION;
            if(!dest) std::memcpy(hintFieldInfo.values, &params.subproofValues[FIELD_EXTENSION*hintField.id], FIELD_EXTENSION * sizeof(Goldilocks::Element));
        } else if (hintField.operand == opType::challenge) {
            hintFieldInfo.size = FIELD_EXTENSION;
            hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
            hintFieldInfo.fieldType = HintFieldType::FieldExtended;
            hintFieldInfo.offset = FIELD_EXTENSION;
            std::memcpy(hintFieldInfo.values, &params.challenges[FIELD_EXTENSION*hintField.id], FIELD_EXTENSION * sizeof(Goldilocks::Element));
        } else {
            zklog.error("Unknown HintFieldType");
            exitProcess();
            exit(-1);
        }

        return hintFieldInfo;
    }

    void setHintField(Goldilocks::Element* values, uint64_t hintId, std::string hintFieldName) {
        
        Hint hint = cHelpers.hints[hintId];
        HintField hintField = hint.fields[hintFieldName];

        if(hint.fields.count(hintFieldName) == 0) {
            zklog.error("Hint field " + hintFieldName + " not found in hint " + hint.name + ".");
            exitProcess();
            exit(-1);
        }

        if(hintField.operand == opType::cm) {
            setPolynomial(values, hintField.id, false);
        } else if(hintField.operand == opType::subproofvalue) {
            setSubproofValue(values, hintField.id);
        } else {
            zklog.error("Only committed pols and subproofvalues can be set");
            exitProcess();
            exit(-1);  
        }
    }

    virtual void calculateExpressions(Goldilocks::Element *dest, ParserArgs &parserArgs, ParserParams &parserParams, bool domainExtended, bool firstStage = false, bool inverse = false) {};

    bool checkConstraint(Goldilocks::Element* dest, ParserParams& parserParams, uint64_t row) {
        if(row < parserParams.firstRow || row > parserParams.lastRow) return true;
        bool isValid = true;
        if(parserParams.destDim == 1) {
            if(!Goldilocks::isZero(dest[row])) {
                isValid = false;
                cout << "Constraint check failed at row " << row << " with value: " << Goldilocks::toString(dest[row]) << endl;
            }
            
        } else if(parserParams.destDim == FIELD_EXTENSION) {
            isValid = true;
            for(uint64_t i = 0; i < FIELD_EXTENSION; ++i) {
                if(!Goldilocks::isZero(dest[FIELD_EXTENSION*row + i])) {
                    isValid = false;
                    cout << "Constraint check failed at row " << row << " with value: [" << Goldilocks::toString(dest[FIELD_EXTENSION*row]) << ", " << Goldilocks::toString(dest[FIELD_EXTENSION*row + 1]) << ", " << Goldilocks::toString(dest[FIELD_EXTENSION*row + 2]) << "]" << endl;
                    break;
                }
            }
        } else {
            exitProcess();
            exit(-1);
        }

        return isValid;
    }
    
    bool verifyConstraints(uint64_t stage) {
        bool isValid = true;
        for (uint64_t i = 0; i < cHelpers.constraintsInfoDebug.size(); i++) {
            if(cHelpers.constraintsInfoDebug[i].stage == stage) {
                Goldilocks::Element* pAddr = &params.pols[starkInfo.mapOffsets[std::make_pair("q", true)]];
                if(!verifyConstraint(pAddr, i)) {
                    isValid = false;
                };
            }
        }
        return isValid;
    }

    bool verifyConstraint(Goldilocks::Element* dest, uint64_t constraintId) {
        TimerLog(CHECKING_CONSTRAINT);
        cout << "--------------------------------------------------------" << endl;
        cout << cHelpers.constraintsInfoDebug[constraintId].line << endl;
        cout << "--------------------------------------------------------" << endl;
        
        calculateExpressions(dest, cHelpers.cHelpersArgsDebug, cHelpers.constraintsInfoDebug[constraintId], false, false, false);

        uint64_t N = (1 << starkInfo.starkStruct.nBits);
        bool isValidConstraint = true;
        uint64_t nInvalidRows = 0;
        uint64_t maxInvalidRowsDisplay = 100;
        for(uint64_t i = 0; i < N; ++i) {
            if(nInvalidRows >= maxInvalidRowsDisplay) {
                cout << "There are more than " << maxInvalidRowsDisplay << " invalid rows for constraint " << i << endl;
                break;
            }
            if(!checkConstraint(dest, cHelpers.constraintsInfoDebug[constraintId], i)) {
                if(isValidConstraint) isValidConstraint = false;
                nInvalidRows++;
            }
        }
        if(isValidConstraint) {
            TimerLog(VALID_CONSTRAINT);
            return true;
        } else {
            TimerLog(INVALID_CONSTRAINT);
            return false;
        }
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
        commitsCalculated[idPol] = true;
    }

    void setSubproofValue(Goldilocks::Element *value, uint64_t subproofValueId) {
        std::memcpy(&params.subproofValues[FIELD_EXTENSION*subproofValueId], value, FIELD_EXTENSION * sizeof(Goldilocks::Element));
        subProofValuesCalculated[subproofValueId] = true;
    }

    void calculateExpression(Goldilocks::Element* dest, uint64_t expressionId, bool inverse = false) {
        bool domainExtended = expressionId == starkInfo.cExpId || expressionId == starkInfo.friExpId;
        calculateExpressions(dest, cHelpers.cHelpersArgsExpressions, cHelpers.expressionsInfo[expressionId], domainExtended, false, inverse);
    }

    void printExpression(Goldilocks::Element* pol, uint64_t deg, uint64_t dim, bool printValues = false) {
        Polinomial p = Polinomial(pol, deg, dim, dim);
        MerkleTreeGL *mt_ = new MerkleTreeGL(starkInfo.starkStruct.merkleTreeArity, true, deg, dim, pol);
        mt_->merkelize();

        Goldilocks::Element root[4];
        mt_->getRoot(&root[0]);

        if(printValues) {
            cout << "PRINTING VALUES" << endl;
            for(uint64_t i = 0; i < 100; ++i) {
            if(dim == 3) {
                    cout << i << " [" << Goldilocks::toString(p[i][0]) << ", " << Goldilocks::toString(p[i][1]) << ", " << Goldilocks::toString(p[i][2]) << " ]" << endl; 
                } else {
                    cout << i << " " << Goldilocks::toString(p[i][0]) << endl;
                }
            }
        }
        

        delete mt_;
    }

    void printPolById(uint64_t polId, bool printValues = false)
    {   
        uint64_t N = 1 << starkInfo.starkStruct.nBits;
        PolMap polInfo = starkInfo.cmPolsMap[polId];
        Polinomial p;
        starkInfo.getPolynomial(p, params.pols, true, polId, false);
    
        Polinomial pCol;
        Goldilocks::Element *pBuffCol = new Goldilocks::Element[polInfo.dim * N];
        pCol.potConstruct(pBuffCol, N, polInfo.dim, polInfo.dim);
        Polinomial::copy(pCol, p);

        cout << "--------------------" << endl;
        cout << "Printing root of: " << polInfo.name << " (pol id " << polId << ")" << endl;
        printExpression(pBuffCol, N, polInfo.dim, printValues);

        delete pBuffCol;
    }
};

#endif