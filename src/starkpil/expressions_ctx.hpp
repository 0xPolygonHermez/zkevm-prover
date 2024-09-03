#ifndef EXPRESSIONS_CTX_HPP
#define EXPRESSIONS_CTX_HPP
#include "expressions_bin.hpp"
#include "const_pols.hpp"
#include "stark_info.hpp"
#include "steps.hpp"
#include "hint_handler.hpp"
#include "setup_ctx.hpp"

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

struct VecU64Result {
    uint64_t nElements;
    uint64_t* ids;
};

class ExpressionsCtx {
public:

    SetupCtx setupCtx;

    vector<bool> subProofValuesCalculated;
    vector<bool> commitsCalculated;

    ExpressionsCtx(SetupCtx& _setupCtx) : setupCtx(_setupCtx) {
        commitsCalculated.resize(setupCtx.starkInfo.cmPolsMap.size(), false);
        subProofValuesCalculated.resize(setupCtx.starkInfo.nSubProofValues, false);
    };

    virtual ~ExpressionsCtx() {};

    void setCommitCalculated(uint64_t id) {
        commitsCalculated[id] = true;
    };

    void setSubproofValueCalculated(uint64_t id) {
        subProofValuesCalculated[id] = true;
    };

    void canImPolsBeCalculated(uint64_t step) {
        for(uint64_t i = 0; i < setupCtx.starkInfo.cmPolsMap.size(); ++i) {
            PolMap cmPol = setupCtx.starkInfo.cmPolsMap[i];
            if((cmPol.stage < step || (cmPol.stage == step && !cmPol.imPol)) && !commitsCalculated[i]) {
                zklog.info("Witness polynomial " + setupCtx.starkInfo.cmPolsMap[i].name + " is not calculated");
                exitProcess();
                exit(-1);
            }
        }
        
    }

    void canStageBeCalculated(uint64_t step) {
        if(step == setupCtx.starkInfo.nStages) {
            for(uint64_t i = 0; i < setupCtx.starkInfo.nSubProofValues; i++) {
                if(!subProofValuesCalculated[i]) {
                    zklog.info("Subproofvalue " + to_string(i) + " is not calculated");
                    exitProcess();
                    exit(-1);
                }
            }
        }

        if(step <= setupCtx.starkInfo.nStages) {
            for(uint64_t i = 0; i < setupCtx.starkInfo.cmPolsMap.size(); i++) {
                if(setupCtx.starkInfo.cmPolsMap[i].stage == step && !commitsCalculated[i]) {
                    zklog.info("Witness polynomial " + setupCtx.starkInfo.cmPolsMap[i].name + " is not calculated");
                    exitProcess();
                    exit(-1);
                }
            }
        }
    }

    VecU64Result getHintIdsByName(std::string name) {
        VecU64Result hintIds;

        hintIds.nElements = 0;
        for (uint64_t i = 0; i < setupCtx.expressionsBin.hints.size(); ++i) {
            if (setupCtx.expressionsBin.hints[i].name == name) {
                hintIds.nElements++;
            }
        }

        uint64_t c = 0;
        hintIds.ids = new uint64_t[hintIds.nElements];
        for (uint64_t i = 0; i < setupCtx.expressionsBin.hints.size(); ++i) {
            if (setupCtx.expressionsBin.hints[i].name == name) {
               hintIds.ids[c++] = i;
            }
        }

        return hintIds;
    }
    
    HintFieldInfo getHintField(StepsParams& params, uint64_t hintId, std::string hintFieldName, bool dest) {
        uint64_t deg = 1 << setupCtx.starkInfo.starkStruct.nBits;

        if(setupCtx.expressionsBin.hints.size() == 0) {
            zklog.error("No hints were found.");
            exitProcess();
            exit(-1);
        }

        Hint hint = setupCtx.expressionsBin.hints[hintId];
        
        auto hintField = std::find_if(hint.fields.begin(), hint.fields.end(), [hintFieldName](const HintField& hintField) {
            return hintField.name == hintFieldName;
        });

        if(hintField == hint.fields.end()) {
            zklog.error("Hint field " + hintFieldName + " not found in hint " + hint.name + ".");
            exitProcess();
            exit(-1);
        }

        if(dest && hintField->operand != opType::cm && hintField->operand == opType::subproofvalue) {
            zklog.error("Invalid destination.");
            exitProcess();
            exit(-1);
        }

        HintFieldInfo hintFieldInfo;

        if(hintField->operand == opType::cm) {
            uint64_t dim = setupCtx.starkInfo.cmPolsMap[hintField->id].dim;
            hintFieldInfo.size = deg*dim;
            hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
            hintFieldInfo.fieldType = dim == 1 ? HintFieldType::Column : HintFieldType::ColumnExtended;
            hintFieldInfo.offset = dim;
            if(!dest) getPolynomial(params, hintFieldInfo.values, true, hintField->id, false);
        } else if(hintField->operand == opType::const_) {
            uint64_t dim = setupCtx.starkInfo.constPolsMap[hintField->id].dim;
            hintFieldInfo.size = deg*dim;
            hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
            hintFieldInfo.fieldType = dim == 1 ? HintFieldType::Column : HintFieldType::ColumnExtended;
            hintFieldInfo.offset = dim;
            getPolynomial(params, hintFieldInfo.values, false, hintField->id, false);
        } else if (hintField->operand == opType::tmp) {
            uint64_t dim = setupCtx.expressionsBin.expressionsInfo[hintField->id].destDim;
            hintFieldInfo.size = deg*dim;
            hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
            hintFieldInfo.fieldType = dim == 1 ? HintFieldType::Column : HintFieldType::ColumnExtended;
            hintFieldInfo.offset = dim;
            calculateExpression(params, hintFieldInfo.values, hintField->id);
        } else if (hintField->operand == opType::public_) {
            hintFieldInfo.size = 1;
            hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
            hintFieldInfo.values[0] = params.publicInputs[hintField->id];
            hintFieldInfo.fieldType = HintFieldType::Field;
            hintFieldInfo.offset = 1;
        } else if (hintField->operand == opType::number) {
            hintFieldInfo.size = 1;
            hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
            hintFieldInfo.values[0] = Goldilocks::fromU64(hintField->value);
            hintFieldInfo.fieldType = HintFieldType::Field;
            hintFieldInfo.offset = 1;
            cout << Goldilocks::toString(hintFieldInfo.values[0]) << endl;
        } else if (hintField->operand == opType::subproofvalue) {
            hintFieldInfo.size = FIELD_EXTENSION;
            hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
            hintFieldInfo.fieldType = HintFieldType::FieldExtended;
            hintFieldInfo.offset = FIELD_EXTENSION;
            if(!dest) std::memcpy(hintFieldInfo.values, &params.subproofValues[FIELD_EXTENSION*hintField->id], FIELD_EXTENSION * sizeof(Goldilocks::Element));
        } else if (hintField->operand == opType::challenge) {
            hintFieldInfo.size = FIELD_EXTENSION;
            hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
            hintFieldInfo.fieldType = HintFieldType::FieldExtended;
            hintFieldInfo.offset = FIELD_EXTENSION;
            std::memcpy(hintFieldInfo.values, &params.challenges[FIELD_EXTENSION*hintField->id], FIELD_EXTENSION * sizeof(Goldilocks::Element));
        } else {
            zklog.error("Unknown HintFieldType");
            exitProcess();
            exit(-1);
        }

        return hintFieldInfo;
    }

    void setHintField(StepsParams& params, Goldilocks::Element* values, uint64_t hintId, std::string hintFieldName) {
        
        Hint hint = setupCtx.expressionsBin.hints[hintId];

        auto hintField = std::find_if(hint.fields.begin(), hint.fields.end(), [hintFieldName](const HintField& hintField) {
            return hintField.name == hintFieldName;
        });

        if(hintField == hint.fields.end()) {
            zklog.error("Hint field " + hintFieldName + " not found in hint " + hint.name + ".");
            exitProcess();
            exit(-1);
        }

        if(hintField->operand == opType::cm) {
            setPolynomial(params, values, hintField->id, false);
        } else if(hintField->operand == opType::subproofvalue) {
            setSubproofValue(params, values, hintField->id);
        } else {
            zklog.error("Only committed pols and subproofvalues can be set");
            exitProcess();
            exit(-1);  
        }
    }

    virtual void calculateExpressions(StepsParams& params, Goldilocks::Element *dest, ParserArgs &parserArgs, ParserParams &parserParams, bool domainExtended, bool inverse = false) {};

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
    
    VecU64Result verifyConstraints(uint64_t stage, StepsParams& params) {
        std::vector<uint64_t> invalid;

        VecU64Result invalidConstraints;
        invalidConstraints.nElements = 0;
        for (uint64_t i = 0; i < setupCtx.expressionsBin.constraintsInfoDebug.size(); i++) {
            if(setupCtx.expressionsBin.constraintsInfoDebug[i].stage == stage) {
                Goldilocks::Element* pAddr = &params.pols[setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)]];
                if(!verifyConstraint(params, pAddr, i)) {
                    invalid.push_back(i);
                    invalidConstraints.nElements++;
                };
            }
        }
        
        if(invalidConstraints.nElements > 0) {
            invalidConstraints.ids = new uint64_t[invalidConstraints.nElements];
            std::copy(invalid.begin(), invalid.end(), invalidConstraints.ids);
        } else {
            invalidConstraints.ids = nullptr;
        }

        return invalidConstraints;
    }

    bool verifyConstraint(StepsParams& params, Goldilocks::Element* dest, uint64_t constraintId) {
        TimerLog(CHECKING_CONSTRAINT);
        cout << "--------------------------------------------------------" << endl;
        cout << setupCtx.expressionsBin.constraintsInfoDebug[constraintId].line << endl;
        cout << "--------------------------------------------------------" << endl;
        
        calculateExpressions(params, dest, setupCtx.expressionsBin.expressionsBinArgsConstraints, setupCtx.expressionsBin.constraintsInfoDebug[constraintId], false, false);

        uint64_t N = (1 << setupCtx.starkInfo.starkStruct.nBits);
        bool isValidConstraint = true;
        uint64_t nInvalidRows = 0;
        uint64_t maxInvalidRowsDisplay = 100;
        for(uint64_t i = 0; i < N; ++i) {
            if(nInvalidRows >= maxInvalidRowsDisplay) {
                cout << "There are more than " << maxInvalidRowsDisplay << " invalid rows for constraint " << i << endl;
                break;
            }
            if(!checkConstraint(dest, setupCtx.expressionsBin.constraintsInfoDebug[constraintId], i)) {
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
 
    void getPolynomial(StepsParams& params, Goldilocks::Element *dest, bool committed, uint64_t idPol, bool domainExtended) {
        PolMap polInfo = committed ? setupCtx.starkInfo.cmPolsMap[idPol] : setupCtx.starkInfo.constPolsMap[idPol];
        uint64_t deg = domainExtended ? 1 << setupCtx.starkInfo.starkStruct.nBitsExt : 1 << setupCtx.starkInfo.starkStruct.nBits;
        uint64_t dim = polInfo.dim;
        std::string stage = committed ? "cm" + to_string(polInfo.stage) : "const";
        uint64_t nCols = setupCtx.starkInfo.mapSectionsN[stage];
        uint64_t offset = setupCtx.starkInfo.mapOffsets[std::make_pair(stage, domainExtended)];
        offset += polInfo.stagePos;
        Goldilocks::Element *pols = committed ? params.pols : domainExtended ? setupCtx.constPols.pConstPolsAddressExtended : setupCtx.constPols.pConstPolsAddress;
        Polinomial pol = Polinomial(&pols[offset], deg, dim, nCols, std::to_string(idPol));

        for(uint64_t j = 0; j < deg; ++j) {
            std::memcpy(&dest[j*FIELD_EXTENSION], pol[j], dim * sizeof(Goldilocks::Element));
        }
    }

    void setPolynomial(StepsParams& params, Goldilocks::Element *values, uint64_t idPol, bool domainExtended) {
        PolMap polInfo = setupCtx.starkInfo.cmPolsMap[idPol];
        uint64_t deg = domainExtended ? 1 << setupCtx.starkInfo.starkStruct.nBitsExt : 1 << setupCtx.starkInfo.starkStruct.nBits;
        uint64_t dim = polInfo.dim;
        std::string stage = "cm" + to_string(polInfo.stage);
        uint64_t nCols = setupCtx.starkInfo.mapSectionsN[stage];
        uint64_t offset = setupCtx.starkInfo.mapOffsets[std::make_pair(stage, domainExtended)];
        offset += polInfo.stagePos;
        Polinomial pol = Polinomial(&params.pols[offset], deg, dim, nCols, std::to_string(idPol));

        for(uint64_t j = 0; j < deg; ++j) {
            std::memcpy(pol[j], &values[j*FIELD_EXTENSION], dim * sizeof(Goldilocks::Element));
        }
        commitsCalculated[idPol] = true;
    }

    void setSubproofValue(StepsParams& params, Goldilocks::Element *value, uint64_t subproofValueId) {
        std::memcpy(&params.subproofValues[FIELD_EXTENSION*subproofValueId], value, FIELD_EXTENSION * sizeof(Goldilocks::Element));
        subProofValuesCalculated[subproofValueId] = true;
    }

    void calculateExpression(StepsParams& params, Goldilocks::Element* dest, uint64_t expressionId, bool inverse = false) {
        bool domainExtended = expressionId == setupCtx.starkInfo.cExpId || expressionId == setupCtx.starkInfo.friExpId;
        calculateExpressions(params, dest, setupCtx.expressionsBin.expressionsBinArgsExpressions, setupCtx.expressionsBin.expressionsInfo[expressionId], domainExtended, inverse);
    }

    void calculateImPolsExpressions(uint64_t step, StepsParams& params) {
        TimerStart(STARK_CALCULATE_IMPOLS_EXPS);

        uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
        
        Goldilocks::Element* pAddr = &params.pols[setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)]];
        for(uint64_t i = 0; i < setupCtx.starkInfo.cmPolsMap.size(); i++) {
            if(setupCtx.starkInfo.cmPolsMap[i].imPol && setupCtx.starkInfo.cmPolsMap[i].stage == step) {
                calculateExpression(params, pAddr, setupCtx.starkInfo.cmPolsMap[i].expId);
                Goldilocks::Element* imAddr = &params.pols[setupCtx.starkInfo.mapOffsets[std::make_pair("cm" + to_string(step), false)] + setupCtx.starkInfo.cmPolsMap[i].stagePos];
            #pragma omp parallel
                for(uint64_t j = 0; j < N; ++j) {
                    std::memcpy(&imAddr[j*setupCtx.starkInfo.mapSectionsN["cm" + to_string(step)]], &pAddr[j*setupCtx.starkInfo.cmPolsMap[i].dim], setupCtx.starkInfo.cmPolsMap[i].dim * sizeof(Goldilocks::Element));
                }
                setCommitCalculated(i);
            }
        }
        
        TimerStopAndLog(STARK_CALCULATE_IMPOLS_EXPS);
    }


    void calculateQuotientPolynomial(StepsParams& params) {
        TimerStart(STARK_CALCULATE_QUOTIENT_POLYNOMIAL);
        calculateExpression(params, &params.pols[setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)]], setupCtx.starkInfo.cExpId);
        for(uint64_t i = 0; i < setupCtx.starkInfo.cmPolsMap.size(); i++) {
            if(setupCtx.starkInfo.cmPolsMap[i].stage == setupCtx.starkInfo.nStages + 1) {
                setCommitCalculated(i);
            }
        }
        TimerStopAndLog(STARK_CALCULATE_QUOTIENT_POLYNOMIAL);
    }

    void printExpression(Goldilocks::Element* pol, uint64_t deg, uint64_t dim, uint64_t printValues = 0) {
        Polinomial p = Polinomial(pol, deg, dim, dim);
        MerkleTreeGL *mt_ = new MerkleTreeGL(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom, deg, dim, pol);
        mt_->merkelize();

        Goldilocks::Element root[4];
        mt_->getRoot(&root[0]);

        if(printValues > 0) cout << "PRINTING VALUES" << endl;
        for(uint64_t i = 0; i < printValues; ++i) {
        if(dim == 3) {
                cout << i << " [" << Goldilocks::toString(p[i][0]) << ", " << Goldilocks::toString(p[i][1]) << ", " << Goldilocks::toString(p[i][2]) << " ]" << endl; 
            } else {
                cout << i << " " << Goldilocks::toString(p[i][0]) << endl;
            }
        }

        delete mt_;
    }

    void printPolById(StepsParams& params, uint64_t polId, uint64_t printValues = 0)
    {   
        uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
        PolMap polInfo = setupCtx.starkInfo.cmPolsMap[polId];
        Polinomial p;
        setupCtx.starkInfo.getPolynomial(p, params.pols, true, polId, false);
    
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