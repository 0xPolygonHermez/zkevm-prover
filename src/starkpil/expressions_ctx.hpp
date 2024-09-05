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

struct ConstraintRowInfo {
    uint64_t row;
    uint64_t dim;
    uint64_t value[3];
};

struct ConstraintInfo {
    uint64_t id;
    uint64_t stage;
    bool imPol;
    const char* line;
    uint64_t nrows;
    ConstraintRowInfo rows[10];
};

struct ConstraintsResults {
    uint64_t nConstraints;
    ConstraintInfo* constraintInfo;
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
    
    HintFieldInfo getHintField(StepsParams& params, uint64_t hintId, std::string hintFieldName, bool dest, bool print_expression) {
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

        if(print_expression) {
            cout << "--------------------------------------------------------" << endl;
            cout << "Hint name " << hintFieldName << " for hint id " << hintId << " is ";
        }
        if(hintField->operand == opType::cm) {
            uint64_t dim = setupCtx.starkInfo.cmPolsMap[hintField->id].dim;
            hintFieldInfo.size = deg*dim;
            hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
            hintFieldInfo.fieldType = dim == 1 ? HintFieldType::Column : HintFieldType::ColumnExtended;
            hintFieldInfo.offset = dim;
            if(print_expression) {
                cout << "witness col " << setupCtx.starkInfo.cmPolsMap[hintField->id].name;
                if(setupCtx.starkInfo.cmPolsMap[hintField->id].lengths.size() > 0) {
                    cout << "[";
                    for(uint64_t i = 0; i < setupCtx.starkInfo.cmPolsMap[hintField->id].lengths.size(); ++i) {
                        cout << setupCtx.starkInfo.cmPolsMap[hintField->id].lengths[i];
                        if(i != setupCtx.starkInfo.cmPolsMap[hintField->id].lengths.size() - 1) cout << ", ";
                    }
                    cout << "]";
                }
                cout << endl;
            }
            if(!dest) getPolynomial(params, hintFieldInfo.values, true, hintField->id, false);
        } else if(hintField->operand == opType::const_) {
            uint64_t dim = setupCtx.starkInfo.constPolsMap[hintField->id].dim;
            hintFieldInfo.size = deg*dim;
            hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
            hintFieldInfo.fieldType = dim == 1 ? HintFieldType::Column : HintFieldType::ColumnExtended;
            hintFieldInfo.offset = dim;
            if(print_expression) cout << "fixed col" << setupCtx.starkInfo.constPolsMap[hintField->id].name;
            if(setupCtx.starkInfo.constPolsMap[hintField->id].lengths.size() > 0) {
                cout << "[";
                for(uint64_t i = 0; i < setupCtx.starkInfo.constPolsMap[hintField->id].lengths.size(); ++i) {
                    cout << setupCtx.starkInfo.constPolsMap[hintField->id].lengths[i];
                    if(i != setupCtx.starkInfo.constPolsMap[hintField->id].lengths.size() - 1) cout << ", ";
                }
                cout << "]";
            }
            cout << endl;
            getPolynomial(params, hintFieldInfo.values, false, hintField->id, false);
        } else if (hintField->operand == opType::tmp) {
            uint64_t dim = setupCtx.expressionsBin.expressionsInfo[hintField->id].destDim;
            hintFieldInfo.size = deg*dim;
            hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
            hintFieldInfo.fieldType = dim == 1 ? HintFieldType::Column : HintFieldType::ColumnExtended;
            hintFieldInfo.offset = dim;
            if(print_expression && setupCtx.expressionsBin.expressionsInfo[hintField->id].line != "") {
                cout << "the expression with id: " << hintField->id << " " << setupCtx.expressionsBin.expressionsInfo[hintField->id].line << endl;
            }
            calculateExpression(params, hintFieldInfo.values, hintField->id);
        } else if (hintField->operand == opType::public_) {
            hintFieldInfo.size = 1;
            hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
            hintFieldInfo.values[0] = params.publicInputs[hintField->id];
            hintFieldInfo.fieldType = HintFieldType::Field;
            hintFieldInfo.offset = 1;
            if(print_expression) cout << "public input " << setupCtx.starkInfo.publicsMap[hintField->id].name << endl;
        } else if (hintField->operand == opType::number) {
            hintFieldInfo.size = 1;
            hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
            hintFieldInfo.values[0] = Goldilocks::fromU64(hintField->value);
            hintFieldInfo.fieldType = HintFieldType::Field;
            hintFieldInfo.offset = 1;
            if(print_expression) cout << "number " << hintField->value << endl;
        } else if (hintField->operand == opType::subproofvalue) {
            hintFieldInfo.size = FIELD_EXTENSION;
            hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
            hintFieldInfo.fieldType = HintFieldType::FieldExtended;
            hintFieldInfo.offset = FIELD_EXTENSION;
            if(print_expression) cout << "subproofValue " << setupCtx.starkInfo.subproofValuesMap[hintField->id].name << endl;
            if(!dest) std::memcpy(hintFieldInfo.values, &params.subproofValues[FIELD_EXTENSION*hintField->id], FIELD_EXTENSION * sizeof(Goldilocks::Element));
        } else if (hintField->operand == opType::challenge) {
            hintFieldInfo.size = FIELD_EXTENSION;
            hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
            hintFieldInfo.fieldType = HintFieldType::FieldExtended;
            hintFieldInfo.offset = FIELD_EXTENSION;
            if(print_expression) cout << "challenge " << setupCtx.starkInfo.challengesMap[hintField->id].name << endl;
            std::memcpy(hintFieldInfo.values, &params.challenges[FIELD_EXTENSION*hintField->id], FIELD_EXTENSION * sizeof(Goldilocks::Element));
        } else {
            zklog.error("Unknown HintFieldType");
            exitProcess();
            exit(-1);
        }

        if(print_expression) cout << "--------------------------------------------------------" << endl;

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

    std::tuple<bool, ConstraintRowInfo> checkConstraint(Goldilocks::Element* dest, ParserParams& parserParams, uint64_t row) {
        bool isValid = true;
        ConstraintRowInfo rowInfo;
        rowInfo.row = row;
        rowInfo.dim = parserParams.destDim;
        if(row < parserParams.firstRow || row > parserParams.lastRow) {
                rowInfo.value[0] = 0;
                rowInfo.value[1] = 0;
                rowInfo.value[2] = 0;
        } else {
             if(parserParams.destDim == 1) {
                rowInfo.value[0] = Goldilocks::toU64(dest[row]);
                rowInfo.value[1] = 0;
                rowInfo.value[2] = 0;
                if(rowInfo.value[0] != 0) isValid = false;
            } else if(parserParams.destDim == FIELD_EXTENSION) {
                rowInfo.value[0] = Goldilocks::toU64(dest[FIELD_EXTENSION*row]);
                rowInfo.value[1] = Goldilocks::toU64(dest[FIELD_EXTENSION*row + 1]);
                rowInfo.value[2] = Goldilocks::toU64(dest[FIELD_EXTENSION*row + 2]);
                if(rowInfo.value[0] != 0 || rowInfo.value[1] != 0 || rowInfo.value[2] != 0) isValid = false;
            } else {
                exitProcess();
                exit(-1);
            }
        }
       

        return std::make_tuple(isValid, rowInfo);
    }
    
    ConstraintsResults *verifyConstraints(StepsParams& params) {
        ConstraintsResults *constraintsInfo = new ConstraintsResults();
        constraintsInfo->nConstraints = setupCtx.expressionsBin.constraintsInfoDebug.size();
        
        constraintsInfo->constraintInfo = new ConstraintInfo[constraintsInfo->nConstraints];
        for (uint64_t i = 0; i < setupCtx.expressionsBin.constraintsInfoDebug.size(); i++) {
            Goldilocks::Element* pAddr = &params.pols[setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)]];
            auto constraintInfo = verifyConstraint(params, pAddr, i);
            constraintsInfo->constraintInfo[i] = constraintInfo;
        }
        
        return constraintsInfo;
    }

    ConstraintInfo verifyConstraint(StepsParams& params, Goldilocks::Element* dest, uint64_t constraintId) {        
        ConstraintInfo constraintInfo;
        constraintInfo.id = constraintId;
        constraintInfo.stage = setupCtx.expressionsBin.constraintsInfoDebug[constraintId].stage;
        constraintInfo.imPol = setupCtx.expressionsBin.constraintsInfoDebug[constraintId].imPol;
        constraintInfo.line = setupCtx.expressionsBin.constraintsInfoDebug[constraintId].line.c_str();
        constraintInfo.nrows = 0;
        calculateExpressions(params, dest, setupCtx.expressionsBin.expressionsBinArgsConstraints, setupCtx.expressionsBin.constraintsInfoDebug[constraintId], false, false);

        uint64_t N = (1 << setupCtx.starkInfo.starkStruct.nBits);

        std::vector<ConstraintRowInfo> constraintInvalidRows;
        for(uint64_t i = 0; i < N; ++i) {
            auto [isValid, rowInfo] = checkConstraint(dest, setupCtx.expressionsBin.constraintsInfoDebug[constraintId], i);
            if(!isValid) {
                constraintInvalidRows.push_back(rowInfo);
                constraintInfo.nrows++;
            }
        }

        uint64_t num_rows = std::min(constraintInfo.nrows, uint64_t(10));
        uint64_t h = num_rows / 2;
        for(uint64_t i = 0; i < h; ++i) {
            constraintInfo.rows[i] = constraintInvalidRows[i];
        }

        for(uint64_t i = h; i < num_rows; ++i) {
            if(constraintInfo.nrows > num_rows) {
                constraintInfo.rows[i] = constraintInvalidRows[constraintInvalidRows.size() - num_rows + i];
            } else {
                constraintInfo.rows[i] = constraintInvalidRows[i];
            }
        }

        return constraintInfo;
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
            std::memcpy(&dest[j*dim], pol[j], dim * sizeof(Goldilocks::Element));
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
            std::memcpy(pol[j], &values[j*dim], dim * sizeof(Goldilocks::Element));
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

    void printExpression(Goldilocks::Element* pol, uint64_t dim, uint64_t firstPrintValue = 0, uint64_t lastPrintValue = 0) {        
        cout << "-------------------------------------------------" << endl;
        for(uint64_t i = firstPrintValue; i < lastPrintValue; ++i) {
            if(dim == 3) {
                cout << "Value at " << i << " is: " << " [" << Goldilocks::toString(pol[i*FIELD_EXTENSION]) << ", " << Goldilocks::toString(pol[i*FIELD_EXTENSION + 1]) << ", " << Goldilocks::toString(pol[i*FIELD_EXTENSION + 2]) << " ]" << endl; 
            } else {
                cout << "Value at " << i << " is: " << Goldilocks::toString(pol[i]) << endl;
            }
        }
        cout << "-------------------------------------------------" << endl;
    }

    void printColById(StepsParams& params, bool committed, uint64_t polId, uint64_t firstPrintValue = 0, uint64_t lastPrintValue = 0)
    {   
        uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
        PolMap polInfo = committed ? setupCtx.starkInfo.cmPolsMap[polId] : setupCtx.starkInfo.constPolsMap[polId];
        Goldilocks::Element *pols = committed ? params.pols : setupCtx.constPols.pConstPolsAddress;
        Polinomial p;
        setupCtx.starkInfo.getPolynomial(p, pols, committed, polId, false);
    
        Polinomial pCol;
        Goldilocks::Element *pBuffCol = new Goldilocks::Element[polInfo.dim * N];
        pCol.potConstruct(pBuffCol, N, polInfo.dim, polInfo.dim);
        Polinomial::copy(pCol, p);

        cout << "--------------------" << endl;
        string type = committed ? "witness" : "fixed";
        cout << "Printing " << type << " column: " << polInfo.name;
        if(polInfo.lengths.size() > 0) {
            cout << "[";
            for(uint64_t i = 0; i < polInfo.lengths.size(); ++i) {
                cout << polInfo.lengths[i];
                if(i != polInfo.lengths.size() - 1) cout << ", ";
            }
            cout << "]";
        }
        cout << " (pol id " << polId << ")" << endl;
        printExpression(pBuffCol, polInfo.dim, firstPrintValue, lastPrintValue);
        delete pBuffCol;
    }

    HintFieldInfo printByName(StepsParams& params, string name, uint64_t *lengths, uint64_t firstPrintValue, uint64_t lastPrintValue, bool returnValues) {
        uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;

        HintFieldInfo hintFieldInfo;
        hintFieldInfo.size = 0;

        for(uint64_t i = 0; i < setupCtx.starkInfo.cmPolsMap.size(); ++i) {
            PolMap cmPol = setupCtx.starkInfo.cmPolsMap[i];
            if(cmPol.name != name) continue;
            if(cmPol.lengths.size() > 0) {
                bool lengths_match = true;
                for(uint64_t j = 0; j < cmPol.lengths.size(); ++j) {
                    if(cmPol.lengths[j] != lengths[j]) {
                        lengths_match = false;
                        break;
                    }
                }
                if(!lengths_match) continue;
            }
            if(cmPol.name == name) {
                printColById(params, true, i, firstPrintValue, lastPrintValue);
                if(returnValues) {
                    hintFieldInfo.size = cmPol.dim * N;
                    hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
                    hintFieldInfo.fieldType = cmPol.dim == 1 ? HintFieldType::Column : HintFieldType::ColumnExtended;
                    hintFieldInfo.offset = cmPol.dim;
                    getPolynomial(params, hintFieldInfo.values, true, i, false);
                }
                return hintFieldInfo;
            } 
        }

        for(uint64_t i = 0; i < setupCtx.starkInfo.constPolsMap.size(); ++i) {
            PolMap constPol = setupCtx.starkInfo.constPolsMap[i];
            if(constPol.name != name) continue;
            if(constPol.lengths.size() > 0) {
                bool lengths_match = true;
                for(uint64_t j = 0; j < constPol.lengths.size(); ++j) {
                    if(constPol.lengths[j] != lengths[j]) {
                        lengths_match = false;
                        break;
                    }
                }
                if(!lengths_match) continue;
            }
            if(constPol.name == name) {
                printColById(params, false, i, firstPrintValue, lastPrintValue);
                if(returnValues) {
                    hintFieldInfo.size = N;
                    hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
                    hintFieldInfo.fieldType = HintFieldType::Column;
                    hintFieldInfo.offset = 1;
                    getPolynomial(params, hintFieldInfo.values, false, i, false);
                }
                return hintFieldInfo;
            } 
        }

        for(uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); ++i) {
            PolMap challenge = setupCtx.starkInfo.challengesMap[i];
            if(challenge.name == name) {
                cout << "Printing challenge: " << name << " (stage " << challenge.stage << " and id " << challenge.stageId << "): ";
                cout << "[" << Goldilocks::toString(params.challenges[i*FIELD_EXTENSION]) << " , " << Goldilocks::toString(params.challenges[i*FIELD_EXTENSION + 1]) << " , " << Goldilocks::toString(params.challenges[i*FIELD_EXTENSION + 2]) << "]" << endl;
                if(returnValues) {
                    hintFieldInfo.size = FIELD_EXTENSION;
                    hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
                    hintFieldInfo.fieldType = HintFieldType::FieldExtended;
                    hintFieldInfo.offset = FIELD_EXTENSION;
                    std::memcpy(hintFieldInfo.values, &params.challenges[FIELD_EXTENSION*i], FIELD_EXTENSION * sizeof(Goldilocks::Element));
                }
                return hintFieldInfo;
            }
        }

        for(uint64_t i = 0; i < setupCtx.starkInfo.publicsMap.size(); ++i) {
            PolMap publicInput = setupCtx.starkInfo.publicsMap[i];
            if(publicInput.name == name) {
                cout << "Printing public: " << name << ": " << Goldilocks::toString(params.publicInputs[i]) << endl;
                if(returnValues) {
                    hintFieldInfo.size = 1;
                    hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
                    hintFieldInfo.values[0] = params.publicInputs[i];
                    hintFieldInfo.fieldType = HintFieldType::Field;
                    hintFieldInfo.offset = 1;
                }
                return hintFieldInfo;
            }
        }

        for(uint64_t i = 0; i < setupCtx.starkInfo.subproofValuesMap.size(); ++i) {
            PolMap subproofValue = setupCtx.starkInfo.subproofValuesMap[i];
            if(subproofValue.name == name) {
                cout << "Printing subproofValue: " << name << ": ";
                cout << "[" << Goldilocks::toString(params.subproofValues[i*FIELD_EXTENSION]) << " , " << Goldilocks::toString(params.subproofValues[i*FIELD_EXTENSION + 1]) << " , " << Goldilocks::toString(params.subproofValues[i*FIELD_EXTENSION + 2]) << "]" << endl;
                if(returnValues) {
                    hintFieldInfo.size = FIELD_EXTENSION;
                    hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
                    hintFieldInfo.fieldType = HintFieldType::FieldExtended;
                    hintFieldInfo.offset = FIELD_EXTENSION;
                    std::memcpy(hintFieldInfo.values, &params.subproofValues[FIELD_EXTENSION*i], FIELD_EXTENSION * sizeof(Goldilocks::Element));
                }
                return hintFieldInfo;
            }
        }

        zklog.info("Unknown name " + name);
        exitProcess();
        exit(-1);
    }
};

#endif