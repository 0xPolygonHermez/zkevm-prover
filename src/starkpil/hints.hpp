#include "expressions_ctx.hpp"

typedef enum {
    Field = 0,
    FieldExtended = 1,
    Column = 2,
    ColumnExtended = 3,
    String = 4,
} HintFieldType;

struct HintFieldInfo {
    uint64_t size; // Destination size (in Goldilocks elements)
    uint8_t offset;
    HintFieldType fieldType;
    Goldilocks::Element* values;
    const char* stringValue;
};


void getPolynomial(SetupCtx& setupCtx, Goldilocks::Element *buffer, Goldilocks::Element *dest, bool committed, uint64_t idPol, bool domainExtended) {
    PolMap polInfo = committed ? setupCtx.starkInfo.cmPolsMap[idPol] : setupCtx.starkInfo.constPolsMap[idPol];
    uint64_t deg = domainExtended ? 1 << setupCtx.starkInfo.starkStruct.nBitsExt : 1 << setupCtx.starkInfo.starkStruct.nBits;
    uint64_t dim = polInfo.dim;
    std::string stage = committed ? "cm" + to_string(polInfo.stage) : "const";
    uint64_t nCols = setupCtx.starkInfo.mapSectionsN[stage];
    uint64_t offset = setupCtx.starkInfo.mapOffsets[std::make_pair(stage, domainExtended)];
    offset += polInfo.stagePos;
    Goldilocks::Element *pols = committed ? buffer : domainExtended ? setupCtx.constPols.pConstPolsAddressExtended : setupCtx.constPols.pConstPolsAddress;
    Polinomial pol = Polinomial(&pols[offset], deg, dim, nCols, std::to_string(idPol));

    for(uint64_t j = 0; j < deg; ++j) {
        std::memcpy(&dest[j*dim], pol[j], dim * sizeof(Goldilocks::Element));
    }
}

void setPolynomial(SetupCtx& setupCtx, Goldilocks::Element *buffer, Goldilocks::Element *values, uint64_t idPol, bool domainExtended) {
    PolMap polInfo = setupCtx.starkInfo.cmPolsMap[idPol];
    uint64_t deg = domainExtended ? 1 << setupCtx.starkInfo.starkStruct.nBitsExt : 1 << setupCtx.starkInfo.starkStruct.nBits;
    uint64_t dim = polInfo.dim;
    std::string stage = "cm" + to_string(polInfo.stage);
    uint64_t nCols = setupCtx.starkInfo.mapSectionsN[stage];
    uint64_t offset = setupCtx.starkInfo.mapOffsets[std::make_pair(stage, domainExtended)];
    offset += polInfo.stagePos;
    Polinomial pol = Polinomial(&buffer[offset], deg, dim, nCols, std::to_string(idPol));

    for(uint64_t j = 0; j < deg; ++j) {
        std::memcpy(pol[j], &values[j*dim], dim * sizeof(Goldilocks::Element));
    }
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

void printRow(SetupCtx& setupCtx, Goldilocks::Element* buffer, uint64_t stage, uint64_t row) {
    Goldilocks::Element *pol = &buffer[setupCtx.starkInfo.mapOffsets[std::make_pair("cm" + to_string(stage), false)] + setupCtx.starkInfo.mapSectionsN["cm" + to_string(stage)] * row];
    cout << "Values at row " << row << " = {" << endl;
    bool first = true;
    for(uint64_t i = 0; i < setupCtx.starkInfo.cmPolsMap.size(); ++i) {
        PolMap cmPol = setupCtx.starkInfo.cmPolsMap[i];
        if(cmPol.stage == stage) {
            if(first) {
                first = false;
            } else {
                cout << endl;
            }
            cout << "    " << cmPol.name;
            if(cmPol.lengths.size() > 0) {
                cout << "[";
                for(uint64_t i = 0; i < cmPol.lengths.size(); ++i) {
                    cout << cmPol.lengths[i];
                    if(i != cmPol.lengths.size() - 1) cout << ", ";
                }
                cout << "]";
            }
            cout << ": ";
            if(cmPol.dim == 1) {
                cout << Goldilocks::toString(pol[cmPol.stagePos]) << ",";
            } else {
                cout << "[" << Goldilocks::toString(pol[cmPol.stagePos]) << ", " << Goldilocks::toString(pol[cmPol.stagePos + 1]) << ", " << Goldilocks::toString(pol[cmPol.stagePos + 2]) << "],";
            }
        }
    }
    cout << endl;
    cout << "}" << endl;
}

void printColById(SetupCtx& setupCtx, Goldilocks::Element* buffer, bool committed, uint64_t polId, uint64_t firstPrintValue = 0, uint64_t lastPrintValue = 0)
{   
    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    PolMap polInfo = committed ? setupCtx.starkInfo.cmPolsMap[polId] : setupCtx.starkInfo.constPolsMap[polId];
    Goldilocks::Element *pols = committed ? buffer : setupCtx.constPols.pConstPolsAddress;
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

HintFieldInfo printByName(SetupCtx& setupCtx, Goldilocks::Element* buffer, Goldilocks::Element* publicInputs, Goldilocks::Element* challenges, Goldilocks::Element *subproofValues, string name, uint64_t *lengths, uint64_t firstPrintValue, uint64_t lastPrintValue, bool returnValues) {
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
            printColById(setupCtx, buffer, true, i, firstPrintValue, lastPrintValue);
            if(returnValues) {
                hintFieldInfo.size = cmPol.dim * N;
                hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
                hintFieldInfo.fieldType = cmPol.dim == 1 ? HintFieldType::Column : HintFieldType::ColumnExtended;
                hintFieldInfo.offset = cmPol.dim;
                getPolynomial(setupCtx, buffer, hintFieldInfo.values, true, i, false);
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
            printColById(setupCtx, buffer, false, i, firstPrintValue, lastPrintValue);
            if(returnValues) {
                hintFieldInfo.size = N;
                hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
                hintFieldInfo.fieldType = HintFieldType::Column;
                hintFieldInfo.offset = 1;
                getPolynomial(setupCtx, buffer, hintFieldInfo.values, false, i, false);
            }
            return hintFieldInfo;
        } 
    }

    for(uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); ++i) {
        PolMap challenge = setupCtx.starkInfo.challengesMap[i];
        if(challenge.name == name) {
            cout << "Printing challenge: " << name << " (stage " << challenge.stage << " and id " << challenge.stageId << "): ";
            cout << "[" << Goldilocks::toString(challenges[i*FIELD_EXTENSION]) << " , " << Goldilocks::toString(challenges[i*FIELD_EXTENSION + 1]) << " , " << Goldilocks::toString(challenges[i*FIELD_EXTENSION + 2]) << "]" << endl;
            if(returnValues) {
                hintFieldInfo.size = FIELD_EXTENSION;
                hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
                hintFieldInfo.fieldType = HintFieldType::FieldExtended;
                hintFieldInfo.offset = FIELD_EXTENSION;
                std::memcpy(hintFieldInfo.values, &challenges[FIELD_EXTENSION*i], FIELD_EXTENSION * sizeof(Goldilocks::Element));
            }
            return hintFieldInfo;
        }
    }

    for(uint64_t i = 0; i < setupCtx.starkInfo.publicsMap.size(); ++i) {
        PolMap publicInput = setupCtx.starkInfo.publicsMap[i];
        if(publicInput.name == name) {
            cout << "Printing public: " << name << ": " << Goldilocks::toString(publicInputs[i]) << endl;
            if(returnValues) {
                hintFieldInfo.size = 1;
                hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
                hintFieldInfo.values[0] = publicInputs[i];
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
            cout << "[" << Goldilocks::toString(subproofValues[i*FIELD_EXTENSION]) << " , " << Goldilocks::toString(subproofValues[i*FIELD_EXTENSION + 1]) << " , " << Goldilocks::toString(subproofValues[i*FIELD_EXTENSION + 2]) << "]" << endl;
            if(returnValues) {
                hintFieldInfo.size = FIELD_EXTENSION;
                hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
                hintFieldInfo.fieldType = HintFieldType::FieldExtended;
                hintFieldInfo.offset = FIELD_EXTENSION;
                std::memcpy(hintFieldInfo.values, &subproofValues[FIELD_EXTENSION*i], FIELD_EXTENSION * sizeof(Goldilocks::Element));
            }
            return hintFieldInfo;
        }
    }

    zklog.info("Unknown name " + name);
    exitProcess();
    exit(-1);
}


void setSubproofValue(Goldilocks::Element *subproofValues, Goldilocks::Element *value, uint64_t subproofValueId) {
    std::memcpy(&subproofValues[FIELD_EXTENSION*subproofValueId], value, FIELD_EXTENSION * sizeof(Goldilocks::Element));
}

HintFieldInfo getHintField(SetupCtx& setupCtx, Goldilocks::Element *buffer, Goldilocks::Element *publicInputs, Goldilocks::Element *challenges, Goldilocks::Element *subproofValues, Goldilocks::Element *evals, uint64_t hintId, std::string hintFieldName, bool dest, bool inverse, bool print_expression) {
    StepsParams params {
        pols : buffer,
        publicInputs,
        challenges,
        subproofValues,
        evals,
        xDivXSub : nullptr,
    };

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
        if(!dest) {
            getPolynomial(setupCtx, buffer, hintFieldInfo.values, true, hintField->id, false);
            if(inverse) {
                zklog.error("Inverse not supported still for polynomials");
                exitProcess();
            }
        } else {
            memset((uint8_t *)hintFieldInfo.values, 0, hintFieldInfo.size * sizeof(Goldilocks::Element));
        }
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
        getPolynomial(setupCtx, buffer, hintFieldInfo.values, false, hintField->id, false);
        if(inverse) {
            zklog.error("Inverse not supported still for polynomials");
            exitProcess();
        }
    } else if (hintField->operand == opType::tmp) {
        uint64_t dim = setupCtx.expressionsBin.expressionsInfo[hintField->id].destDim;
        hintFieldInfo.size = deg*dim;
        hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
        hintFieldInfo.fieldType = dim == 1 ? HintFieldType::Column : HintFieldType::ColumnExtended;
        hintFieldInfo.offset = dim;
        if(print_expression && setupCtx.expressionsBin.expressionsInfo[hintField->id].line != "") {
            cout << "the expression with id: " << hintField->id << " " << setupCtx.expressionsBin.expressionsInfo[hintField->id].line << endl;
        }
        ExpressionsAvx expressionAvx(setupCtx);
        expressionAvx.calculateExpression(params, hintFieldInfo.values, hintField->id, inverse);
    } else if (hintField->operand == opType::public_) {
        hintFieldInfo.size = 1;
        hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
        hintFieldInfo.values[0] = inverse ? Goldilocks::inv(publicInputs[hintField->id]) : publicInputs[hintField->id];
        hintFieldInfo.fieldType = HintFieldType::Field;
        hintFieldInfo.offset = 1;
        if(print_expression) cout << "public input " << setupCtx.starkInfo.publicsMap[hintField->id].name << endl;
    } else if (hintField->operand == opType::number) {
        hintFieldInfo.size = 1;
        hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
        hintFieldInfo.values[0] = inverse ? Goldilocks::inv(Goldilocks::fromU64(hintField->value)) : Goldilocks::fromU64(hintField->value);
        hintFieldInfo.fieldType = HintFieldType::Field;
        hintFieldInfo.offset = 1;
        if(print_expression) cout << "number " << hintField->value << endl;
    } else if (hintField->operand == opType::subproofvalue) {
        hintFieldInfo.size = FIELD_EXTENSION;
        hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
        hintFieldInfo.fieldType = HintFieldType::FieldExtended;
        hintFieldInfo.offset = FIELD_EXTENSION;
        if(print_expression) cout << "subproofValue " << setupCtx.starkInfo.subproofValuesMap[hintField->id].name << endl;
        if(!dest) {
            if(inverse)  {
                Goldilocks3::inv((Goldilocks3::Element *)hintFieldInfo.values, (Goldilocks3::Element *)&subproofValues[FIELD_EXTENSION*hintField->id]);
            } else {
                std::memcpy(hintFieldInfo.values, &subproofValues[FIELD_EXTENSION*hintField->id], FIELD_EXTENSION * sizeof(Goldilocks::Element));
            }
        } else {
            memset((uint8_t *)hintFieldInfo.values, 0, hintFieldInfo.size * sizeof(Goldilocks::Element));
        }
    } else if (hintField->operand == opType::challenge) {
        hintFieldInfo.size = FIELD_EXTENSION;
        hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
        hintFieldInfo.fieldType = HintFieldType::FieldExtended;
        hintFieldInfo.offset = FIELD_EXTENSION;
        if(print_expression) cout << "challenge " << setupCtx.starkInfo.challengesMap[hintField->id].name << endl;
        if(inverse) {
            Goldilocks3::inv((Goldilocks3::Element *)hintFieldInfo.values, (Goldilocks3::Element *)&challenges[FIELD_EXTENSION*hintField->id]);
        } else {
            std::memcpy(hintFieldInfo.values, &challenges[FIELD_EXTENSION*hintField->id], FIELD_EXTENSION * sizeof(Goldilocks::Element));
        }
    } else if (hintField->operand == opType::string_) {
        hintFieldInfo.size = 0;
        hintFieldInfo.values = nullptr;
        hintFieldInfo.fieldType = HintFieldType::String;
        hintFieldInfo.stringValue = hintField->stringValue.c_str();
        hintFieldInfo.offset = 0;
        if(print_expression) cout << "string " << hintField->stringValue << endl;
    } else {
        zklog.error("Unknown HintFieldType");
        exitProcess();
        exit(-1);
    }

    if(print_expression) cout << "--------------------------------------------------------" << endl;

    return hintFieldInfo;
}

uint64_t setHintField(SetupCtx& setupCtx, Goldilocks::Element *buffer, Goldilocks::Element *subproofValues, Goldilocks::Element* values, uint64_t hintId, std::string hintFieldName) {
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
        setPolynomial(setupCtx, buffer, values, hintField->id, false);
    } else if(hintField->operand == opType::subproofvalue) {
        setSubproofValue(subproofValues, values, hintField->id);
    } else {
        zklog.error("Only committed pols and subproofvalues can be set");
        exitProcess();
        exit(-1);  
    }

    return hintField->id;
}