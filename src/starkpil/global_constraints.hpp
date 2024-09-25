#ifndef GLOBAL_CONSTRAINTS_HPP
#define GLOBAL_CONSTRAINTS_HPP
#include "timer.hpp"
#include "goldilocks_base_field.hpp"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

const int GLOBAL_CONSTRAINTS_SECTION = 2;

bool verifyGlobalConstraint(Goldilocks::Element* publics, Goldilocks::Element** subproofValues, ParserArgs &parserArgs, ParserParams &parserParams) {

    uint8_t* ops = &parserArgs.ops[parserParams.opsOffset];
    uint16_t* args = &parserArgs.args[parserParams.argsOffset];
    uint64_t* numbers = &parserArgs.numbers[parserParams.numbersOffset];

    uint64_t i_args = 0;

    Goldilocks::Element tmp1[parserParams.nTemp1];
    Goldilocks::Element tmp3[parserParams.nTemp3*FIELD_EXTENSION];

    Goldilocks::Element numbers_[parserParams.nNumbers];
    for(uint64_t i = 0; i < parserParams.nNumbers; ++i) {
        numbers_[i] = Goldilocks::fromU64(numbers[i]);
    }

    for (uint64_t kk = 0; kk < parserParams.nOps; ++kk) {
        switch (ops[kk]) {
            case 0: {
                // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: tmp1
                Goldilocks::op_pack(1, args[i_args], &tmp1[args[i_args + 1]], &tmp1[args[i_args + 2]], &tmp1[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 1: {
                // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: public
                Goldilocks::op_pack(1, args[i_args], &tmp1[args[i_args + 1]], &tmp1[args[i_args + 2]], &publics[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 2: {
                // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: number
                Goldilocks::op_pack(1, args[i_args], &tmp1[args[i_args + 1]], &tmp1[args[i_args + 2]], &numbers_[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 3: {
                // OPERATION WITH DEST: tmp1 - SRC0: public - SRC1: public
                Goldilocks::op_pack(1, args[i_args], &tmp1[args[i_args + 1]], &publics[args[i_args + 2]], &publics[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 4: {
                // OPERATION WITH DEST: tmp1 - SRC0: public - SRC1: number
                Goldilocks::op_pack(1, args[i_args], &tmp1[args[i_args + 1]], &publics[args[i_args + 2]], &numbers_[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 5: {
                // OPERATION WITH DEST: tmp1 - SRC0: number - SRC1: number
                Goldilocks::op_pack(1, args[i_args], &tmp1[args[i_args + 1]], &numbers_[args[i_args + 2]], &numbers_[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 6: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: tmp1
                Goldilocks3::op_31_pack(1, args[i_args], &tmp3[args[i_args + 1] * FIELD_EXTENSION], &tmp3[args[i_args + 2] * FIELD_EXTENSION], &tmp1[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 7: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: public
                Goldilocks3::op_31_pack(1, args[i_args], &tmp3[args[i_args + 1] * FIELD_EXTENSION], &tmp3[args[i_args + 2] * FIELD_EXTENSION], &publics[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 8: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: number
                Goldilocks3::op_31_pack(1, args[i_args], &tmp3[args[i_args + 1] * FIELD_EXTENSION], &tmp3[args[i_args + 2] * FIELD_EXTENSION], &numbers_[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 9: {
                // OPERATION WITH DEST: tmp3 - SRC0: subproofValue - SRC1: tmp1
                Goldilocks3::op_31_pack(1, args[i_args], &tmp3[args[i_args + 1] * FIELD_EXTENSION], &subproofValues[args[i_args + 2]][args[i_args + 3] * FIELD_EXTENSION], &tmp1[args[i_args + 4]]);
                i_args += 5;
                break;
            }
            case 10: {
                // OPERATION WITH DEST: tmp3 - SRC0: subproofValue - SRC1: public
                Goldilocks3::op_31_pack(1, args[i_args], &tmp3[args[i_args + 1] * FIELD_EXTENSION], &subproofValues[args[i_args + 2]][args[i_args + 3] * FIELD_EXTENSION], &publics[args[i_args + 4]]);
                i_args += 5;
                break;
            }
            case 11: {
                // OPERATION WITH DEST: tmp3 - SRC0: subproofValue - SRC1: number
                Goldilocks3::op_31_pack(1, args[i_args], &tmp3[args[i_args + 1] * FIELD_EXTENSION], &subproofValues[args[i_args + 2]][args[i_args + 3] * FIELD_EXTENSION], &numbers_[args[i_args + 4]]);
                i_args += 5;
                break;
            }
            case 12: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: tmp3
                Goldilocks3::op_pack(1, args[i_args], &tmp3[args[i_args + 1] * FIELD_EXTENSION], &tmp3[args[i_args + 2] * FIELD_EXTENSION], &tmp3[args[i_args + 3] * FIELD_EXTENSION]);
                i_args += 4;
                break;
            }
            case 13: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: subproofValue
                Goldilocks3::op_pack(1, args[i_args], &tmp3[args[i_args + 1] * FIELD_EXTENSION], &tmp3[args[i_args + 2] * FIELD_EXTENSION], &subproofValues[args[i_args + 3]][args[i_args + 4] * FIELD_EXTENSION]);
                i_args += 5;
                break;
            }
            case 14: {
                // OPERATION WITH DEST: tmp3 - SRC0: subproofValue - SRC1: subproofValue
                Goldilocks3::op_pack(1, args[i_args], &tmp3[args[i_args + 1] * FIELD_EXTENSION], &subproofValues[args[i_args + 2]][args[i_args + 3] * FIELD_EXTENSION], &subproofValues[args[i_args + 4]][args[i_args + 5] * FIELD_EXTENSION]);
                i_args += 6;
                break;
            }
            default: {
                std::cout << " Wrong operation!" << std::endl;
                exit(1);
            }
        }
    }

    if (i_args != parserParams.nArgs) std::cout << " " << i_args << " - " << parserParams.nArgs << std::endl;
    assert(i_args == parserParams.nArgs);

    bool isValidConstraint = true;
    if(parserParams.destDim == 1) {
        if(!Goldilocks::isZero(tmp1[parserParams.destId])) {
            cout << "Global constraint check failed with value: " << Goldilocks::toString(tmp1[parserParams.destId]) << endl;
            isValidConstraint = false;
        }
    } else {
        for(uint64_t i = 0; i < FIELD_EXTENSION; ++i) {
            if(!Goldilocks::isZero(tmp3[parserParams.destId*FIELD_EXTENSION + i])) {
                cout << "Global constraint check failed with value: [" << Goldilocks::toString(tmp3[parserParams.destId*FIELD_EXTENSION]) << ", " << Goldilocks::toString(tmp3[parserParams.destId*FIELD_EXTENSION + 1]) << ", " << Goldilocks::toString(tmp3[parserParams.destId*FIELD_EXTENSION + 2]) << "]" << endl;
                isValidConstraint = false;
                break;
            }
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

  
bool verifyGlobalConstraints(string globalConstraintsBin, Goldilocks::Element* publicInputs, Goldilocks::Element** airgroupValues)
{
    std::unique_ptr<BinFileUtils::BinFile> globalConstraintsBinFile = BinFileUtils::openExisting(globalConstraintsBin, "chps", 1);
    BinFileUtils::BinFile *binFile = globalConstraintsBinFile.get();

    binFile->startReadSection(GLOBAL_CONSTRAINTS_SECTION);

    uint32_t nOpsDebug = binFile->readU32LE();
    uint32_t nArgsDebug = binFile->readU32LE();
    uint32_t nNumbersDebug = binFile->readU32LE();

    ParserArgs globalConstraintsArgs;

    std::vector<ParserParams> globalConstraintsInfo;

    globalConstraintsArgs.ops = new uint8_t[nOpsDebug];
    globalConstraintsArgs.args = new uint16_t[nArgsDebug];
    globalConstraintsArgs.numbers = new uint64_t[nNumbersDebug];

    uint32_t nGlobalConstraints = binFile->readU32LE();

    for(uint64_t i = 0; i < nGlobalConstraints; ++i) {
        ParserParams parserParamsConstraint;

        parserParamsConstraint.destDim = binFile->readU32LE();
        parserParamsConstraint.destId = binFile->readU32LE();

        parserParamsConstraint.nTemp1 = binFile->readU32LE();
        parserParamsConstraint.nTemp3 = binFile->readU32LE();

        parserParamsConstraint.nOps = binFile->readU32LE();
        parserParamsConstraint.opsOffset = binFile->readU32LE();

        parserParamsConstraint.nArgs = binFile->readU32LE();
        parserParamsConstraint.argsOffset = binFile->readU32LE();

        parserParamsConstraint.nNumbers = binFile->readU32LE();
        parserParamsConstraint.numbersOffset = binFile->readU32LE();

        parserParamsConstraint.line = binFile->readString();

        globalConstraintsInfo.push_back(parserParamsConstraint);
    }


    for(uint64_t j = 0; j < nOpsDebug; ++j) {
        globalConstraintsArgs.ops[j] = binFile->readU8LE();
    }
    for(uint64_t j = 0; j < nArgsDebug; ++j) {
        globalConstraintsArgs.args[j] = binFile->readU16LE();
    }
    for(uint64_t j = 0; j < nNumbersDebug; ++j) {
        globalConstraintsArgs.numbers[j] = binFile->readU64LE();
    }

    binFile->endReadSection();

    bool validGlobalConstraints = true;
    for(uint64_t i = 0; i < nGlobalConstraints; ++i) {
        TimerLog(CHECKING_CONSTRAINT);
        cout << "--------------------------------------------------------" << endl;
        cout << globalConstraintsInfo[i].line << endl;
        cout << "--------------------------------------------------------" << endl;
        if(!verifyGlobalConstraint(publicInputs, airgroupValues, globalConstraintsArgs, globalConstraintsInfo[i])) {
            validGlobalConstraints = false;
        };
    }

    return validGlobalConstraints;
}

#endif