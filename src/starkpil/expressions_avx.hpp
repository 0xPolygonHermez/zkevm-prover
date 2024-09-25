#ifndef EXPRESSIONS_AVX_HPP
#define EXPRESSIONS_AVX_HPP
#include "expressions_ctx.hpp"

class ExpressionsAvx : public ExpressionsCtx {
public:
    uint64_t nrowsPack = 4;
    uint64_t nCols;
    vector<uint64_t> nColsStages;
    vector<uint64_t> nColsStagesAcc;
    vector<uint64_t> offsetsStages;
    ExpressionsAvx(SetupCtx& setupCtx) : ExpressionsCtx(setupCtx) {};

    void setBufferTInfo(uint64_t stage, bool domainExtended, uint64_t expId) {
        uint64_t nOpenings = setupCtx.starkInfo.openingPoints.size();
        offsetsStages.resize((setupCtx.starkInfo.nStages + 2)*nOpenings + 1);
        nColsStages.resize((setupCtx.starkInfo.nStages + 2)*nOpenings + 1);
        nColsStagesAcc.resize((setupCtx.starkInfo.nStages + 2)*nOpenings + 1);

        nCols = setupCtx.starkInfo.nConstants;
        uint64_t ns = setupCtx.starkInfo.nStages + 2;
        for(uint64_t o = 0; o < nOpenings; ++o) {
            for(uint64_t stage = 0; stage <= ns; ++stage) {
                std::string section = stage == 0 ? "const" : "cm" + to_string(stage);
                offsetsStages[(setupCtx.starkInfo.nStages + 2)*o + stage] = setupCtx.starkInfo.mapOffsets[std::make_pair(section, domainExtended)];
                nColsStages[(setupCtx.starkInfo.nStages + 2)*o + stage] = setupCtx.starkInfo.mapSectionsN[section];
                nColsStagesAcc[(setupCtx.starkInfo.nStages + 2)*o + stage] = stage == 0 && o == 0 ? 0 : nColsStagesAcc[(setupCtx.starkInfo.nStages + 2)*o + stage - 1] + nColsStages[stage - 1];
            }
        }

        nColsStagesAcc[(setupCtx.starkInfo.nStages + 2)*nOpenings] = nColsStagesAcc[(setupCtx.starkInfo.nStages + 2)*nOpenings - 1] + nColsStages[(setupCtx.starkInfo.nStages + 2)*nOpenings - 1];
        if(expId == setupCtx.starkInfo.cExpId) {
            nCols = nColsStagesAcc[(setupCtx.starkInfo.nStages + 2)*nOpenings] + setupCtx.starkInfo.boundaries.size() + 1;
        } else if(expId == setupCtx.starkInfo.friExpId) {
            nCols = nColsStagesAcc[(setupCtx.starkInfo.nStages + 2)*nOpenings] + nOpenings*FIELD_EXTENSION;
        } else {
            nCols = nColsStagesAcc[(setupCtx.starkInfo.nStages + 2)*nOpenings] + 1;
        }
    }

    inline void loadPolynomials(StepsParams& params, ParserArgs &parserArgs, ParserParams &parserParams, __m256i *bufferT_, uint64_t row, bool domainExtended) {
        uint64_t nOpenings = setupCtx.starkInfo.openingPoints.size();
        uint64_t domainSize = domainExtended ? 1 << setupCtx.starkInfo.starkStruct.nBitsExt : 1 << setupCtx.starkInfo.starkStruct.nBits;

        uint64_t extendBits = (setupCtx.starkInfo.starkStruct.nBitsExt - setupCtx.starkInfo.starkStruct.nBits);
        int64_t extend = domainExtended ? (1 << extendBits) : 1;
        uint64_t nextStrides[nOpenings];
        for(uint64_t i = 0; i < nOpenings; ++i) {
            uint64_t opening = setupCtx.starkInfo.openingPoints[i] < 0 ? setupCtx.starkInfo.openingPoints[i] + domainSize : setupCtx.starkInfo.openingPoints[i];
            nextStrides[i] = opening * extend;
        }

        Goldilocks::Element *constPols = domainExtended ? setupCtx.constPols.pConstPolsAddressExtended : setupCtx.constPols.pConstPolsAddress;

        uint16_t* cmPolsUsed = &parserArgs.cmPolsIds[parserParams.cmPolsOffset];
        uint16_t* constPolsUsed = &parserArgs.constPolsIds[parserParams.constPolsOffset];

        Goldilocks::Element bufferT[nOpenings*nrowsPack];

        for(uint64_t k = 0; k < parserParams.nConstPolsUsed; ++k) {
            uint64_t id = constPolsUsed[k];
            for(uint64_t o = 0; o < nOpenings; ++o) {
                for(uint64_t j = 0; j < nrowsPack; ++j) {
                    uint64_t l = (row + j + nextStrides[o]) % domainSize;
                    bufferT[nrowsPack*o + j] = constPols[l * nColsStages[0] + id];
                }
                Goldilocks::load_avx(bufferT_[nColsStagesAcc[(setupCtx.starkInfo.nStages + 2)*o] + id], &bufferT[nrowsPack*o]);
            }
        }

        for(uint64_t k = 0; k < parserParams.nCmPolsUsed; ++k) {
            uint64_t polId = cmPolsUsed[k];
            PolMap polInfo = setupCtx.starkInfo.cmPolsMap[polId];
            uint64_t stage = polInfo.stage;
            uint64_t stagePos = polInfo.stagePos;
            for(uint64_t d = 0; d < polInfo.dim; ++d) {
                for(uint64_t o = 0; o < nOpenings; ++o) {
                    for(uint64_t j = 0; j < nrowsPack; ++j) {
                        uint64_t l = (row + j + nextStrides[o]) % domainSize;
                        bufferT[nrowsPack*o + j] = params.pols[offsetsStages[stage] + l * nColsStages[stage] + stagePos + d];
                    }
                    Goldilocks::load_avx(bufferT_[nColsStagesAcc[(setupCtx.starkInfo.nStages + 2)*o + stage] + (stagePos + d)], &bufferT[nrowsPack*o]);
                }
            }
        }

        if(parserParams.expId == setupCtx.starkInfo.cExpId) {
            for(uint64_t j = 0; j < nrowsPack; ++j) {
                bufferT[j] = setupCtx.constPols.x_2ns[row + j];
            }
            Goldilocks::load_avx(bufferT_[nColsStagesAcc[(setupCtx.starkInfo.nStages + 2)*nOpenings]], &bufferT[0]);
            for(uint64_t d = 0; d < setupCtx.starkInfo.boundaries.size(); ++d) {
                for(uint64_t j = 0; j < nrowsPack; ++j) {
                    bufferT[j] = setupCtx.constPols.zi[row + j + d*domainSize];
                }
                Goldilocks::load_avx(bufferT_[nColsStagesAcc[(setupCtx.starkInfo.nStages + 2)*nOpenings] + 1 + d], &bufferT[0]);
            }
        } else if(parserParams.expId == setupCtx.starkInfo.friExpId) {
            for(uint64_t d = 0; d < setupCtx.starkInfo.openingPoints.size(); ++d) {
               for(uint64_t k = 0; k < FIELD_EXTENSION; ++k) {
                    for(uint64_t j = 0; j < nrowsPack; ++j) {
                        bufferT[j] = params.xDivXSub[(row + j + d*domainSize)*FIELD_EXTENSION + k];
                    }
                    Goldilocks::load_avx(bufferT_[nColsStagesAcc[(setupCtx.starkInfo.nStages + 2)*nOpenings] + d*FIELD_EXTENSION + k], &bufferT[0]);
                }
            }
        } else {
            for(uint64_t j = 0; j < nrowsPack; ++j) {
                bufferT[j] = setupCtx.constPols.x_n[row + j];
            }
            Goldilocks::load_avx(bufferT_[nColsStagesAcc[(setupCtx.starkInfo.nStages + 2)*nOpenings]], &bufferT[0]);
        }
    }

    inline void storePolynomial(Goldilocks::Element* dest, ParserParams& parserParams, uint64_t row, __m256i* tmp1, Goldilocks3::Element_avx* tmp3, bool inverse) {
        if(parserParams.destDim == 1) {
            Goldilocks::store_avx(&dest[row], tmp1[parserParams.destId]);
            if(inverse) {
                for(uint64_t i = 0; i < nrowsPack; ++i) {
                    Goldilocks::inv(dest[row + i], dest[row + i]);
                }
            }
        } else {
            Goldilocks::store_avx(&dest[row*FIELD_EXTENSION], uint64_t(FIELD_EXTENSION), tmp3[parserParams.destId][0]);
            Goldilocks::store_avx(&dest[row*FIELD_EXTENSION + 1], uint64_t(FIELD_EXTENSION), tmp3[parserParams.destId][1]);
            Goldilocks::store_avx(&dest[row*FIELD_EXTENSION + 2], uint64_t(FIELD_EXTENSION), tmp3[parserParams.destId][2]);
            if(inverse) {
                for(uint64_t i = 0; i < nrowsPack; ++i) {
                    Goldilocks3::inv((Goldilocks3::Element *)&dest[(row + i)*FIELD_EXTENSION], (Goldilocks3::Element *)&dest[(row + i)*FIELD_EXTENSION]);
                }
            }
        }
    }

    inline void storeImPolynomials(StepsParams &params, __m256i *bufferT_, uint64_t row) {
        auto openingPointIndex = std::find(setupCtx.starkInfo.openingPoints.begin(), setupCtx.starkInfo.openingPoints.end(), 0) - setupCtx.starkInfo.openingPoints.begin();

        auto firstImPol = std::find_if(setupCtx.starkInfo.cmPolsMap.begin(), setupCtx.starkInfo.cmPolsMap.end(), [](const PolMap& s) { return s.imPol; });

        if(firstImPol != setupCtx.starkInfo.cmPolsMap.end()) {
            uint64_t firstImPolPos = firstImPol->stagePos;
            uint64_t stage = setupCtx.starkInfo.nStages;
            for(uint64_t k = firstImPolPos; k < nColsStages[stage]; ++k) {
                Goldilocks::store_avx(&params.pols[offsetsStages[stage] + k + row * nColsStages[stage]], nColsStages[stage], bufferT_[nColsStagesAcc[(setupCtx.starkInfo.nStages + 2)*openingPointIndex + stage] + k]);
            }
        }
    }

    inline void printTmp1(uint64_t row, __m256i tmp) {
        Goldilocks::Element dest[nrowsPack];
        Goldilocks::store_avx(dest, tmp);
        for(uint64_t i = 0; i < 1; ++i) {
            cout << "Value at row " << row + i << " is " << Goldilocks::toString(dest[i]) << endl;
        }
    }

    inline void printTmp3(uint64_t row, Goldilocks3::Element_avx tmp) {
        Goldilocks::Element dest[FIELD_EXTENSION*nrowsPack];
        Goldilocks::store_avx(&dest[0], uint64_t(FIELD_EXTENSION), tmp[0]);
        Goldilocks::store_avx(&dest[1], uint64_t(FIELD_EXTENSION), tmp[1]);
        Goldilocks::store_avx(&dest[2], uint64_t(FIELD_EXTENSION), tmp[2]);
        for(uint64_t i = 0; i < 1; ++i) {
            cout << "Value at row " << row + i << " is [" << Goldilocks::toString(dest[FIELD_EXTENSION*i]) << ", " << Goldilocks::toString(dest[FIELD_EXTENSION*i + 1]) << ", " << Goldilocks::toString(dest[FIELD_EXTENSION*i + 2]) << "]" << endl;
        }
    }

    inline void printCommit(uint64_t row, __m256i* bufferT, bool extended) {
        if(extended) {
            Goldilocks::Element dest[FIELD_EXTENSION*nrowsPack];
            Goldilocks::store_avx(&dest[0], uint64_t(FIELD_EXTENSION), bufferT[0]);
            Goldilocks::store_avx(&dest[1], uint64_t(FIELD_EXTENSION), bufferT[setupCtx.starkInfo.openingPoints.size()]);
            Goldilocks::store_avx(&dest[2], uint64_t(FIELD_EXTENSION), bufferT[2*setupCtx.starkInfo.openingPoints.size()]);
            for(uint64_t i = 0; i < 1; ++i) {
                cout << "Value at row " << row + i << " is [" << Goldilocks::toString(dest[FIELD_EXTENSION*i]) << ", " << Goldilocks::toString(dest[FIELD_EXTENSION*i + 1]) << ", " << Goldilocks::toString(dest[FIELD_EXTENSION*i + 2]) << "]" << endl;
            }
        } else {
            Goldilocks::Element dest[nrowsPack];
            Goldilocks::store_avx(&dest[0], bufferT[0]);
            for(uint64_t i = 0; i < 1; ++i) {
                cout << "Value at row " << row + i << " is " << Goldilocks::toString(dest[i]) << endl;
            }
        }
    }

    void calculateExpressions(StepsParams& params, Goldilocks::Element *dest, ParserArgs &parserArgs, ParserParams &parserParams, bool domainExtended, bool inverse, bool imPols) override {
        uint8_t* ops = &parserArgs.ops[parserParams.opsOffset];
        uint16_t* args = &parserArgs.args[parserParams.argsOffset];
        uint64_t* numbers = &parserArgs.numbers[parserParams.numbersOffset];

        uint64_t nOpenings = setupCtx.starkInfo.openingPoints.size();
        uint64_t domainSize = domainExtended ? 1 << setupCtx.starkInfo.starkStruct.nBitsExt : 1 << setupCtx.starkInfo.starkStruct.nBits;

        setBufferTInfo(parserParams.stage, domainExtended, parserParams.expId);

        Goldilocks3::Element_avx challenges[setupCtx.starkInfo.challengesMap.size()];
        Goldilocks3::Element_avx challenges_ops[setupCtx.starkInfo.challengesMap.size()];
        for(uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); ++i) {
            challenges[i][0] = _mm256_set1_epi64x(params.challenges[i * FIELD_EXTENSION].fe);
            challenges[i][1] = _mm256_set1_epi64x(params.challenges[i * FIELD_EXTENSION + 1].fe);
            challenges[i][2] = _mm256_set1_epi64x(params.challenges[i * FIELD_EXTENSION + 2].fe);

            Goldilocks::Element challenges_aux[3];
            challenges_aux[0] = params.challenges[i * FIELD_EXTENSION] + params.challenges[i * FIELD_EXTENSION + 1];
            challenges_aux[1] = params.challenges[i * FIELD_EXTENSION] + params.challenges[i * FIELD_EXTENSION + 2];
            challenges_aux[2] = params.challenges[i * FIELD_EXTENSION + 1] + params.challenges[i * FIELD_EXTENSION + 2];
            challenges_ops[i][0] = _mm256_set1_epi64x(challenges_aux[0].fe);
            challenges_ops[i][1] =  _mm256_set1_epi64x(challenges_aux[1].fe);
            challenges_ops[i][2] =  _mm256_set1_epi64x(challenges_aux[2].fe);
        }

        __m256i numbers_[parserParams.nNumbers];
        for(uint64_t i = 0; i < parserParams.nNumbers; ++i) {
            numbers_[i] = _mm256_set1_epi64x(numbers[i]);
        }

        __m256i publics[setupCtx.starkInfo.nPublics];
        for(uint64_t i = 0; i < setupCtx.starkInfo.nPublics; ++i) {
            publics[i] = _mm256_set1_epi64x(params.publicInputs[i].fe);
        }

        Goldilocks3::Element_avx subproofValues[setupCtx.starkInfo.nSubProofValues];
        for(uint64_t i = 0; i < setupCtx.starkInfo.nSubProofValues; ++i) {
            subproofValues[i][0] = _mm256_set1_epi64x(params.subproofValues[i * FIELD_EXTENSION].fe);
            subproofValues[i][1] = _mm256_set1_epi64x(params.subproofValues[i * FIELD_EXTENSION + 1].fe);
            subproofValues[i][2] = _mm256_set1_epi64x(params.subproofValues[i * FIELD_EXTENSION + 2].fe);
        }

        Goldilocks3::Element_avx evals[setupCtx.starkInfo.evMap.size()];
        for(uint64_t i = 0; i < setupCtx.starkInfo.evMap.size(); ++i) {
            evals[i][0] = _mm256_set1_epi64x(params.evals[i * FIELD_EXTENSION].fe);
            evals[i][1] = _mm256_set1_epi64x(params.evals[i * FIELD_EXTENSION + 1].fe);
            evals[i][2] = _mm256_set1_epi64x(params.evals[i * FIELD_EXTENSION + 2].fe);
        }

    #pragma omp parallel for
        for (uint64_t i = 0; i < domainSize; i+= nrowsPack) {
            uint64_t i_args = 0;

            __m256i bufferT_[nOpenings*nCols];
            __m256i tmp1[parserParams.nTemp1];
            Goldilocks3::Element_avx tmp3[parserParams.nTemp3];

            loadPolynomials(params, parserArgs, parserParams, bufferT_, i, domainExtended);

            for (uint64_t kk = 0; kk < parserParams.nOps; ++kk) {
                switch (ops[kk]) {
            case 0: {
                // COPY commit1 to commit1
                Goldilocks::copy_avx(bufferT_[nColsStagesAcc[args[i_args]] + args[i_args + 1]], bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 1: {
                // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: commit1
                Goldilocks::op_avx(args[i_args], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], bufferT_[nColsStagesAcc[args[i_args + 5]] + args[i_args + 6]]);
                i_args += 7;
                break;
            }
            case 2: {
                // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: tmp1
                Goldilocks::op_avx(args[i_args], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], tmp1[args[i_args + 5]]);
                i_args += 6;
                break;
            }
            case 3: {
                // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: public
                Goldilocks::op_avx(args[i_args], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], publics[args[i_args + 5]]);
                i_args += 6;
                break;
            }
            case 4: {
                // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: number
                Goldilocks::op_avx(args[i_args], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], numbers_[args[i_args + 5]]);
                i_args += 6;
                break;
            }
            case 5: {
                // COPY tmp1 to commit1
                Goldilocks::copy_avx(bufferT_[nColsStagesAcc[args[i_args]] + args[i_args + 1]], tmp1[args[i_args + 2]]);
                i_args += 3;
                break;
            }
            case 6: {
                // OPERATION WITH DEST: commit1 - SRC0: tmp1 - SRC1: tmp1
                Goldilocks::op_avx(args[i_args], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], tmp1[args[i_args + 3]], tmp1[args[i_args + 4]]);
                i_args += 5;
                break;
            }
            case 7: {
                // OPERATION WITH DEST: commit1 - SRC0: tmp1 - SRC1: public
                Goldilocks::op_avx(args[i_args], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], tmp1[args[i_args + 3]], publics[args[i_args + 4]]);
                i_args += 5;
                break;
            }
            case 8: {
                // OPERATION WITH DEST: commit1 - SRC0: tmp1 - SRC1: number
                Goldilocks::op_avx(args[i_args], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], tmp1[args[i_args + 3]], numbers_[args[i_args + 4]]);
                i_args += 5;
                break;
            }
            case 9: {
                // COPY public to commit1
                Goldilocks::copy_avx(bufferT_[nColsStagesAcc[args[i_args]] + args[i_args + 1]], publics[args[i_args + 2]]);
                i_args += 3;
                break;
            }
            case 10: {
                // OPERATION WITH DEST: commit1 - SRC0: public - SRC1: public
                Goldilocks::op_avx(args[i_args], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], publics[args[i_args + 3]], publics[args[i_args + 4]]);
                i_args += 5;
                break;
            }
            case 11: {
                // OPERATION WITH DEST: commit1 - SRC0: public - SRC1: number
                Goldilocks::op_avx(args[i_args], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], publics[args[i_args + 3]], numbers_[args[i_args + 4]]);
                i_args += 5;
                break;
            }
            case 12: {
                // COPY number to commit1
                Goldilocks::copy_avx(bufferT_[nColsStagesAcc[args[i_args]] + args[i_args + 1]], numbers_[args[i_args + 2]]);
                i_args += 3;
                break;
            }
            case 13: {
                // OPERATION WITH DEST: commit1 - SRC0: number - SRC1: number
                Goldilocks::op_avx(args[i_args], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], numbers_[args[i_args + 3]], numbers_[args[i_args + 4]]);
                i_args += 5;
                break;
            }
            case 14: {
                // COPY commit1 to tmp1
                Goldilocks::copy_avx(tmp1[args[i_args]], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]]);
                i_args += 3;
                break;
            }
            case 15: {
                // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: commit1
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], bufferT_[nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]]);
                i_args += 6;
                break;
            }
            case 16: {
                // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: tmp1
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], tmp1[args[i_args + 4]]);
                i_args += 5;
                break;
            }
            case 17: {
                // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: public
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], publics[args[i_args + 4]]);
                i_args += 5;
                break;
            }
            case 18: {
                // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: number
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], numbers_[args[i_args + 4]]);
                i_args += 5;
                break;
            }
            case 19: {
                // COPY tmp1 to tmp1
                Goldilocks::copy_avx(tmp1[args[i_args]], tmp1[args[i_args + 1]]);
                i_args += 2;
                break;
            }
            case 20: {
                // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: tmp1
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], tmp1[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 21: {
                // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: public
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], publics[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 22: {
                // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: number
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], numbers_[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 23: {
                // COPY public to tmp1
                Goldilocks::copy_avx(tmp1[args[i_args]], publics[args[i_args + 1]]);
                i_args += 2;
                break;
            }
            case 24: {
                // OPERATION WITH DEST: tmp1 - SRC0: public - SRC1: public
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], publics[args[i_args + 2]], publics[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 25: {
                // OPERATION WITH DEST: tmp1 - SRC0: public - SRC1: number
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], publics[args[i_args + 2]], numbers_[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 26: {
                // COPY number to tmp1
                Goldilocks::copy_avx(tmp1[args[i_args]], numbers_[args[i_args + 1]]);
                i_args += 2;
                break;
            }
            case 27: {
                // OPERATION WITH DEST: tmp1 - SRC0: number - SRC1: number
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], numbers_[args[i_args + 2]], numbers_[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 28: {
                // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: commit1
                Goldilocks3::op_31_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], bufferT_[nColsStagesAcc[args[i_args + 5]] + args[i_args + 6]]);
                i_args += 7;
                break;
            }
            case 29: {
                // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: tmp1
                Goldilocks3::op_31_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], tmp1[args[i_args + 5]]);
                i_args += 6;
                break;
            }
            case 30: {
                // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: public
                Goldilocks3::op_31_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], publics[args[i_args + 5]]);
                i_args += 6;
                break;
            }
            case 31: {
                // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: number
                Goldilocks3::op_31_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], numbers_[args[i_args + 5]]);
                i_args += 6;
                break;
            }
            case 32: {
                // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: commit1
                Goldilocks3::op_31_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], tmp3[args[i_args + 3]], bufferT_[nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]]);
                i_args += 6;
                break;
            }
            case 33: {
                // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: tmp1
                Goldilocks3::op_31_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], tmp3[args[i_args + 3]], tmp1[args[i_args + 4]]);
                i_args += 5;
                break;
            }
            case 34: {
                // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: public
                Goldilocks3::op_31_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], tmp3[args[i_args + 3]], publics[args[i_args + 4]]);
                i_args += 5;
                break;
            }
            case 35: {
                // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: number
                Goldilocks3::op_31_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], tmp3[args[i_args + 3]], numbers_[args[i_args + 4]]);
                i_args += 5;
                break;
            }
            case 36: {
                // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: commit1
                Goldilocks3::op_31_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], challenges[args[i_args + 3]], bufferT_[nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]]);
                i_args += 6;
                break;
            }
            case 37: {
                // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: tmp1
                Goldilocks3::op_31_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], challenges[args[i_args + 3]], tmp1[args[i_args + 4]]);
                i_args += 5;
                break;
            }
            case 38: {
                // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: public
                Goldilocks3::op_31_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], challenges[args[i_args + 3]], publics[args[i_args + 4]]);
                i_args += 5;
                break;
            }
            case 39: {
                // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: number
                Goldilocks3::op_31_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], challenges[args[i_args + 3]], numbers_[args[i_args + 4]]);
                i_args += 5;
                break;
            }
            case 40: {
                // OPERATION WITH DEST: commit3 - SRC0: subproofValue - SRC1: commit1
                Goldilocks3::op_31_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], subproofValues[args[i_args + 3]], bufferT_[nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]]);
                i_args += 6;
                break;
            }
            case 41: {
                // OPERATION WITH DEST: commit3 - SRC0: subproofValue - SRC1: tmp1
                Goldilocks3::op_31_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], subproofValues[args[i_args + 3]], tmp1[args[i_args + 4]]);
                i_args += 5;
                break;
            }
            case 42: {
                // OPERATION WITH DEST: commit3 - SRC0: subproofValue - SRC1: public
                Goldilocks3::op_31_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], subproofValues[args[i_args + 3]], publics[args[i_args + 4]]);
                i_args += 5;
                break;
            }
            case 43: {
                // OPERATION WITH DEST: commit3 - SRC0: subproofValue - SRC1: number
                Goldilocks3::op_31_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], subproofValues[args[i_args + 3]], numbers_[args[i_args + 4]]);
                i_args += 5;
                break;
            }
            case 44: {
                // COPY commit3 to commit3
                Goldilocks3::copy_avx((Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args]] + args[i_args + 1]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 45: {
                // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: commit3
                Goldilocks3::op_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 5]] + args[i_args + 6]]);
                i_args += 7;
                break;
            }
            case 46: {
                // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: tmp3
                Goldilocks3::op_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], tmp3[args[i_args + 5]]);
                i_args += 6;
                break;
            }
            case 47: {
                // MULTIPLICATION WITH DEST: commit3 - SRC0: commit3 - SRC1: challenge
                Goldilocks3::mul_avx((Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], challenges[args[i_args + 5]], challenges_ops[args[i_args + 5]]);
                i_args += 6;
                break;
            }
            case 48: {
                // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: challenge
                Goldilocks3::op_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], challenges[args[i_args + 5]]);
                i_args += 6;
                break;
            }
            case 49: {
                // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: subproofValue
                Goldilocks3::op_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], subproofValues[args[i_args + 5]]);
                i_args += 6;
                break;
            }
            case 50: {
                // COPY tmp3 to commit3
                Goldilocks3::copy_avx((Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args]] + args[i_args + 1]], tmp3[args[i_args + 2]]);
                i_args += 3;
                break;
            }
            case 51: {
                // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: tmp3
                Goldilocks3::op_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], tmp3[args[i_args + 3]], tmp3[args[i_args + 4]]);
                i_args += 5;
                break;
            }
            case 52: {
                // MULTIPLICATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: challenge
                Goldilocks3::mul_avx((Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], tmp3[args[i_args + 3]], challenges[args[i_args + 4]], challenges_ops[args[i_args + 4]]);
                i_args += 5;
                break;
            }
            case 53: {
                // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: challenge
                Goldilocks3::op_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], tmp3[args[i_args + 3]], challenges[args[i_args + 4]]);
                i_args += 5;
                break;
            }
            case 54: {
                // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: subproofValue
                Goldilocks3::op_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], tmp3[args[i_args + 3]], subproofValues[args[i_args + 4]]);
                i_args += 5;
                break;
            }
            case 55: {
                // MULTIPLICATION WITH DEST: commit3 - SRC0: challenge - SRC1: challenge
                Goldilocks3::mul_avx((Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], challenges[args[i_args + 3]], challenges[args[i_args + 4]], challenges_ops[args[i_args + 4]]);
                i_args += 5;
                break;
            }
            case 56: {
                // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: challenge
                Goldilocks3::op_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], challenges[args[i_args + 3]], challenges[args[i_args + 4]]);
                i_args += 5;
                break;
            }
            case 57: {
                // MULTIPLICATION WITH DEST: commit3 - SRC0: subproofValue - SRC1: challenge
                Goldilocks3::mul_avx((Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], subproofValues[args[i_args + 3]], challenges[args[i_args + 4]], challenges_ops[args[i_args + 4]]);
                i_args += 5;
                break;
            }
            case 58: {
                // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: subproofValue
                Goldilocks3::op_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], challenges[args[i_args + 3]], subproofValues[args[i_args + 4]]);
                i_args += 5;
                break;
            }
            case 59: {
                // OPERATION WITH DEST: commit3 - SRC0: subproofValue - SRC1: subproofValue
                Goldilocks3::op_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], subproofValues[args[i_args + 3]], subproofValues[args[i_args + 4]]);
                i_args += 5;
                break;
            }
            case 60: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: commit1
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], bufferT_[nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]]);
                i_args += 6;
                break;
            }
            case 61: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: tmp1
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], tmp1[args[i_args + 4]]);
                i_args += 5;
                break;
            }
            case 62: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: public
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], publics[args[i_args + 4]]);
                i_args += 5;
                break;
            }
            case 63: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: number
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], numbers_[args[i_args + 4]]);
                i_args += 5;
                break;
            }
            case 64: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: commit1
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]]);
                i_args += 5;
                break;
            }
            case 65: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: tmp1
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], tmp1[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 66: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: public
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], publics[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 67: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: number
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], numbers_[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 68: {
                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: commit1
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]]);
                i_args += 5;
                break;
            }
            case 69: {
                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: tmp1
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], tmp1[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 70: {
                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: public
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], publics[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 71: {
                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: number
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], numbers_[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 72: {
                // OPERATION WITH DEST: tmp3 - SRC0: subproofValue - SRC1: commit1
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], subproofValues[args[i_args + 2]], bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]]);
                i_args += 5;
                break;
            }
            case 73: {
                // OPERATION WITH DEST: tmp3 - SRC0: subproofValue - SRC1: tmp1
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], subproofValues[args[i_args + 2]], tmp1[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 74: {
                // OPERATION WITH DEST: tmp3 - SRC0: subproofValue - SRC1: public
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], subproofValues[args[i_args + 2]], publics[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 75: {
                // OPERATION WITH DEST: tmp3 - SRC0: subproofValue - SRC1: number
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], subproofValues[args[i_args + 2]], numbers_[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 76: {
                // COPY commit3 to tmp3
                Goldilocks3::copy_avx(tmp3[args[i_args]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]]);
                i_args += 3;
                break;
            }
            case 77: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: commit3
                Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]]);
                i_args += 6;
                break;
            }
            case 78: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: tmp3
                Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], tmp3[args[i_args + 4]]);
                i_args += 5;
                break;
            }
            case 79: {
                // MULTIPLICATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: challenge
                Goldilocks3::mul_avx(tmp3[args[i_args + 1]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], challenges[args[i_args + 4]], challenges_ops[args[i_args + 4]]);
                i_args += 5;
                break;
            }
            case 80: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: challenge
                Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], challenges[args[i_args + 4]]);
                i_args += 5;
                break;
            }
            case 81: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: subproofValue
                Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], subproofValues[args[i_args + 4]]);
                i_args += 5;
                break;
            }
            case 82: {
                // COPY tmp3 to tmp3
                Goldilocks3::copy_avx(tmp3[args[i_args]], tmp3[args[i_args + 1]]);
                i_args += 2;
                break;
            }
            case 83: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: tmp3
                Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], tmp3[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 84: {
                // MULTIPLICATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: challenge
                Goldilocks3::mul_avx(tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 85: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: challenge
                Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], challenges[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 86: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: subproofValue
                Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], subproofValues[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 87: {
                // MULTIPLICATION WITH DEST: tmp3 - SRC0: challenge - SRC1: challenge
                Goldilocks3::mul_avx(tmp3[args[i_args + 1]], challenges[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 88: {
                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: challenge
                Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], challenges[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 89: {
                // MULTIPLICATION WITH DEST: tmp3 - SRC0: subproofValue - SRC1: challenge
                Goldilocks3::mul_avx(tmp3[args[i_args + 1]], subproofValues[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 90: {
                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: subproofValue
                Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], subproofValues[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 91: {
                // OPERATION WITH DEST: tmp3 - SRC0: subproofValue - SRC1: subproofValue
                Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], subproofValues[args[i_args + 2]], subproofValues[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 92: {
                // COPY eval to tmp3
                Goldilocks3::copy_avx(tmp3[args[i_args]], evals[args[i_args + 1]]);
                i_args += 2;
                break;
            }
            case 93: {
                // MULTIPLICATION WITH DEST: tmp3 - SRC0: eval - SRC1: challenge
                Goldilocks3::mul_avx(tmp3[args[i_args + 1]], evals[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 94: {
                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: eval
                Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], evals[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 95: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: eval
                Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], evals[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 96: {
                // OPERATION WITH DEST: tmp3 - SRC0: eval - SRC1: commit1
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], evals[args[i_args + 2]], bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]]);
                i_args += 5;
                break;
            }
            case 97: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: eval
                Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], evals[args[i_args + 4]]);
                i_args += 5;
                break;
            }
                    default: {
                        std::cout << " Wrong operation!" << std::endl;
                        exit(1);
                    }
                }
            }

            if(imPols) {
                storeImPolynomials(params, bufferT_, i);
            } else {
                storePolynomial(dest, parserParams, i, tmp1, tmp3, inverse);
            }

            if (i_args != parserParams.nArgs) std::cout << " " << i_args << " - " << parserParams.nArgs << std::endl;
            assert(i_args == parserParams.nArgs);
        }

    }
};

#endif