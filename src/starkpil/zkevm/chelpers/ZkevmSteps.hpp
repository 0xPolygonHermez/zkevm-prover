#include "chelpers_steps.hpp"


class ZkevmSteps : public CHelpersSteps {
public:
    void storePolinomial(Goldilocks::Element *pols, __m256i *bufferT, uint64_t* mapBufferT, uint64_t* nColsSteps, uint64_t *offsetsSteps, uint64_t *buffTOffsetsSteps, uint64_t *nextStrides, uint64_t nOpenings, uint64_t domainSize, bool domainExtended, uint64_t nStages, bool needModule, uint64_t row, uint64_t stage, uint64_t stagePos, uint64_t openingPointIndex, uint64_t dim) {
        bool isTmpPol = !domainExtended && nStages + 1 == stage;
        if(needModule) {
            uint64_t offsetsDest[4];
            uint64_t nextStrideOffset = row + nextStrides[openingPointIndex];
            if(isTmpPol) {
                uint64_t stepOffset = offsetsSteps[stage] + stagePos * domainSize;
                offsetsDest[0] = stepOffset + (nextStrideOffset % domainSize) * dim;
                offsetsDest[1] = stepOffset + ((nextStrideOffset + 1) % domainSize) * dim;
                offsetsDest[2] = stepOffset + ((nextStrideOffset + 2) % domainSize) * dim;
                offsetsDest[3] = stepOffset + ((nextStrideOffset + 3) % domainSize) * dim;
            } else {
                uint64_t stepOffset = offsetsSteps[stage] + stagePos;
                offsetsDest[0] = stepOffset + (nextStrideOffset % domainSize) * nColsSteps[stage];
                offsetsDest[1] = stepOffset + ((nextStrideOffset + 1) % domainSize) * nColsSteps[stage];
                offsetsDest[2] = stepOffset + ((nextStrideOffset + 2) % domainSize) * nColsSteps[stage];
                offsetsDest[3] = stepOffset + ((nextStrideOffset + 3) % domainSize) * nColsSteps[stage];
            }
            if(dim == 1) {
                Goldilocks::store_avx(&pols[0], offsetsDest, bufferT[mapBufferT[buffTOffsetsSteps[stage] + nOpenings * stagePos + openingPointIndex]]);
            } else {
                Goldilocks3::store_avx(&pols[0], offsetsDest, &bufferT[mapBufferT[buffTOffsetsSteps[stage] + nOpenings * stagePos + openingPointIndex]], nOpenings);
            }
        } else {
            if(dim == 1) {
                if(isTmpPol) {
                    Goldilocks::store_avx(&pols[offsetsSteps[stage] + stagePos * domainSize + (row + nextStrides[openingPointIndex])], uint64_t(1), bufferT[mapBufferT[buffTOffsetsSteps[stage] + nOpenings * stagePos + openingPointIndex]]);
                } else {
                    Goldilocks::store_avx(&pols[offsetsSteps[stage] + stagePos + (row + nextStrides[openingPointIndex]) * nColsSteps[stage]], nColsSteps[stage], bufferT[mapBufferT[buffTOffsetsSteps[stage] + nOpenings * stagePos + openingPointIndex]]);
                }
            } else {
                if(isTmpPol) {
                    Goldilocks3::store_avx(&pols[offsetsSteps[stage] + stagePos * domainSize + (row + nextStrides[openingPointIndex]) * FIELD_EXTENSION], uint64_t(FIELD_EXTENSION), &bufferT[mapBufferT[buffTOffsetsSteps[stage] + nOpenings * stagePos + openingPointIndex]], nOpenings);
                } else {
                    Goldilocks3::store_avx(&pols[offsetsSteps[stage] + stagePos + (row + nextStrides[openingPointIndex]) * nColsSteps[stage]], nColsSteps[stage], &bufferT[mapBufferT[buffTOffsetsSteps[stage] + nOpenings * stagePos + openingPointIndex]], nOpenings);
                }
            }
        }
    }

    void calculateExpressions(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams, uint32_t nrowsBatch, bool domainExtended) {
        uint64_t domainSize = domainExtended ? 1 << starkInfo.starkStruct.nBitsExt : 1 << starkInfo.starkStruct.nBits;
        uint64_t extendBits = (starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits);
        int64_t extend = domainExtended ? (1 << extendBits) : 1;
        Goldilocks::Element *x = domainExtended ? params.x_2ns : params.x_n;
        ConstantPolsStarks *constPols = domainExtended ? params.pConstPols2ns : params.pConstPols;
        Goldilocks3::Element_avx challenges[starkInfo.challengesMap.size()];
        Goldilocks3::Element_avx challenges_ops[starkInfo.challengesMap.size()];

        uint8_t *ops = &parserArgs.ops[parserParams.opsOffset];

        uint16_t *args = &parserArgs.args[parserParams.argsOffset]; 

        uint64_t* numbers = &parserArgs.numbers[parserParams.numbersOffset];

        uint16_t* cmPolsUsed = &parserArgs.cmPolsIds[parserParams.cmPolsOffset];

        uint16_t* cmPolsCalculated = &parserArgs.cmPolsCalculatedIds[parserParams.cmPolsCalculatedOffset];

        uint16_t* constPolsUsed = &parserArgs.constPolsIds[parserParams.constPolsOffset];

        __m256i numbers_[parserParams.nNumbers];

        uint64_t nStages = starkInfo.nStages;
        uint64_t nOpenings = starkInfo.openingPoints.size();
        uint64_t nextStrides[nOpenings];
        for(uint64_t i = 0; i < nOpenings; ++i) {
            uint64_t opening = starkInfo.openingPoints[i] < 0 ? starkInfo.openingPoints[i] + domainSize : starkInfo.openingPoints[i];
            nextStrides[i] = opening * extend;
        }
        std::vector<bool> validConstraint(domainSize, true);
        uint64_t nCols = starkInfo.nConstants;
        uint64_t buffTOffsetsSteps_[nStages + 2];
        uint64_t nColsSteps[nStages + 2];
        uint64_t nColsStepsAccumulated[nStages + 2];
        uint64_t offsetsSteps[nStages + 2];

        offsetsSteps[0] = 0;
        nColsSteps[0] = starkInfo.nConstants;
        nColsStepsAccumulated[0] = 0;
        buffTOffsetsSteps_[0] = 0;
        for(uint64_t stage = 1; stage <= nStages; ++stage) {
            std::string section = "cm" + to_string(stage);
            offsetsSteps[stage] = starkInfo.mapOffsets[std::make_pair(section, domainExtended)];
            nColsSteps[stage] = starkInfo.mapSectionsN[section];
            nColsStepsAccumulated[stage] = nColsStepsAccumulated[stage - 1] + nColsSteps[stage - 1];
            buffTOffsetsSteps_[stage] = buffTOffsetsSteps_[stage - 1] + nOpenings*nColsSteps[stage - 1];
            nCols += nColsSteps[stage];
        }
        if(parserParams.stage <= nStages) {
            offsetsSteps[nStages + 1] = starkInfo.mapOffsets[std::make_pair("tmpExp", false)];
            nColsSteps[nStages + 1] = starkInfo.mapSectionsN["tmpExp"];
        } else {
            std::string section = "cm" + to_string(nStages + 1);
            offsetsSteps[nStages + 1] = starkInfo.mapOffsets[std::make_pair(section, true)];
            nColsSteps[nStages + 1] = starkInfo.mapSectionsN[section];
        }
        nColsStepsAccumulated[nStages + 1] = nColsStepsAccumulated[nStages] + nColsSteps[nStages];
        buffTOffsetsSteps_[nStages + 1] = buffTOffsetsSteps_[nStages] + nOpenings*nColsSteps[nStages];
        nCols += nColsSteps[nStages + 1];

    #pragma omp parallel for
        for(uint64_t i = 0; i < starkInfo.challengesMap.size(); ++i) {
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

    #pragma omp parallel for
        for(uint64_t i = 0; i < parserParams.nNumbers; ++i) {
            numbers_[i] = _mm256_set1_epi64x(numbers[i]);
        }

        __m256i publics[starkInfo.nPublics];
    #pragma omp parallel for
        for(uint64_t i = 0; i < starkInfo.nPublics; ++i) {
            publics[i] = _mm256_set1_epi64x(params.publicInputs[i].fe);
        }

        Goldilocks3::Element_avx subproofValues[starkInfo.nSubProofValues];
    #pragma omp parallel for
        for(uint64_t i = 0; i < starkInfo.nSubProofValues; ++i) {
            subproofValues[i][0] = _mm256_set1_epi64x(params.subproofValues[i * FIELD_EXTENSION].fe);
            subproofValues[i][1] = _mm256_set1_epi64x(params.subproofValues[i * FIELD_EXTENSION + 1].fe);
            subproofValues[i][2] = _mm256_set1_epi64x(params.subproofValues[i * FIELD_EXTENSION + 2].fe);
        }

        Goldilocks3::Element_avx evals[starkInfo.evMap.size()];
    #pragma omp parallel for
        for(uint64_t i = 0; i < starkInfo.evMap.size(); ++i) {
            evals[i][0] = _mm256_set1_epi64x(params.evals[i * FIELD_EXTENSION].fe);
            evals[i][1] = _mm256_set1_epi64x(params.evals[i * FIELD_EXTENSION + 1].fe);
            evals[i][2] = _mm256_set1_epi64x(params.evals[i * FIELD_EXTENSION + 2].fe);
        }

        uint64_t nPols = parserParams.nConstPolsUsed * nOpenings;

        for(uint64_t k = 0; k < parserParams.nCmPolsUsed; ++k) {
            uint64_t polId = cmPolsUsed[k];
            PolMap polInfo = starkInfo.cmPolsMap[polId];
            nPols += polInfo.dim * nOpenings;
        }

        for(uint64_t k = 0; k < parserParams.nCmPolsCalculated; ++k) {
            uint64_t polId = cmPolsCalculated[k];
            if(std::find(cmPolsUsed, cmPolsUsed + parserParams.nCmPolsUsed, polId) != cmPolsUsed + parserParams.nCmPolsUsed) continue;
            PolMap polInfo = starkInfo.cmPolsMap[polId];
            nPols += polInfo.dim * nOpenings;
        }

        uint64_t mapBufferT_[nPols];
        uint64_t kk = 0;

        for(uint64_t k = 0; k < parserParams.nConstPolsUsed; ++k) {
            uint64_t id = constPolsUsed[k];
            for(uint64_t o = 0; o < nOpenings; ++o) {
                mapBufferT_[nOpenings * id + o] = kk++;
            }
        }

        for(uint64_t k = 0; k < parserParams.nCmPolsUsed; ++k) {
            uint64_t polId = cmPolsUsed[k];
            PolMap polInfo = starkInfo.cmPolsMap[polId];
            uint64_t stage = polInfo.stage == string("tmpExp") ? nStages + 1 : polInfo.stageNum;
            uint64_t stagePos = polInfo.stagePos;
            for(uint64_t d = 0; d < polInfo.dim; ++d) {
                for(uint64_t o = 0; o < nOpenings; ++o) {
                    mapBufferT_[nOpenings * nColsStepsAccumulated[stage] + nOpenings * (stagePos + d) + o] = kk++;
                }
            }
        }

        for(uint64_t k = 0; k < parserParams.nCmPolsCalculated; ++k) {
            uint64_t polId = cmPolsCalculated[k];
            if(std::find(cmPolsUsed, cmPolsUsed + parserParams.nCmPolsUsed, polId) != cmPolsUsed + parserParams.nCmPolsUsed) continue;
            PolMap polInfo = starkInfo.cmPolsMap[polId];
            uint64_t stage = polInfo.stage == string("tmpExp") ? nStages + 1 : polInfo.stageNum;
            uint64_t stagePos = polInfo.stagePos;
            for(uint64_t d = 0; d < polInfo.dim; ++d) {
                for(uint64_t o = 0; o < nOpenings; ++o) {
                    mapBufferT_[nOpenings * nColsStepsAccumulated[stage] + nOpenings * (stagePos + d) + o] = kk++;
                }
            }
        }

    #pragma omp parallel for
        for (uint64_t i = 0; i < domainSize; i+= nrowsBatch) {
            bool const needModule = ((static_cast<int64_t>(i + nrowsBatch) + extend*starkInfo.openingPoints[nOpenings - 1]) >= static_cast<int64_t>(domainSize)) || ((static_cast<int64_t>(i) + extend*starkInfo.openingPoints[0]) < 0);
            uint64_t i_args = 0;

            __m256i tmp1[parserParams.nTemp1];
            Goldilocks3::Element_avx tmp3[parserParams.nTemp3];
            Goldilocks3::Element_avx tmp3_;
            // Goldilocks3::Element_avx tmp3_0;
            Goldilocks3::Element_avx tmp3_1;
            __m256i tmp1_1;
            __m256i bufferT_[nPols];

            Goldilocks::Element bufferT[nOpenings*nrowsBatch];

            for(uint64_t k = 0; k < parserParams.nConstPolsUsed; ++k) {
                uint64_t id = constPolsUsed[k];
                for(uint64_t o = 0; o < nOpenings; ++o) {
                    for(uint64_t j = 0; j < nrowsBatch; ++j) {
                        uint64_t l = (i + j + nextStrides[o]) % domainSize;
                        bufferT[nrowsBatch*o + j] = ((Goldilocks::Element *)constPols->address())[l * nColsSteps[0] + id];
                    }
                    Goldilocks::load_avx(bufferT_[mapBufferT_[nOpenings * id + o]], &bufferT[nrowsBatch*o]);
                }
            }

            for(uint64_t k = 0; k < parserParams.nCmPolsUsed; ++k) {
                uint64_t polId = cmPolsUsed[k];
                PolMap polInfo = starkInfo.cmPolsMap[polId];
                bool isTmpPol = polInfo.stage == string("tmpExp") && !domainExtended;
                uint64_t stage = isTmpPol ? nStages + 1 : polInfo.stageNum;
                uint64_t stagePos = polInfo.stagePos;
                uint64_t dim = polInfo.dim;
                for(uint64_t d = 0; d < polInfo.dim; ++d) {
                    for(uint64_t o = 0; o < nOpenings; ++o) {
                        for(uint64_t j = 0; j < nrowsBatch; ++j) {
                            uint64_t l = (i + j + nextStrides[o]) % domainSize;
                            if(isTmpPol) {
                                bufferT[nrowsBatch*o + j] = params.pols[offsetsSteps[stage] + stagePos * domainSize + l*dim + d];
                            } else {
                                bufferT[nrowsBatch*o + j] = params.pols[offsetsSteps[stage] + l * nColsSteps[stage] + stagePos + d];
                            }
                        }
                        Goldilocks::load_avx(bufferT_[mapBufferT_[nOpenings * nColsStepsAccumulated[stage] + nOpenings * (stagePos + d) + o]], &bufferT[nrowsBatch*o]);
                    }
                }
            }
    

            for (uint64_t kk = 0; kk < parserParams.nOps; ++kk) {
                switch (ops[kk]) {
            case 0: {
                // COPY commit1 to commit1
                Goldilocks::copy_avx(bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args]] + nOpenings * args[i_args + 1] + args[i_args + 2]]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 3]] + nOpenings * args[i_args + 4] + args[i_args + 5]]]);
                storePolinomial(params.pols, bufferT_, mapBufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, nOpenings, domainSize, domainExtended, nStages, needModule, i, args[i_args], args[i_args + 1], args[i_args + 2], 1);
                i_args += 6;
                break;
            }
            case 1: {
                // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: commit1
                Goldilocks::op_avx(args[i_args], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 1]] + nOpenings * args[i_args + 2] + args[i_args + 3]]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 4]] + nOpenings * args[i_args + 5] + args[i_args + 6]]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 7]] + nOpenings * args[i_args + 8] + args[i_args + 9]]]);
                storePolinomial(params.pols, bufferT_, mapBufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, nOpenings, domainSize, domainExtended, nStages, needModule, i, args[i_args + 1], args[i_args + 2], args[i_args + 3], 1);
                i_args += 10;
                break;
            }
            case 2: {
                // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: tmp1
                Goldilocks::op_avx(args[i_args], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 1]] + nOpenings * args[i_args + 2] + args[i_args + 3]]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 4]] + nOpenings * args[i_args + 5] + args[i_args + 6]]], tmp1[args[i_args + 7]]);
                storePolinomial(params.pols, bufferT_, mapBufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, nOpenings, domainSize, domainExtended, nStages, needModule, i, args[i_args + 1], args[i_args + 2], args[i_args + 3], 1);
                i_args += 8;
                break;
            }
            case 3: {
                // COPY tmp1 to commit1
                Goldilocks::copy_avx(bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args]] + nOpenings * args[i_args + 1] + args[i_args + 2]]], tmp1[args[i_args + 3]]);
                storePolinomial(params.pols, bufferT_, mapBufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, nOpenings, domainSize, domainExtended, nStages, needModule, i, args[i_args], args[i_args + 1], args[i_args + 2], 1);
                i_args += 4;
                break;
            }
            case 4: {
                // OPERATION WITH DEST: commit1 - SRC0: tmp1 - SRC1: tmp1
                Goldilocks::op_avx(args[i_args], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 1]] + nOpenings * args[i_args + 2] + args[i_args + 3]]], tmp1[args[i_args + 4]], tmp1[args[i_args + 5]]);
                storePolinomial(params.pols, bufferT_, mapBufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, nOpenings, domainSize, domainExtended, nStages, needModule, i, args[i_args + 1], args[i_args + 2], args[i_args + 3], 1);
                i_args += 6;
                break;
            }
            case 5: {
                // OPERATION WITH DEST: commit1 - SRC0: tmp1 - SRC1: number
                Goldilocks::op_avx(args[i_args], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 1]] + nOpenings * args[i_args + 2] + args[i_args + 3]]], tmp1[args[i_args + 4]], numbers_[args[i_args + 5]]);
                storePolinomial(params.pols, bufferT_, mapBufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, nOpenings, domainSize, domainExtended, nStages, needModule, i, args[i_args + 1], args[i_args + 2], args[i_args + 3], 1);
                i_args += 6;
                break;
            }
            case 6: {
                // COPY commit1 to tmp1
                Goldilocks::copy_avx(tmp1[args[i_args]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 1]] + nOpenings * args[i_args + 2] + args[i_args + 3]]]);
                i_args += 4;
                break;
            }
            case 7: {
                // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: commit1
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 5]] + nOpenings * args[i_args + 6] + args[i_args + 7]]]);
                i_args += 8;
                break;
            }
            case 8: {
                // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: tmp1
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], tmp1[args[i_args + 5]]);
                i_args += 6;
                break;
            }
            case 9: {
                // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: public
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], publics[args[i_args + 5]]);
                i_args += 6;
                break;
            }
            case 10: {
                // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: number
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], numbers_[args[i_args + 5]]);
                i_args += 6;
                break;
            }
            case 11: {
                // COPY tmp1 to tmp1
                Goldilocks::copy_avx(tmp1[args[i_args]], tmp1[args[i_args + 1]]);
                i_args += 2;
                break;
            }
            case 12: {
                // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: tmp1
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], tmp1[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 13: {
                // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: number
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], numbers_[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 14: {
                // COPY number to tmp1
                Goldilocks::copy_avx(tmp1[args[i_args]], numbers_[args[i_args + 1]]);
                i_args += 2;
                break;
            }
            case 15: {
                // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: commit1
                Goldilocks3::op_31_avx(args[i_args], &bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 1]] + nOpenings * args[i_args + 2] + args[i_args + 3]]], nOpenings, 
                        &(tmp3[args[i_args + 4]][0]), 1, 
                        bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 5]] + nOpenings * args[i_args + 6] + args[i_args + 7]]]);
                storePolinomial(params.pols, bufferT_, mapBufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, nOpenings, domainSize, domainExtended, nStages, needModule, i, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                i_args += 8;
                break;
            }
            case 16: {
                // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: tmp1
                Goldilocks3::op_31_avx(args[i_args], &bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 1]] + nOpenings * args[i_args + 2] + args[i_args + 3]]], nOpenings, 
                        &(tmp3[args[i_args + 4]][0]), 1, 
                        tmp1[args[i_args + 5]]);
                storePolinomial(params.pols, bufferT_, mapBufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, nOpenings, domainSize, domainExtended, nStages, needModule, i, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                i_args += 6;
                break;
            }
            case 17: {
                // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: tmp3
                Goldilocks3::op_avx(args[i_args], &bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 1]] + nOpenings * args[i_args + 2] + args[i_args + 3]]], nOpenings, 
                        &bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 4]] + nOpenings * args[i_args + 5] + args[i_args + 6]]], nOpenings, 
                        &(tmp3[args[i_args + 7]][0]), 1);
                storePolinomial(params.pols, bufferT_, mapBufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, nOpenings, domainSize, domainExtended, nStages, needModule, i, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                i_args += 8;
                break;
            }
            case 18: {
                // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: tmp3
                Goldilocks3::op_avx(args[i_args], &bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 1]] + nOpenings * args[i_args + 2] + args[i_args + 3]]], nOpenings, 
                        &(tmp3[args[i_args + 4]][0]), 1, 
                        &(tmp3[args[i_args + 5]][0]), 1);
                storePolinomial(params.pols, bufferT_, mapBufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, nOpenings, domainSize, domainExtended, nStages, needModule, i, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                i_args += 6;
                break;
            }
            case 19: {
                // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: challenge
                Goldilocks3::op_avx(args[i_args], &bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 1]] + nOpenings * args[i_args + 2] + args[i_args + 3]]], nOpenings, 
                        &(tmp3[args[i_args + 4]][0]), 1, 
                        &(challenges[args[i_args + 5]][0]), 1);
                storePolinomial(params.pols, bufferT_, mapBufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, nOpenings, domainSize, domainExtended, nStages, needModule, i, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                i_args += 6;
                break;
            }
            case 20: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: number
                Goldilocks3::op_31_avx(args[i_args], &(tmp3[args[i_args + 1]][0]), 1, 
                        &bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], nOpenings, 
                        numbers_[args[i_args + 5]]);
                i_args += 6;
                break;
            }
            case 21: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: commit1
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 3]] + nOpenings * args[i_args + 4] + args[i_args + 5]]]);
                i_args += 6;
                break;
            }
            case 22: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: tmp1
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], tmp1[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 23: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: x
                Goldilocks::load_avx(tmp1_1, &x[i], uint64_t(1));
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], tmp1_1);
                i_args += 3;
                break;
            }
            case 24: {
                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: commit1
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 3]] + nOpenings * args[i_args + 4] + args[i_args + 5]]]);
                i_args += 6;
                break;
            }
            case 25: {
                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: tmp1
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], tmp1[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 26: {
                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: x
                Goldilocks::load_avx(tmp1_1, &x[i], uint64_t(1));
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], tmp1_1);
                i_args += 3;
                break;
            }
            case 27: {
                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: number
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], numbers_[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 28: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: commit3
                Goldilocks3::op_avx(args[i_args], &(tmp3[args[i_args + 1]][0]), 1, 
                        &bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], nOpenings, 
                        &bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 5]] + nOpenings * args[i_args + 6] + args[i_args + 7]]], nOpenings);
                i_args += 8;
                break;
            }
            case 29: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: tmp3
                Goldilocks3::op_avx(args[i_args], &(tmp3[args[i_args + 1]][0]), 1, 
                        &bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], nOpenings, 
                        &(tmp3[args[i_args + 5]][0]), 1);
                i_args += 6;
                break;
            }
            case 30: {
                // MULTIPLICATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: challenge
                Goldilocks3::mul_avx(&(tmp3[args[i_args + 1]][0]), 1, 
                        &bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], nOpenings, 
                        challenges[args[i_args + 5]], challenges_ops[args[i_args + 5]]);
                i_args += 6;
                break;
            }
            case 31: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: challenge
                Goldilocks3::op_avx(args[i_args], &(tmp3[args[i_args + 1]][0]), 1, 
                        &bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], nOpenings, 
                        &(challenges[args[i_args + 5]][0]), 1);
                i_args += 6;
                break;
            }
            case 32: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: tmp3
                Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], tmp3[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 33: {
                // MULTIPLICATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: challenge
                Goldilocks3::mul_avx(tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 34: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: challenge
                Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], challenges[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 35: {
                // OPERATION WITH DEST: tmp3 - SRC0: eval - SRC1: commit1
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], evals[args[i_args + 2]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 3]] + nOpenings * args[i_args + 4] + args[i_args + 5]]]);
                i_args += 6;
                break;
            }
            case 36: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: eval
                Goldilocks3::op_avx(args[i_args], &(tmp3[args[i_args + 1]][0]), 1, 
                        &bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], nOpenings, 
                        &(evals[args[i_args + 5]][0]), 1);
                i_args += 6;
                break;
            }
            case 37: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: xDivXSubXi
                Goldilocks3::load_avx(tmp3_1, &params.xDivXSubXi[(i + args[i_args + 3]*domainSize)*FIELD_EXTENSION], uint64_t(FIELD_EXTENSION));
                Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], tmp3_1);
                i_args += 4;
                break;
            }
            case 38: {
                // OPERATION WITH DEST: q - SRC0: tmp3 - SRC1: Zi
                Goldilocks::Element tmp_inv[3];
                Goldilocks::Element ti0[4];
                Goldilocks::Element ti1[4];
                Goldilocks::Element ti2[4];
                Goldilocks::store_avx(ti0, tmp3[args[i_args]][0]);
                Goldilocks::store_avx(ti1, tmp3[args[i_args]][1]);
                Goldilocks::store_avx(ti2, tmp3[args[i_args]][2]);
                for (uint64_t j = 0; j < AVX_SIZE_; ++j) {
                    tmp_inv[0] = ti0[j];
                    tmp_inv[1] = ti1[j];
                    tmp_inv[2] = ti2[j];
                    Goldilocks3::mul((Goldilocks3::Element &)(params.q_2ns[(i + j) * FIELD_EXTENSION]), params.zi[i + j],(Goldilocks3::Element &)tmp_inv);
                }
                i_args += 1;
                break;
            }
            case 39: {
                // OPERATION WITH DEST: f - SRC0: tmp3 - SRC1: tmp3
                Goldilocks3::op_avx(args[i_args], tmp3_, tmp3[args[i_args + 1]], tmp3[args[i_args + 2]]);
                Goldilocks3::store_avx(&params.f_2ns[i*FIELD_EXTENSION], uint64_t(FIELD_EXTENSION), tmp3_);
                i_args += 3;
                break;
            }
            case 40: {
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 5]] + nOpenings * args[i_args + 6] + args[i_args + 7]]]);
                i_args += 8;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 5]] + nOpenings * args[i_args + 6] + args[i_args + 7]]]);
                i_args += 8;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], tmp1[args[i_args + 3]]);
                i_args += 4;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], numbers_[args[i_args + 5]]);
                i_args += 6;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], tmp1[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 41: {
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 5]] + nOpenings * args[i_args + 6] + args[i_args + 7]]]);
                i_args += 8;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 5]] + nOpenings * args[i_args + 6] + args[i_args + 7]]]);
                i_args += 8;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], tmp1[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 42: {
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 5]] + nOpenings * args[i_args + 6] + args[i_args + 7]]]);
                i_args += 8;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], tmp1[args[i_args + 3]]);
                i_args += 4;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], numbers_[args[i_args + 5]]);
                i_args += 6;
                break;
            }
            case 43: {
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], tmp1[args[i_args + 3]]);
                i_args += 4;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 5]] + nOpenings * args[i_args + 6] + args[i_args + 7]]]);
                i_args += 8;
                break;
            }
            case 44: {
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 3]] + nOpenings * args[i_args + 4] + args[i_args + 5]]]);
                i_args += 6;
                Goldilocks3::mul_avx(tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                i_args += 4;
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 3]] + nOpenings * args[i_args + 4] + args[i_args + 5]]]);
                i_args += 6;
                Goldilocks3::mul_avx(tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                i_args += 4;
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 3]] + nOpenings * args[i_args + 4] + args[i_args + 5]]]);
                i_args += 6;
                Goldilocks3::mul_avx(tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                i_args += 4;
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 3]] + nOpenings * args[i_args + 4] + args[i_args + 5]]]);
                i_args += 6;
                Goldilocks3::mul_avx(tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 45: {
                Goldilocks::copy_avx(tmp1[args[i_args]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 1]] + nOpenings * args[i_args + 2] + args[i_args + 3]]]);
                i_args += 4;
                Goldilocks::copy_avx(tmp1[args[i_args]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 1]] + nOpenings * args[i_args + 2] + args[i_args + 3]]]);
                i_args += 4;
                Goldilocks::copy_avx(tmp1[args[i_args]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 1]] + nOpenings * args[i_args + 2] + args[i_args + 3]]]);
                i_args += 4;
                Goldilocks::copy_avx(tmp1[args[i_args]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 1]] + nOpenings * args[i_args + 2] + args[i_args + 3]]]);
                i_args += 4;
                break;
            }
            case 46: {
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], numbers_[args[i_args + 5]]);
                i_args += 6;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], tmp1[args[i_args + 3]]);
                i_args += 4;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], tmp1[args[i_args + 3]]);
                i_args += 4;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 5]] + nOpenings * args[i_args + 6] + args[i_args + 7]]]);
                i_args += 8;
                break;
            }
            case 47: {
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], tmp1[args[i_args + 3]]);
                i_args += 4;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], numbers_[args[i_args + 5]]);
                i_args += 6;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], tmp1[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 48: {
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 5]] + nOpenings * args[i_args + 6] + args[i_args + 7]]]);
                i_args += 8;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], tmp1[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 49: {
                Goldilocks3::mul_avx(tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                i_args += 4;
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], tmp1[args[i_args + 3]]);
                i_args += 4;
                Goldilocks3::mul_avx(tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                i_args += 4;
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], tmp1[args[i_args + 3]]);
                i_args += 4;
                Goldilocks3::mul_avx(tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                i_args += 4;
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], tmp1[args[i_args + 3]]);
                i_args += 4;
                Goldilocks3::mul_avx(tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                i_args += 4;
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], tmp1[args[i_args + 3]]);
                i_args += 4;
                Goldilocks3::mul_avx(tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                i_args += 4;
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], tmp1[args[i_args + 3]]);
                i_args += 4;
                Goldilocks3::mul_avx(tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                i_args += 4;
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], tmp1[args[i_args + 3]]);
                i_args += 4;
                Goldilocks3::mul_avx(tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                i_args += 4;
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], tmp1[args[i_args + 3]]);
                i_args += 4;
                Goldilocks3::mul_avx(tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                i_args += 4;
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], tmp1[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 50: {
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], numbers_[args[i_args + 5]]);
                i_args += 6;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], tmp1[args[i_args + 5]]);
                i_args += 6;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], tmp1[args[i_args + 3]]);
                i_args += 4;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], numbers_[args[i_args + 5]]);
                i_args += 6;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], tmp1[args[i_args + 5]]);
                i_args += 6;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], numbers_[args[i_args + 5]]);
                i_args += 6;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], tmp1[args[i_args + 5]]);
                i_args += 6;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], tmp1[args[i_args + 3]]);
                i_args += 4;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], numbers_[args[i_args + 5]]);
                i_args += 6;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], tmp1[args[i_args + 5]]);
                i_args += 6;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], numbers_[args[i_args + 5]]);
                i_args += 6;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], tmp1[args[i_args + 5]]);
                i_args += 6;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], tmp1[args[i_args + 3]]);
                i_args += 4;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], numbers_[args[i_args + 5]]);
                i_args += 6;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], tmp1[args[i_args + 5]]);
                i_args += 6;
                break;
            }
            case 51: {
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], tmp1[args[i_args + 5]]);
                i_args += 6;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], tmp1[args[i_args + 3]]);
                i_args += 4;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], tmp1[args[i_args + 5]]);
                i_args += 6;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], tmp1[args[i_args + 3]]);
                i_args += 4;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], tmp1[args[i_args + 5]]);
                i_args += 6;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], tmp1[args[i_args + 3]]);
                i_args += 4;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], tmp1[args[i_args + 5]]);
                i_args += 6;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], tmp1[args[i_args + 3]]);
                i_args += 4;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], tmp1[args[i_args + 5]]);
                i_args += 6;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], tmp1[args[i_args + 3]]);
                i_args += 4;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], tmp1[args[i_args + 5]]);
                i_args += 6;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], tmp1[args[i_args + 3]]);
                i_args += 4;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], tmp1[args[i_args + 5]]);
                i_args += 6;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], tmp1[args[i_args + 3]]);
                i_args += 4;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], tmp1[args[i_args + 5]]);
                i_args += 6;
                break;
            }
            case 52: {
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], tmp1[args[i_args + 5]]);
                i_args += 6;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], tmp1[args[i_args + 5]]);
                i_args += 6;
                break;
            }
            case 53: {
                Goldilocks3::mul_avx(tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                i_args += 4;
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], evals[args[i_args + 2]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 3]] + nOpenings * args[i_args + 4] + args[i_args + 5]]]);
                i_args += 6;
                Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], tmp3[args[i_args + 3]]);
                i_args += 4;
                Goldilocks3::mul_avx(tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                i_args += 4;
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], evals[args[i_args + 2]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 3]] + nOpenings * args[i_args + 4] + args[i_args + 5]]]);
                i_args += 6;
                Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], tmp3[args[i_args + 3]]);
                i_args += 4;
                Goldilocks3::mul_avx(tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                i_args += 4;
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], evals[args[i_args + 2]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 3]] + nOpenings * args[i_args + 4] + args[i_args + 5]]]);
                i_args += 6;
                Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], tmp3[args[i_args + 3]]);
                i_args += 4;
                Goldilocks3::mul_avx(tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                i_args += 4;
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], evals[args[i_args + 2]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 3]] + nOpenings * args[i_args + 4] + args[i_args + 5]]]);
                i_args += 6;
                Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], tmp3[args[i_args + 3]]);
                i_args += 4;
                Goldilocks3::mul_avx(tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                i_args += 4;
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], evals[args[i_args + 2]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 3]] + nOpenings * args[i_args + 4] + args[i_args + 5]]]);
                i_args += 6;
                Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], tmp3[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 54: {
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 3]] + nOpenings * args[i_args + 4] + args[i_args + 5]]]);
                i_args += 6;
                Goldilocks3::mul_avx(tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                i_args += 4;
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 3]] + nOpenings * args[i_args + 4] + args[i_args + 5]]]);
                i_args += 6;
                Goldilocks3::mul_avx(tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                i_args += 4;
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 3]] + nOpenings * args[i_args + 4] + args[i_args + 5]]]);
                i_args += 6;
                Goldilocks3::mul_avx(tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                i_args += 4;
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 3]] + nOpenings * args[i_args + 4] + args[i_args + 5]]]);
                i_args += 6;
                Goldilocks3::mul_avx(tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                i_args += 4;
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 3]] + nOpenings * args[i_args + 4] + args[i_args + 5]]]);
                i_args += 6;
                Goldilocks3::mul_avx(tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                i_args += 4;
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 3]] + nOpenings * args[i_args + 4] + args[i_args + 5]]]);
                i_args += 6;
                Goldilocks3::mul_avx(tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                i_args += 4;
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 3]] + nOpenings * args[i_args + 4] + args[i_args + 5]]]);
                i_args += 6;
                Goldilocks3::mul_avx(tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                i_args += 4;
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 3]] + nOpenings * args[i_args + 4] + args[i_args + 5]]]);
                i_args += 6;
                Goldilocks3::mul_avx(tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                i_args += 4;
                break;
            }
            case 55: {
                Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], tmp3[args[i_args + 3]]);
                i_args += 4;
                Goldilocks3::mul_avx(tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                i_args += 4;
                Goldilocks3::op_avx(args[i_args], &(tmp3[args[i_args + 1]][0]), 1, 
                        &bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], nOpenings, 
                        &(evals[args[i_args + 5]][0]), 1);
                i_args += 6;
                break;
            }
            case 56: {
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], tmp1[args[i_args + 3]]);
                i_args += 4;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 5]] + nOpenings * args[i_args + 6] + args[i_args + 7]]]);
                i_args += 8;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 5]] + nOpenings * args[i_args + 6] + args[i_args + 7]]]);
                i_args += 8;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], tmp1[args[i_args + 3]]);
                i_args += 4;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 5]] + nOpenings * args[i_args + 6] + args[i_args + 7]]]);
                i_args += 8;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], tmp1[args[i_args + 3]]);
                i_args += 4;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 5]] + nOpenings * args[i_args + 6] + args[i_args + 7]]]);
                i_args += 8;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], tmp1[args[i_args + 3]]);
                i_args += 4;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 5]] + nOpenings * args[i_args + 6] + args[i_args + 7]]]);
                i_args += 8;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], tmp1[args[i_args + 3]]);
                i_args += 4;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[mapBufferT_[buffTOffsetsSteps_[args[i_args + 2]] + nOpenings * args[i_args + 3] + args[i_args + 4]]], numbers_[args[i_args + 5]]);
                i_args += 6;
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], tmp1[args[i_args + 3]]);
                i_args += 4;
                break;
            }
                    default: {
                        std::cout << " Wrong operation!" << std::endl;
                        exit(1);
                    }
                }
            }
            if (parserParams.destDim != 0) {
                if(parserParams.destDim == 1) {
                    Goldilocks::Element res[4];
                    Goldilocks::store_avx(res, tmp1[parserParams.destId]);
                    for(uint64_t j = 0; j < 4; ++j) {
                        if(!Goldilocks::isZero(res[j])) {
                            validConstraint[i + j] = false;
                        }
                    }
                } else if(parserParams.destDim == 3) {
                    Goldilocks::Element res[12];
                    Goldilocks::store_avx(&res[0], tmp3[parserParams.destId][0]);
                    Goldilocks::store_avx(&res[4], tmp3[parserParams.destId][1]);
                    Goldilocks::store_avx(&res[8], tmp3[parserParams.destId][2]);
                    for(uint64_t j = 0; j < 12; ++j) {
                        if(!Goldilocks::isZero(res[j])) {
                            validConstraint[i + j%4] = false;
                        }
                    }
                } 
            }
            if (i_args != parserParams.nArgs) std::cout << " " << i_args << " - " << parserParams.nArgs << std::endl;
            assert(i_args == parserParams.nArgs);
        }
        if(parserParams.destDim != 0) {
            bool isValidConstraint = true;
            uint64_t nInvalidRows = 0;
            uint64_t maxInvalidRowsDisplay = 100;
            for(uint64_t i = 0; i < domainSize; ++i) {
                if(!validConstraint[i]) {
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
    }
};