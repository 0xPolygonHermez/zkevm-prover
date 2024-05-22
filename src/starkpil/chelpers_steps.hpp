#ifndef CHELPERS_STEPS_HPP
#define CHELPERS_STEPS_HPP
#include "chelpers.hpp"

#include "steps.hpp"

class CHelpersSteps {
public:
    virtual void storePolinomial(Goldilocks::Element *pols, __m256i *bufferT, uint64_t* nColsSteps, uint64_t *offsetsSteps, uint64_t *buffTOffsetsSteps, uint64_t *nextStrides, uint64_t nOpenings, uint64_t domainSize, bool domainExtended, uint64_t nStages, bool needModule, uint64_t row, uint64_t stage, uint64_t stagePos, uint64_t openingPointIndex, uint64_t dim) {
        bool isTmpPol = !domainExtended && stage == 4;
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
                Goldilocks::store_avx(&pols[0], offsetsDest, bufferT[buffTOffsetsSteps[stage] + nOpenings * stagePos + openingPointIndex]);
            } else {
                Goldilocks3::store_avx(&pols[0], offsetsDest, &bufferT[buffTOffsetsSteps[stage] + nOpenings * stagePos + openingPointIndex], nOpenings);
            }
        } else {
            if(dim == 1) {
                if(isTmpPol) {
                    Goldilocks::store_avx(&pols[offsetsSteps[stage] + stagePos * domainSize + (row + nextStrides[openingPointIndex])], uint64_t(1), bufferT[buffTOffsetsSteps[stage] + nOpenings * stagePos + openingPointIndex]);
                } else {
                    Goldilocks::store_avx(&pols[offsetsSteps[stage] + stagePos + (row + nextStrides[openingPointIndex]) * nColsSteps[stage]], nColsSteps[stage], bufferT[buffTOffsetsSteps[stage] + nOpenings * stagePos + openingPointIndex]);
                }
            } else {
                if(isTmpPol) {
                    Goldilocks3::store_avx(&pols[offsetsSteps[stage] + stagePos * domainSize + (row + nextStrides[openingPointIndex]) * FIELD_EXTENSION], uint64_t(FIELD_EXTENSION), &bufferT[buffTOffsetsSteps[stage] + nOpenings * stagePos + openingPointIndex], nOpenings);
                } else {
                    Goldilocks3::store_avx(&pols[offsetsSteps[stage] + stagePos + (row + nextStrides[openingPointIndex]) * nColsSteps[stage]], nColsSteps[stage], &bufferT[buffTOffsetsSteps[stage] + nOpenings * stagePos + openingPointIndex], nOpenings);
                }
            }
        }
    }

    virtual void calculateExpressions(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams) {
        uint32_t nrowsBatch = 4;
        bool domainExtended = parserParams.stage > 3 ? true : false;
        uint64_t domainSize = domainExtended ? 1 << starkInfo.starkStruct.nBitsExt : 1 << starkInfo.starkStruct.nBits;
        Polinomial &x = domainExtended ? params.x_2ns : params.x_n;
        ConstantPolsStarks *constPols = domainExtended ? params.pConstPols2ns : params.pConstPols;
        uint8_t *ops = &parserArgs.ops[parserParams.opsOffset];

        uint16_t *args = &parserArgs.args[parserParams.argsOffset]; 

        uint64_t* numbers = &parserArgs.numbers[parserParams.numbersOffset];

        uint64_t nStages = 3;
        uint64_t nextStride = domainExtended ? 1 << (starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits) : 1;
        uint64_t nextStrides[2] = { 0, nextStride };
        uint64_t nCols = starkInfo.nConstants;
        uint64_t buffTOffsetsSteps_[nStages + 2];
        uint64_t nColsSteps[nStages + 2];
        uint64_t nColsStepsAccumulated[nStages + 2];
        uint64_t offsetsSteps[nStages + 2];

        nColsSteps[0] = starkInfo.nConstants;
        nColsStepsAccumulated[0] = 0;
        buffTOffsetsSteps_[0] = 0;
        offsetsSteps[1] = domainExtended ? starkInfo.mapOffsets.section[eSection::cm1_2ns] : starkInfo.mapOffsets.section[eSection::cm1_n];
        nColsSteps[1] = starkInfo.mapSectionsN.section[eSection::cm1_2ns];
        nColsStepsAccumulated[1] = nColsStepsAccumulated[0] + nColsSteps[0];
        buffTOffsetsSteps_[1] = 2*nColsSteps[0];
        nCols += nColsSteps[1];

        offsetsSteps[2] = domainExtended ? starkInfo.mapOffsets.section[eSection::cm2_2ns] : starkInfo.mapOffsets.section[eSection::cm2_n];
        nColsSteps[2] = starkInfo.mapSectionsN.section[eSection::cm2_2ns];
        nColsStepsAccumulated[2] = nColsStepsAccumulated[1] + nColsSteps[1];
        buffTOffsetsSteps_[2] = buffTOffsetsSteps_[1] + 2*nColsSteps[1];
        nCols += nColsSteps[2];

        offsetsSteps[3] = domainExtended ? starkInfo.mapOffsets.section[eSection::cm3_2ns] : starkInfo.mapOffsets.section[eSection::cm3_n];
        nColsSteps[3] = starkInfo.mapSectionsN.section[eSection::cm3_2ns];
        nColsStepsAccumulated[3] = nColsStepsAccumulated[2] + nColsSteps[2];
        buffTOffsetsSteps_[3] = buffTOffsetsSteps_[2] + 2*nColsSteps[2];
        nCols += nColsSteps[3];

        if(parserParams.stage <= nStages) {
            offsetsSteps[4] = starkInfo.mapOffsets.section[eSection::tmpExp_n];
            nColsSteps[4] = starkInfo.mapSectionsN.section[eSection::tmpExp_n];
            nColsStepsAccumulated[4] = nColsStepsAccumulated[3] + nColsSteps[3];
        } else {
            offsetsSteps[4] = starkInfo.mapOffsets.section[eSection::cm4_2ns];
            nColsSteps[4] = starkInfo.mapSectionsN.section[eSection::cm4_2ns];
            nColsStepsAccumulated[4] = nColsStepsAccumulated[3] + nColsSteps[3];
        }
        buffTOffsetsSteps_[4] = buffTOffsetsSteps_[3] + 2*nColsSteps[3];
        nCols += nColsSteps[4];

        Goldilocks3::Element_avx challenges[params.challenges.degree()];
        Goldilocks3::Element_avx challenges_ops[params.challenges.degree()];
        for(uint64_t i = 0; i < params.challenges.degree(); ++i) {
            challenges[i][0] = _mm256_set1_epi64x(params.challenges[i][0].fe);
            challenges[i][1] = _mm256_set1_epi64x(params.challenges[i][1].fe);
            challenges[i][2] = _mm256_set1_epi64x(params.challenges[i][2].fe);

            Goldilocks::Element challenges_aux[3];
            challenges_aux[0] = params.challenges[i][0] + params.challenges[i][1];
            challenges_aux[1] = params.challenges[i][0] + params.challenges[i][2];
            challenges_aux[2] = params.challenges[i][1] + params.challenges[i][2];
            challenges_ops[i][0] = _mm256_set1_epi64x(challenges_aux[0].fe);
            challenges_ops[i][1] =  _mm256_set1_epi64x(challenges_aux[1].fe);
            challenges_ops[i][2] =  _mm256_set1_epi64x(challenges_aux[2].fe);
        }
        __m256i numbers_[parserParams.nNumbers];
        for(uint64_t i = 0; i < parserParams.nNumbers; ++i) {
            numbers_[i] = _mm256_set1_epi64x(numbers[i]);
        }
        __m256i publics[starkInfo.nPublics];
        for(uint64_t i = 0; i < starkInfo.nPublics; ++i) {
            publics[i] = _mm256_set1_epi64x(params.publicInputs[i].fe);
        }
        Goldilocks3::Element_avx evals[params.evals.degree()];
        for(uint64_t i = 0; i < params.evals.degree(); ++i) {
            evals[i][0] = _mm256_set1_epi64x(params.evals[i][0].fe);
            evals[i][1] = _mm256_set1_epi64x(params.evals[i][1].fe);
            evals[i][2] = _mm256_set1_epi64x(params.evals[i][2].fe);
        }
    #pragma omp parallel for
        for (uint64_t i = 0; i < domainSize; i+= nrowsBatch) {
            uint64_t i_args = 0;

            bool const needModule = i + nrowsBatch + nextStride >= domainSize;
            __m256i bufferT_[2*nCols];
            __m256i tmp1[parserParams.nTemp1];
            __m256i tmp1_1;
            Goldilocks3::Element_avx tmp3[parserParams.nTemp3];
            Goldilocks3::Element_avx tmp3_;
            Goldilocks3::Element_avx tmp3_1;
            __m256i tmp1_0;
            Goldilocks::Element bufferT[2*nrowsBatch];

            for(uint64_t k = 0; k < nColsSteps[0]; ++k) {
                for(uint64_t o = 0; o < 2; ++o) {
                    for(uint64_t j = 0; j < nrowsBatch; ++j) {
                        uint64_t l = (i + j + nextStrides[o]) % domainSize;
                        bufferT[nrowsBatch*o + j] = ((Goldilocks::Element *)constPols->address())[l * nColsSteps[0] + k];
                    }
                    Goldilocks::load_avx(bufferT_[2 * k + o], &bufferT[nrowsBatch*o]);
                }
            }
            for(uint64_t s = 1; s <= nStages; ++s) {
                if(parserParams.stage < s) break;
                for(uint64_t k = 0; k < nColsSteps[s]; ++k) {
                    for(uint64_t o = 0; o < 2; ++o) {
                        for(uint64_t j = 0; j < nrowsBatch; ++j) {
                            uint64_t l = (i + j + nextStrides[o]) % domainSize;
                            bufferT[nrowsBatch*o + j] = params.pols[offsetsSteps[s] + l * nColsSteps[s] + k];
                        }
                        Goldilocks::load_avx(bufferT_[2 * (nColsStepsAccumulated[s] + k) + o], &bufferT[nrowsBatch*o]);
                    }
                }
            }
            for(uint64_t k = 0; k < nColsSteps[nStages + 1]; ++k) {
                for(uint64_t o = 0; o < 2; ++o) {
                    for(uint64_t j = 0; j < nrowsBatch; ++j) {
                        uint64_t l = (i + j + nextStrides[o]) % domainSize;
                        if(!domainExtended) {
                            bufferT[nrowsBatch*o + j] = params.pols[offsetsSteps[nStages + 1] + k * domainSize + l];
                        } else {
                            bufferT[nrowsBatch*o + j] = params.pols[offsetsSteps[nStages + 1] + l * nColsSteps[nStages + 1] + k];
                        }
                    }
                    Goldilocks::load_avx(bufferT_[2 * (nColsStepsAccumulated[nStages + 1] + k) + o], &bufferT[nrowsBatch*o]);
                }
            }
    

            for (uint64_t kk = 0; kk < parserParams.nOps; ++kk) {
                switch (ops[kk]) {
                case 0: {
                    // COPY commit1 to commit1
                Goldilocks::copy_avx(bufferT_[buffTOffsetsSteps_[args[i_args]] + 2 * args[i_args + 1] + args[i_args + 2]], bufferT_[buffTOffsetsSteps_[args[i_args + 3]] + 2 * args[i_args + 4] + args[i_args + 5]]);
                storePolinomial(params.pols, bufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, 2, domainSize, domainExtended, nStages, needModule, i, args[i_args], args[i_args + 1], args[i_args + 2], 1);
                    i_args += 6;
                    break;
            }
                case 1: {
                    // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: commit1
                Goldilocks::op_avx(args[i_args], bufferT_[buffTOffsetsSteps_[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], bufferT_[buffTOffsetsSteps_[args[i_args + 4]] + 2 * args[i_args + 5] + args[i_args + 6]], bufferT_[buffTOffsetsSteps_[args[i_args + 7]] + 2 * args[i_args + 8] + args[i_args + 9]]);
                storePolinomial(params.pols, bufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, 2, domainSize, domainExtended, nStages, needModule, i, args[i_args + 1], args[i_args + 2], args[i_args + 3], 1);
                    i_args += 10;
                    break;
            }
                case 2: {
                    // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: tmp1
                Goldilocks::op_avx(args[i_args], bufferT_[buffTOffsetsSteps_[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], bufferT_[buffTOffsetsSteps_[args[i_args + 4]] + 2 * args[i_args + 5] + args[i_args + 6]], tmp1[args[i_args + 7]]);
                storePolinomial(params.pols, bufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, 2, domainSize, domainExtended, nStages, needModule, i, args[i_args + 1], args[i_args + 2], args[i_args + 3], 1);
                    i_args += 8;
                    break;
            }
                case 3: {
                    // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: public
                Goldilocks::op_avx(args[i_args], bufferT_[buffTOffsetsSteps_[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], bufferT_[buffTOffsetsSteps_[args[i_args + 4]] + 2 * args[i_args + 5] + args[i_args + 6]], publics[args[i_args + 7]]);
                storePolinomial(params.pols, bufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, 2, domainSize, domainExtended, nStages, needModule, i, args[i_args + 1], args[i_args + 2], args[i_args + 3], 1);
                    i_args += 8;
                    break;
            }
                case 4: {
                    // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: number
                Goldilocks::op_avx(args[i_args], bufferT_[buffTOffsetsSteps_[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], bufferT_[buffTOffsetsSteps_[args[i_args + 4]] + 2 * args[i_args + 5] + args[i_args + 6]], numbers_[args[i_args + 7]]);
                storePolinomial(params.pols, bufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, 2, domainSize, domainExtended, nStages, needModule, i, args[i_args + 1], args[i_args + 2], args[i_args + 3], 1);
                    i_args += 8;
                    break;
            }
                case 5: {
                    // COPY tmp1 to commit1
                Goldilocks::copy_avx(bufferT_[buffTOffsetsSteps_[args[i_args]] + 2 * args[i_args + 1] + args[i_args + 2]], tmp1[args[i_args + 3]]);
                storePolinomial(params.pols, bufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, 2, domainSize, domainExtended, nStages, needModule, i, args[i_args], args[i_args + 1], args[i_args + 2], 1);
                    i_args += 4;
                    break;
            }
                case 6: {
                    // OPERATION WITH DEST: commit1 - SRC0: tmp1 - SRC1: tmp1
                Goldilocks::op_avx(args[i_args], bufferT_[buffTOffsetsSteps_[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], tmp1[args[i_args + 4]], tmp1[args[i_args + 5]]);
                storePolinomial(params.pols, bufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, 2, domainSize, domainExtended, nStages, needModule, i, args[i_args + 1], args[i_args + 2], args[i_args + 3], 1);
                    i_args += 6;
                    break;
            }
                case 7: {
                    // OPERATION WITH DEST: commit1 - SRC0: tmp1 - SRC1: public
                Goldilocks::op_avx(args[i_args], bufferT_[buffTOffsetsSteps_[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], tmp1[args[i_args + 4]], publics[args[i_args + 5]]);
                storePolinomial(params.pols, bufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, 2, domainSize, domainExtended, nStages, needModule, i, args[i_args + 1], args[i_args + 2], args[i_args + 3], 1);
                    i_args += 6;
                    break;
            }
                case 8: {
                    // OPERATION WITH DEST: commit1 - SRC0: tmp1 - SRC1: number
                Goldilocks::op_avx(args[i_args], bufferT_[buffTOffsetsSteps_[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], tmp1[args[i_args + 4]], numbers_[args[i_args + 5]]);
                storePolinomial(params.pols, bufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, 2, domainSize, domainExtended, nStages, needModule, i, args[i_args + 1], args[i_args + 2], args[i_args + 3], 1);
                    i_args += 6;
                    break;
            }
                case 9: {
                    // COPY public to commit1
                Goldilocks::copy_avx(bufferT_[buffTOffsetsSteps_[args[i_args]] + 2 * args[i_args + 1] + args[i_args + 2]], publics[args[i_args + 3]]);
                storePolinomial(params.pols, bufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, 2, domainSize, domainExtended, nStages, needModule, i, args[i_args], args[i_args + 1], args[i_args + 2], 1);
                    i_args += 4;
                    break;
            }
                case 10: {
                    // OPERATION WITH DEST: commit1 - SRC0: public - SRC1: public
                Goldilocks::op_avx(args[i_args], bufferT_[buffTOffsetsSteps_[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], publics[args[i_args + 4]], publics[args[i_args + 5]]);
                storePolinomial(params.pols, bufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, 2, domainSize, domainExtended, nStages, needModule, i, args[i_args + 1], args[i_args + 2], args[i_args + 3], 1);
                    i_args += 6;
                    break;
            }
                case 11: {
                    // OPERATION WITH DEST: commit1 - SRC0: public - SRC1: number
                Goldilocks::op_avx(args[i_args], bufferT_[buffTOffsetsSteps_[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], publics[args[i_args + 4]], numbers_[args[i_args + 5]]);
                storePolinomial(params.pols, bufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, 2, domainSize, domainExtended, nStages, needModule, i, args[i_args + 1], args[i_args + 2], args[i_args + 3], 1);
                    i_args += 6;
                    break;
            }
                case 12: {
                    // COPY x to commit1
                    Goldilocks::load_avx(tmp1_0, x[i], x.offset());
                Goldilocks::copy_avx(bufferT_[buffTOffsetsSteps_[args[i_args]] + 2 * args[i_args + 1] + args[i_args + 2]], tmp1_0);
                storePolinomial(params.pols, bufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, 2, domainSize, domainExtended, nStages, needModule, i, args[i_args], args[i_args + 1], args[i_args + 2], 1);
                    i_args += 3;
                    break;
            }
                case 13: {
                    // COPY number to commit1
                Goldilocks::copy_avx(bufferT_[buffTOffsetsSteps_[args[i_args]] + 2 * args[i_args + 1] + args[i_args + 2]], numbers_[args[i_args + 3]]);
                storePolinomial(params.pols, bufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, 2, domainSize, domainExtended, nStages, needModule, i, args[i_args], args[i_args + 1], args[i_args + 2], 1);
                    i_args += 4;
                    break;
            }
                case 14: {
                    // OPERATION WITH DEST: commit1 - SRC0: number - SRC1: number
                Goldilocks::op_avx(args[i_args], bufferT_[buffTOffsetsSteps_[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], numbers_[args[i_args + 4]], numbers_[args[i_args + 5]]);
                storePolinomial(params.pols, bufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, 2, domainSize, domainExtended, nStages, needModule, i, args[i_args + 1], args[i_args + 2], args[i_args + 3], 1);
                    i_args += 6;
                    break;
            }
                case 15: {
                    // COPY commit1 to tmp1
                Goldilocks::copy_avx(tmp1[args[i_args]], bufferT_[buffTOffsetsSteps_[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]]);
                    i_args += 4;
                    break;
            }
                case 16: {
                    // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: commit1
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[buffTOffsetsSteps_[args[i_args + 2]] + 2 * args[i_args + 3] + args[i_args + 4]], bufferT_[buffTOffsetsSteps_[args[i_args + 5]] + 2 * args[i_args + 6] + args[i_args + 7]]);
                    i_args += 8;
                    break;
            }
                case 17: {
                    // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: tmp1
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[buffTOffsetsSteps_[args[i_args + 2]] + 2 * args[i_args + 3] + args[i_args + 4]], tmp1[args[i_args + 5]]);
                    i_args += 6;
                    break;
            }
                case 18: {
                    // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: public
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[buffTOffsetsSteps_[args[i_args + 2]] + 2 * args[i_args + 3] + args[i_args + 4]], publics[args[i_args + 5]]);
                    i_args += 6;
                    break;
            }
                case 19: {
                    // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: number
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[buffTOffsetsSteps_[args[i_args + 2]] + 2 * args[i_args + 3] + args[i_args + 4]], numbers_[args[i_args + 5]]);
                    i_args += 6;
                    break;
            }
                case 20: {
                    // COPY tmp1 to tmp1
                Goldilocks::copy_avx(tmp1[args[i_args]], tmp1[args[i_args + 1]]);
                    i_args += 2;
                    break;
            }
                case 21: {
                    // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: tmp1
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], tmp1[args[i_args + 3]]);
                    i_args += 4;
                    break;
            }
                case 22: {
                    // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: public
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], publics[args[i_args + 3]]);
                    i_args += 4;
                    break;
            }
                case 23: {
                    // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: number
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], numbers_[args[i_args + 3]]);
                    i_args += 4;
                    break;
            }
                case 24: {
                    // COPY public to tmp1
                Goldilocks::copy_avx(tmp1[args[i_args]], publics[args[i_args + 1]]);
                    i_args += 2;
                    break;
            }
                case 25: {
                    // OPERATION WITH DEST: tmp1 - SRC0: public - SRC1: public
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], publics[args[i_args + 2]], publics[args[i_args + 3]]);
                    i_args += 4;
                    break;
            }
                case 26: {
                    // OPERATION WITH DEST: tmp1 - SRC0: public - SRC1: number
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], publics[args[i_args + 2]], numbers_[args[i_args + 3]]);
                    i_args += 4;
                    break;
            }
                case 27: {
                    // COPY x to tmp1
                    Goldilocks::load_avx(tmp1_0, x[i], x.offset());
                Goldilocks::copy_avx(tmp1[args[i_args]], tmp1_0);
                    i_args += 1;
                    break;
            }
                case 28: {
                    // COPY number to tmp1
                Goldilocks::copy_avx(tmp1[args[i_args]], numbers_[args[i_args + 1]]);
                    i_args += 2;
                    break;
            }
                case 29: {
                    // OPERATION WITH DEST: tmp1 - SRC0: number - SRC1: number
                Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], numbers_[args[i_args + 2]], numbers_[args[i_args + 3]]);
                    i_args += 4;
                    break;
            }
                case 30: {
                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: commit1
                Goldilocks3::op_31_avx(args[i_args], &bufferT_[buffTOffsetsSteps_[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &bufferT_[buffTOffsetsSteps_[args[i_args + 4]] + 2 * args[i_args + 5] + args[i_args + 6]], 2, 
                        bufferT_[buffTOffsetsSteps_[args[i_args + 7]] + 2 * args[i_args + 8] + args[i_args + 9]]);
                storePolinomial(params.pols, bufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, 2, domainSize, domainExtended, nStages, needModule, i, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 10;
                    break;
            }
                case 31: {
                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: tmp1
                Goldilocks3::op_31_avx(args[i_args], &bufferT_[buffTOffsetsSteps_[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &bufferT_[buffTOffsetsSteps_[args[i_args + 4]] + 2 * args[i_args + 5] + args[i_args + 6]], 2, 
                        tmp1[args[i_args + 7]]);
                storePolinomial(params.pols, bufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, 2, domainSize, domainExtended, nStages, needModule, i, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 8;
                    break;
            }
                case 32: {
                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: public
                Goldilocks3::op_31_avx(args[i_args], &bufferT_[buffTOffsetsSteps_[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &bufferT_[buffTOffsetsSteps_[args[i_args + 4]] + 2 * args[i_args + 5] + args[i_args + 6]], 2, 
                        publics[args[i_args + 7]]);
                storePolinomial(params.pols, bufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, 2, domainSize, domainExtended, nStages, needModule, i, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 8;
                    break;
            }
                case 33: {
                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: x
                Goldilocks::load_avx(tmp1_1, x[i], x.offset());
                Goldilocks3::op_31_avx(args[i_args], &bufferT_[buffTOffsetsSteps_[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &bufferT_[buffTOffsetsSteps_[args[i_args + 4]] + 2 * args[i_args + 5] + args[i_args + 6]], 2, 
                        tmp1_1);
                storePolinomial(params.pols, bufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, 2, domainSize, domainExtended, nStages, needModule, i, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 7;
                    break;
            }
                case 34: {
                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: number
                Goldilocks3::op_31_avx(args[i_args], &bufferT_[buffTOffsetsSteps_[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &bufferT_[buffTOffsetsSteps_[args[i_args + 4]] + 2 * args[i_args + 5] + args[i_args + 6]], 2, 
                        numbers_[args[i_args + 7]]);
                storePolinomial(params.pols, bufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, 2, domainSize, domainExtended, nStages, needModule, i, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 8;
                    break;
            }
                case 35: {
                    // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: commit1
                Goldilocks3::op_31_avx(args[i_args], &bufferT_[buffTOffsetsSteps_[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &(tmp3[args[i_args + 4]][0]), 1, 
                        bufferT_[buffTOffsetsSteps_[args[i_args + 5]] + 2 * args[i_args + 6] + args[i_args + 7]]);
                storePolinomial(params.pols, bufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, 2, domainSize, domainExtended, nStages, needModule, i, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 8;
                    break;
            }
                case 36: {
                    // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: tmp1
                Goldilocks3::op_31_avx(args[i_args], &bufferT_[buffTOffsetsSteps_[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &(tmp3[args[i_args + 4]][0]), 1, 
                        tmp1[args[i_args + 5]]);
                storePolinomial(params.pols, bufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, 2, domainSize, domainExtended, nStages, needModule, i, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 6;
                    break;
            }
                case 37: {
                    // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: public
                Goldilocks3::op_31_avx(args[i_args], &bufferT_[buffTOffsetsSteps_[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &(tmp3[args[i_args + 4]][0]), 1, 
                        publics[args[i_args + 5]]);
                storePolinomial(params.pols, bufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, 2, domainSize, domainExtended, nStages, needModule, i, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 6;
                    break;
            }
                case 38: {
                    // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: x
                Goldilocks::load_avx(tmp1_1, x[i], x.offset());
                Goldilocks3::op_31_avx(args[i_args], &bufferT_[buffTOffsetsSteps_[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &(tmp3[args[i_args + 4]][0]), 1, 
                        tmp1_1);
                storePolinomial(params.pols, bufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, 2, domainSize, domainExtended, nStages, needModule, i, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 5;
                    break;
            }
                case 39: {
                    // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: number
                Goldilocks3::op_31_avx(args[i_args], &bufferT_[buffTOffsetsSteps_[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &(tmp3[args[i_args + 4]][0]), 1, 
                        numbers_[args[i_args + 5]]);
                storePolinomial(params.pols, bufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, 2, domainSize, domainExtended, nStages, needModule, i, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 6;
                    break;
            }
                case 40: {
                    // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: commit1
                Goldilocks3::op_31_avx(args[i_args], &bufferT_[buffTOffsetsSteps_[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &(challenges[args[i_args + 4]][0]), 1, 
                        bufferT_[buffTOffsetsSteps_[args[i_args + 5]] + 2 * args[i_args + 6] + args[i_args + 7]]);
                storePolinomial(params.pols, bufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, 2, domainSize, domainExtended, nStages, needModule, i, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 8;
                    break;
            }
                case 41: {
                    // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: tmp1
                Goldilocks3::op_31_avx(args[i_args], &bufferT_[buffTOffsetsSteps_[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &(challenges[args[i_args + 4]][0]), 1, 
                        tmp1[args[i_args + 5]]);
                storePolinomial(params.pols, bufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, 2, domainSize, domainExtended, nStages, needModule, i, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 6;
                    break;
            }
                case 42: {
                    // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: public
                Goldilocks3::op_31_avx(args[i_args], &bufferT_[buffTOffsetsSteps_[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &(challenges[args[i_args + 4]][0]), 1, 
                        publics[args[i_args + 5]]);
                storePolinomial(params.pols, bufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, 2, domainSize, domainExtended, nStages, needModule, i, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 6;
                    break;
            }
                case 43: {
                    // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: x
                Goldilocks::load_avx(tmp1_1, x[i], x.offset());
                Goldilocks3::op_31_avx(args[i_args], &bufferT_[buffTOffsetsSteps_[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &(challenges[args[i_args + 4]][0]), 1, 
                        tmp1_1);
                storePolinomial(params.pols, bufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, 2, domainSize, domainExtended, nStages, needModule, i, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 5;
                    break;
            }
                case 44: {
                    // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: number
                Goldilocks3::op_31_avx(args[i_args], &bufferT_[buffTOffsetsSteps_[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &(challenges[args[i_args + 4]][0]), 1, 
                        numbers_[args[i_args + 5]]);
                storePolinomial(params.pols, bufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, 2, domainSize, domainExtended, nStages, needModule, i, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 6;
                    break;
            }
                case 45: {
                    // COPY commit3 to commit3
                Goldilocks3::copy_avx(&bufferT_[buffTOffsetsSteps_[args[i_args]] + 2 * args[i_args + 1] + args[i_args + 2]], 2, 
                        &bufferT_[buffTOffsetsSteps_[args[i_args + 3]] + 2 * args[i_args + 4] + args[i_args + 5]], 2);
                storePolinomial(params.pols, bufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, 2, domainSize, domainExtended, nStages, needModule, i, args[i_args], args[i_args + 1], args[i_args + 2], FIELD_EXTENSION);
                    i_args += 6;
                    break;
            }
                case 46: {
                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: commit3
                Goldilocks3::op_avx(args[i_args], &bufferT_[buffTOffsetsSteps_[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &bufferT_[buffTOffsetsSteps_[args[i_args + 4]] + 2 * args[i_args + 5] + args[i_args + 6]], 2, 
                        &bufferT_[buffTOffsetsSteps_[args[i_args + 7]] + 2 * args[i_args + 8] + args[i_args + 9]], 2);
                storePolinomial(params.pols, bufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, 2, domainSize, domainExtended, nStages, needModule, i, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 10;
                    break;
            }
                case 47: {
                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: tmp3
                Goldilocks3::op_avx(args[i_args], &bufferT_[buffTOffsetsSteps_[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &bufferT_[buffTOffsetsSteps_[args[i_args + 4]] + 2 * args[i_args + 5] + args[i_args + 6]], 2, 
                        &(tmp3[args[i_args + 7]][0]), 1);
                storePolinomial(params.pols, bufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, 2, domainSize, domainExtended, nStages, needModule, i, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 8;
                    break;
            }
                case 48: {
                    // MULTIPLICATION WITH DEST: commit3 - SRC0: commit3 - SRC1: challenge
                Goldilocks3::mul_avx(&bufferT_[buffTOffsetsSteps_[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &bufferT_[buffTOffsetsSteps_[args[i_args + 4]] + 2 * args[i_args + 5] + args[i_args + 6]], 2, 
                        challenges[args[i_args + 7]], challenges_ops[args[i_args + 7]]);
                storePolinomial(params.pols, bufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, 2, domainSize, domainExtended, nStages, needModule, i, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 8;
                    break;
            }
                case 49: {
                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: challenge
                Goldilocks3::op_avx(args[i_args], &bufferT_[buffTOffsetsSteps_[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &bufferT_[buffTOffsetsSteps_[args[i_args + 4]] + 2 * args[i_args + 5] + args[i_args + 6]], 2, 
                        &(challenges[args[i_args + 7]][0]), 1);
                storePolinomial(params.pols, bufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, 2, domainSize, domainExtended, nStages, needModule, i, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 8;
                    break;
            }
                case 50: {
                    // COPY tmp3 to commit3
                Goldilocks3::copy_avx(&bufferT_[buffTOffsetsSteps_[args[i_args]] + 2 * args[i_args + 1] + args[i_args + 2]], 2, 
                        &(tmp3[args[i_args + 3]][0]), 1);
                storePolinomial(params.pols, bufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, 2, domainSize, domainExtended, nStages, needModule, i, args[i_args], args[i_args + 1], args[i_args + 2], FIELD_EXTENSION);
                    i_args += 4;
                    break;
            }
                case 51: {
                    // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: tmp3
                Goldilocks3::op_avx(args[i_args], &bufferT_[buffTOffsetsSteps_[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &(tmp3[args[i_args + 4]][0]), 1, 
                        &(tmp3[args[i_args + 5]][0]), 1);
                storePolinomial(params.pols, bufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, 2, domainSize, domainExtended, nStages, needModule, i, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 6;
                    break;
            }
                case 52: {
                    // MULTIPLICATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: challenge
                Goldilocks3::mul_avx(&bufferT_[buffTOffsetsSteps_[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &(tmp3[args[i_args + 4]][0]), 1, 
                        challenges[args[i_args + 5]], challenges_ops[args[i_args + 5]]);
                storePolinomial(params.pols, bufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, 2, domainSize, domainExtended, nStages, needModule, i, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 6;
                    break;
            }
                case 53: {
                    // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: challenge
                Goldilocks3::op_avx(args[i_args], &bufferT_[buffTOffsetsSteps_[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &(tmp3[args[i_args + 4]][0]), 1, 
                        &(challenges[args[i_args + 5]][0]), 1);
                storePolinomial(params.pols, bufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, 2, domainSize, domainExtended, nStages, needModule, i, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 6;
                    break;
            }
                case 54: {
                    // MULTIPLICATION WITH DEST: commit3 - SRC0: challenge - SRC1: challenge
                Goldilocks3::mul_avx(&bufferT_[buffTOffsetsSteps_[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &(challenges[args[i_args + 4]][0]), 1, 
                        challenges[args[i_args + 5]], challenges_ops[args[i_args + 5]]);
                storePolinomial(params.pols, bufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, 2, domainSize, domainExtended, nStages, needModule, i, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 6;
                    break;
            }
                case 55: {
                    // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: challenge
                Goldilocks3::op_avx(args[i_args], &bufferT_[buffTOffsetsSteps_[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &(challenges[args[i_args + 4]][0]), 1, 
                        &(challenges[args[i_args + 5]][0]), 1);
                storePolinomial(params.pols, bufferT_, nColsSteps, offsetsSteps, buffTOffsetsSteps_, nextStrides, 2, domainSize, domainExtended, nStages, needModule, i, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 6;
                    break;
            }
                case 56: {
                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: commit1
                Goldilocks3::op_31_avx(args[i_args], &(tmp3[args[i_args + 1]][0]), 1, 
                        &bufferT_[buffTOffsetsSteps_[args[i_args + 2]] + 2 * args[i_args + 3] + args[i_args + 4]], 2, 
                        bufferT_[buffTOffsetsSteps_[args[i_args + 5]] + 2 * args[i_args + 6] + args[i_args + 7]]);
                    i_args += 8;
                    break;
            }
                case 57: {
                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: tmp1
                Goldilocks3::op_31_avx(args[i_args], &(tmp3[args[i_args + 1]][0]), 1, 
                        &bufferT_[buffTOffsetsSteps_[args[i_args + 2]] + 2 * args[i_args + 3] + args[i_args + 4]], 2, 
                        tmp1[args[i_args + 5]]);
                    i_args += 6;
                    break;
            }
                case 58: {
                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: public
                Goldilocks3::op_31_avx(args[i_args], &(tmp3[args[i_args + 1]][0]), 1, 
                        &bufferT_[buffTOffsetsSteps_[args[i_args + 2]] + 2 * args[i_args + 3] + args[i_args + 4]], 2, 
                        publics[args[i_args + 5]]);
                    i_args += 6;
                    break;
            }
                case 59: {
                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: x
                Goldilocks::load_avx(tmp1_1, x[i], x.offset());
                Goldilocks3::op_31_avx(args[i_args], &(tmp3[args[i_args + 1]][0]), 1, 
                        &bufferT_[buffTOffsetsSteps_[args[i_args + 2]] + 2 * args[i_args + 3] + args[i_args + 4]], 2, 
                        tmp1_1);
                    i_args += 5;
                    break;
            }
                case 60: {
                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: number
                Goldilocks3::op_31_avx(args[i_args], &(tmp3[args[i_args + 1]][0]), 1, 
                        &bufferT_[buffTOffsetsSteps_[args[i_args + 2]] + 2 * args[i_args + 3] + args[i_args + 4]], 2, 
                        numbers_[args[i_args + 5]]);
                    i_args += 6;
                    break;
            }
                case 61: {
                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: commit1
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], bufferT_[buffTOffsetsSteps_[args[i_args + 3]] + 2 * args[i_args + 4] + args[i_args + 5]]);
                    i_args += 6;
                    break;
            }
                case 62: {
                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: tmp1
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], tmp1[args[i_args + 3]]);
                    i_args += 4;
                    break;
            }
                case 63: {
                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: public
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], publics[args[i_args + 3]]);
                    i_args += 4;
                    break;
            }
                case 64: {
                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: x
                Goldilocks::load_avx(tmp1_1, x[i], x.offset());
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], tmp1_1);
                    i_args += 3;
                    break;
            }
                case 65: {
                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: number
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], numbers_[args[i_args + 3]]);
                    i_args += 4;
                    break;
            }
                case 66: {
                    // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: commit1
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], bufferT_[buffTOffsetsSteps_[args[i_args + 3]] + 2 * args[i_args + 4] + args[i_args + 5]]);
                    i_args += 6;
                    break;
            }
                case 67: {
                    // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: tmp1
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], tmp1[args[i_args + 3]]);
                    i_args += 4;
                    break;
            }
                case 68: {
                    // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: public
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], publics[args[i_args + 3]]);
                    i_args += 4;
                    break;
            }
                case 69: {
                    // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: x
                Goldilocks::load_avx(tmp1_1, x[i], x.offset());
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], tmp1_1);
                    i_args += 3;
                    break;
            }
                case 70: {
                    // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: number
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], numbers_[args[i_args + 3]]);
                    i_args += 4;
                    break;
            }
                case 71: {
                    // COPY commit3 to tmp3
                Goldilocks3::copy_avx(&(tmp3[args[i_args]][0]), 1, 
                        &bufferT_[buffTOffsetsSteps_[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2);
                    i_args += 4;
                    break;
            }
                case 72: {
                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: commit3
                Goldilocks3::op_avx(args[i_args], &(tmp3[args[i_args + 1]][0]), 1, 
                        &bufferT_[buffTOffsetsSteps_[args[i_args + 2]] + 2 * args[i_args + 3] + args[i_args + 4]], 2, 
                        &bufferT_[buffTOffsetsSteps_[args[i_args + 5]] + 2 * args[i_args + 6] + args[i_args + 7]], 2);
                    i_args += 8;
                    break;
            }
                case 73: {
                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: tmp3
                Goldilocks3::op_avx(args[i_args], &(tmp3[args[i_args + 1]][0]), 1, 
                        &bufferT_[buffTOffsetsSteps_[args[i_args + 2]] + 2 * args[i_args + 3] + args[i_args + 4]], 2, 
                        &(tmp3[args[i_args + 5]][0]), 1);
                    i_args += 6;
                    break;
            }
                case 74: {
                    // MULTIPLICATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: challenge
                Goldilocks3::mul_avx(&(tmp3[args[i_args + 1]][0]), 1, 
                        &bufferT_[buffTOffsetsSteps_[args[i_args + 2]] + 2 * args[i_args + 3] + args[i_args + 4]], 2, 
                        challenges[args[i_args + 5]], challenges_ops[args[i_args + 5]]);
                    i_args += 6;
                    break;
            }
                case 75: {
                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: challenge
                Goldilocks3::op_avx(args[i_args], &(tmp3[args[i_args + 1]][0]), 1, 
                        &bufferT_[buffTOffsetsSteps_[args[i_args + 2]] + 2 * args[i_args + 3] + args[i_args + 4]], 2, 
                        &(challenges[args[i_args + 5]][0]), 1);
                    i_args += 6;
                    break;
            }
                case 76: {
                    // COPY tmp3 to tmp3
                Goldilocks3::copy_avx(tmp3[args[i_args]], tmp3[args[i_args + 1]]);
                    i_args += 2;
                    break;
            }
                case 77: {
                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: tmp3
                Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], tmp3[args[i_args + 3]]);
                    i_args += 4;
                    break;
            }
                case 78: {
                    // MULTIPLICATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: challenge
                Goldilocks3::mul_avx(tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                    i_args += 4;
                    break;
            }
                case 79: {
                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: challenge
                Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], challenges[args[i_args + 3]]);
                    i_args += 4;
                    break;
            }
                case 80: {
                    // MULTIPLICATION WITH DEST: tmp3 - SRC0: challenge - SRC1: challenge
                Goldilocks3::mul_avx(tmp3[args[i_args + 1]], challenges[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                    i_args += 4;
                    break;
            }
                case 81: {
                    // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: challenge
                Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], challenges[args[i_args + 3]]);
                    i_args += 4;
                    break;
            }
                case 82: {
                    // COPY eval to tmp3
                Goldilocks3::copy_avx(tmp3[args[i_args]], evals[args[i_args + 1]]);
                    i_args += 2;
                    break;
            }
                case 83: {
                    // MULTIPLICATION WITH DEST: tmp3 - SRC0: eval - SRC1: challenge
                Goldilocks3::mul_avx(tmp3[args[i_args + 1]], evals[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                    i_args += 4;
                    break;
            }
                case 84: {
                    // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: eval
                Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], evals[args[i_args + 3]]);
                    i_args += 4;
                    break;
            }
                case 85: {
                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: eval
                Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], evals[args[i_args + 3]]);
                    i_args += 4;
                    break;
            }
                case 86: {
                    // OPERATION WITH DEST: tmp3 - SRC0: eval - SRC1: commit1
                Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], evals[args[i_args + 2]], bufferT_[buffTOffsetsSteps_[args[i_args + 3]] + 2 * args[i_args + 4] + args[i_args + 5]]);
                    i_args += 6;
                    break;
            }
                case 87: {
                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: eval
                Goldilocks3::op_avx(args[i_args], &(tmp3[args[i_args + 1]][0]), 1, 
                        &bufferT_[buffTOffsetsSteps_[args[i_args + 2]] + 2 * args[i_args + 3] + args[i_args + 4]], 2, 
                        &(evals[args[i_args + 5]][0]), 1);
                    i_args += 6;
                    break;
            }
                case 88: {
                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: xDivXSubXi
                Goldilocks3::load_avx(tmp3_1, params.xDivXSubXi[i + args[i_args + 3]*domainSize], uint64_t(FIELD_EXTENSION));
                Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], tmp3_1);
                    i_args += 4;
                    break;
            }
                case 89: {
                    // OPERATION WITH DEST: q - SRC0: tmp3 - SRC1: Zi
                Goldilocks::Element tmp_inv[3];
                Goldilocks::Element ti0[nrowsBatch];
                Goldilocks::Element ti1[nrowsBatch];
                Goldilocks::Element ti2[nrowsBatch];
                Goldilocks::store_avx(ti0, tmp3[args[i_args]][0]);
                Goldilocks::store_avx(ti1, tmp3[args[i_args]][1]);
                Goldilocks::store_avx(ti2, tmp3[args[i_args]][2]);
                for (uint64_t j = 0; j < nrowsBatch; ++j) {
                    tmp_inv[0] = ti0[j];
                    tmp_inv[1] = ti1[j];
                    tmp_inv[2] = ti2[j];
                    Goldilocks3::mul((Goldilocks3::Element &)(params.q_2ns[(i + j) * FIELD_EXTENSION]), params.zi[i + j][0],(Goldilocks3::Element &)tmp_inv);
                }
                    i_args += 1;
                    break;
            }
                case 90: {
                    // OPERATION WITH DEST: f - SRC0: tmp3 - SRC1: tmp3
                Goldilocks3::op_avx(args[i_args], tmp3_, tmp3[args[i_args + 1]], tmp3[args[i_args + 2]]);
                Goldilocks3::store_avx(&params.f_2ns[i*FIELD_EXTENSION], uint64_t(FIELD_EXTENSION), tmp3_);
                    i_args += 3;
                    break;
            }
                case 91: {
                    // COPY tmp3 to f
                    Goldilocks3::copy_avx(tmp3_, tmp3[args[i_args]]);
                    Goldilocks3::store_avx(&params.f_2ns[i*3], uint64_t(3), tmp3_);
                    i_args += 1;
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
        }
    }
};

#endif