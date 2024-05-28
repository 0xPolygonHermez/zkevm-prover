#ifndef CHELPERS_STEPS_AVX512_HPP
#define CHELPERS_STEPS_AVX512_HPP
#include "chelpers.hpp"
#include "chelpers_steps.hpp"
#include "steps.hpp"

class CHelpersStepsAvx512 : public CHelpersSteps {
public:
    inline virtual void storePolinomial(StarkInfo& starkInfo, Goldilocks::Element *pols, __m512i *bufferT, uint64_t row, uint64_t nrowsPack, bool domainExtended, uint64_t stage, uint64_t stagePos, uint64_t openingPointIndex, uint64_t dim) {
        uint64_t domainSize = domainExtended ? 1 << starkInfo.starkStruct.nBitsExt : 1 << starkInfo.starkStruct.nBits;
        uint64_t nOpenings = 2;
        uint64_t nextStride = domainExtended ? starkInfo.nextStrideExt : starkInfo.nextStride;
        std::vector<uint64_t> nextStrides = domainExtended ? starkInfo.nextStridesExt : starkInfo.nextStrides;
        std::vector<uint64_t> buffTOffsetsStages = domainExtended ? starkInfo.buffTOffsetsStagesExt : starkInfo.buffTOffsetsStages;
        std::vector<uint64_t> nColsStages = domainExtended ? starkInfo.nColsStagesExt : starkInfo.nColsStages;
        std::vector<uint64_t> nColsStagesAcc = domainExtended ? starkInfo.nColsStagesAccExt : starkInfo.nColsStagesAcc;
        std::vector<uint64_t> offsetsStages = domainExtended ? starkInfo.offsetsStagesExt : starkInfo.offsetsStages;
        bool isTmpPol = !domainExtended && stage == 4;
        bool const needModule = row + nrowsPack + nextStride >= domainSize;
        __m512i *buffT = &bufferT[(buffTOffsetsStages[stage] + nOpenings * stagePos + openingPointIndex)];
        if(needModule) {
            uint64_t offsetsDest[nrowsPack];
            uint64_t nextStrideOffset = row + nextStrides[openingPointIndex];
            if(isTmpPol) {
                uint64_t stepOffset = offsetsStages[stage] + stagePos * domainSize;
                for(uint64_t i = 0; i < nrowsPack; ++i) {
                    offsetsDest[i] = stepOffset + ((nextStrideOffset + i) % domainSize) * dim;
                }
                if(dim == 1) {
                    Goldilocks::store_avx512(&pols[0], offsetsDest, buffT[0]);
                } else {
                    Goldilocks3::store_avx512(&pols[0], offsetsDest, buffT, nOpenings);
                }
            } else {
                uint64_t stepOffset = offsetsStages[stage] + stagePos;
                for(uint64_t i = 0; i < nrowsPack; ++i) {
                    offsetsDest[i] = stepOffset + ((nextStrideOffset + i) % domainSize) * nColsStages[stage];
                }
                Goldilocks::store_avx512(&pols[0], offsetsDest, buffT[0]);
            }
        } else {
            if(isTmpPol) {
                if(dim == 1) {
                        Goldilocks::store_avx512(&pols[offsetsStages[stage] + stagePos * domainSize + (row + nextStrides[openingPointIndex])], uint64_t(1), buffT[0]);
                } else {
                        Goldilocks3::store_avx512(&pols[offsetsStages[stage] + stagePos * domainSize + (row + nextStrides[openingPointIndex]) * FIELD_EXTENSION], uint64_t(FIELD_EXTENSION), buffT, nOpenings);
                }
            } else {
                Goldilocks::store_avx512(&pols[offsetsStages[stage] + stagePos + (row + nextStrides[openingPointIndex]) * nColsStages[stage]], nColsStages[stage], buffT[0]);
            }
        }
    }

    inline virtual void storePolinomials(StarkInfo &starkInfo, StepsParams &params, __m512i *bufferT_, vector<uint64_t> storePol, uint64_t row, uint64_t nrowsPack, uint64_t domainExtended) {
        uint64_t nStages = starkInfo.nStages;
        std::vector<uint64_t> nColsStages = domainExtended ? starkInfo.nColsStagesExt : starkInfo.nColsStages;
        std::vector<uint64_t> nColsStagesAcc = domainExtended ? starkInfo.nColsStagesAccExt : starkInfo.nColsStagesAcc;
        for(uint64_t s = 2; s <= nStages + 1; ++s) {
            for(uint64_t k = 0; k < nColsStages[s]; ++k) {
                for(uint64_t o = 0; o < 2; ++o) {
                    if(storePol[2 * (nColsStagesAcc[s] + k) + o]) {
                        storePolinomial(starkInfo, params.pols, bufferT_, row, nrowsPack, domainExtended, s, k, o, storePol[2 * (nColsStagesAcc[s] + k) + o]);
                    }
                }
            }
        }
    }

    inline virtual void setStorePol(std::vector<uint64_t> &storePol, std::vector<uint64_t> buffTOffsetsStages, uint64_t stage, uint64_t stagePos, uint64_t openingPointIndex, uint64_t dim) {
        if(stage == 4) {
            storePol[buffTOffsetsStages[stage] + 2 * stagePos + openingPointIndex] = dim;
        } else {
            if(dim == 1) {
                storePol[buffTOffsetsStages[stage] + 2 * stagePos + openingPointIndex] = 1;
            } else {
                storePol[buffTOffsetsStages[stage] + 2 * stagePos + openingPointIndex] = 1;
                storePol[buffTOffsetsStages[stage] + 2 * stagePos + openingPointIndex + 2] = 1;
                storePol[buffTOffsetsStages[stage] + 2 * stagePos + openingPointIndex + 4] = 1;
            }
        }
    }

    inline virtual void loadPolinomials(StarkInfo &starkInfo, StepsParams &params, __m512i *bufferT_, uint64_t row, uint64_t stage, uint64_t nrowsPack, uint64_t domainExtended) {
        Goldilocks::Element bufferT[2*nrowsPack];
        ConstantPolsStarks *constPols = domainExtended ? params.pConstPols2ns : params.pConstPols;
        uint64_t domainSize = domainExtended ? 1 << starkInfo.starkStruct.nBitsExt : 1 << starkInfo.starkStruct.nBits;
        uint64_t nStages = starkInfo.nStages;
        std::vector<uint64_t> nextStrides = domainExtended ? starkInfo.nextStridesExt : starkInfo.nextStrides;
        std::vector<uint64_t> buffTOffsetsStages = domainExtended ? starkInfo.buffTOffsetsStagesExt : starkInfo.buffTOffsetsStages;
        std::vector<uint64_t> nColsStages = domainExtended ? starkInfo.nColsStagesExt : starkInfo.nColsStages;
        std::vector<uint64_t> nColsStagesAcc = domainExtended ? starkInfo.nColsStagesAccExt : starkInfo.nColsStagesAcc;
        std::vector<uint64_t> offsetsStages = domainExtended ? starkInfo.offsetsStagesExt : starkInfo.offsetsStages;
        for(uint64_t k = 0; k < nColsStages[0]; ++k) {
            for(uint64_t o = 0; o < 2; ++o) {
                for(uint64_t j = 0; j < nrowsPack; ++j) {
                    uint64_t l = (row + j + nextStrides[o]) % domainSize;
                    bufferT[nrowsPack*o + j] = ((Goldilocks::Element *)constPols->address())[l * nColsStages[0] + k];
                }
                Goldilocks::load_avx512(bufferT_[2 * k + o], &bufferT[nrowsPack*o]);
            }
        }
        for(uint64_t s = 1; s <= nStages; ++s) {
            if(stage < s) break;
            for(uint64_t k = 0; k < nColsStages[s]; ++k) {
                for(uint64_t o = 0; o < 2; ++o) {
                    for(uint64_t j = 0; j < nrowsPack; ++j) {
                        uint64_t l = (row + j + nextStrides[o]) % domainSize;
                        bufferT[nrowsPack*o + j] = params.pols[offsetsStages[s] + l * nColsStages[s] + k];
                    }
                    Goldilocks::load_avx512(bufferT_[2 * (nColsStagesAcc[s] + k) + o], &bufferT[nrowsPack*o]);
                }
            }
        }
        for(uint64_t k = 0; k < nColsStages[nStages + 1]; ++k) {
            for(uint64_t o = 0; o < 2; ++o) {
                for(uint64_t j = 0; j < nrowsPack; ++j) {
                    uint64_t l = (row + j + nextStrides[o]) % domainSize;
                    if(!domainExtended) {
                        bufferT[nrowsPack*o + j] = params.pols[offsetsStages[nStages + 1] + k * domainSize + l];
                    } else {
                        bufferT[nrowsPack*o + j] = params.pols[offsetsStages[nStages + 1] + l * nColsStages[nStages + 1] + k];
                    }
                }
                Goldilocks::load_avx512(bufferT_[2 * (nColsStagesAcc[nStages + 1] + k) + o], &bufferT[nrowsPack*o]);
            }
        }
    }

    virtual void calculateExpressions(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams) {
        uint32_t nrowsPack =  8;
        bool domainExtended = parserParams.stage > 3 ? true : false;
        uint64_t domainSize = domainExtended ? 1 << starkInfo.starkStruct.nBitsExt : 1 << starkInfo.starkStruct.nBits;
        Polinomial &x = domainExtended ? params.x_2ns : params.x_n;
        uint8_t *ops = &parserArgs.ops[parserParams.opsOffset];
        uint16_t *args = &parserArgs.args[parserParams.argsOffset];
        uint64_t *numbers = &parserArgs.numbers[parserParams.numbersOffset];

        uint64_t nCols = domainExtended ? starkInfo.nColsExt : starkInfo.nCols;
        std::vector<uint64_t> nextStrides = domainExtended ? starkInfo.nextStridesExt : starkInfo.nextStrides;
        std::vector<uint64_t> buffTOffsetsStages = domainExtended ? starkInfo.buffTOffsetsStagesExt : starkInfo.buffTOffsetsStages;
        std::vector<uint64_t> nColsStages = domainExtended ? starkInfo.nColsStagesExt : starkInfo.nColsStages;
        std::vector<uint64_t> nColsStagesAcc = domainExtended ? starkInfo.nColsStagesAccExt : starkInfo.nColsStagesAcc;
        std::vector<uint64_t> offsetsStages = domainExtended ? starkInfo.offsetsStagesExt : starkInfo.offsetsStages;

        Goldilocks3::Element_avx512 challenges[params.challenges.degree()];
        Goldilocks3::Element_avx512 challenges_ops[params.challenges.degree()];
        for(uint64_t i = 0; i < params.challenges.degree(); ++i) {
            challenges[i][0] = _mm512_set1_epi64(params.challenges[i][0].fe);
            challenges[i][1] = _mm512_set1_epi64(params.challenges[i][1].fe);
            challenges[i][2] = _mm512_set1_epi64(params.challenges[i][2].fe);

            Goldilocks::Element challenges_aux[3];
            challenges_aux[0] = params.challenges[i][0] + params.challenges[i][1];
            challenges_aux[1] = params.challenges[i][0] + params.challenges[i][2];
            challenges_aux[2] = params.challenges[i][1] + params.challenges[i][2];
            challenges_ops[i][0] = _mm512_set1_epi64(challenges_aux[0].fe);
            challenges_ops[i][1] =  _mm512_set1_epi64(challenges_aux[1].fe);
            challenges_ops[i][2] =  _mm512_set1_epi64(challenges_aux[2].fe);
        }
        __m512i numbers_[parserParams.nNumbers];
        for(uint64_t i = 0; i < parserParams.nNumbers; ++i) {
            numbers_[i] = _mm512_set1_epi64(numbers[i]);
        }
        __m512i publics[starkInfo.nPublics];
        for(uint64_t i = 0; i < starkInfo.nPublics; ++i) {
            publics[i] = _mm512_set1_epi64(params.publicInputs[i].fe);
        }

        Goldilocks3::Element_avx512 evals[params.evals.degree()];
        for(uint64_t i = 0; i < params.evals.degree(); ++i) {
            evals[i][0] = _mm512_set1_epi64(params.evals[i][0].fe);
            evals[i][1] = _mm512_set1_epi64(params.evals[i][1].fe);
            evals[i][2] = _mm512_set1_epi64(params.evals[i][2].fe);
        }

    #pragma omp parallel for
        for (uint64_t i = 0; i < domainSize; i+= nrowsPack) {
            uint64_t i_args = 0;

            std::vector<uint64_t> storePol(2*nCols, 0);

            __m512i bufferT_[2*nCols];

            __m512i tmp1[parserParams.nTemp1];
            __m512i tmp1_1;
            __m512i tmp1_0;
    

            Goldilocks3::Element_avx512 tmp3[parserParams.nTemp3];
            Goldilocks3::Element_avx512 tmp3_;
            Goldilocks3::Element_avx512 tmp3_1;

            loadPolinomials(starkInfo, params, bufferT_, i, parserParams.stage, nrowsPack, domainExtended);

            for (uint64_t kk = 0; kk < parserParams.nOps; ++kk) {
                switch (ops[kk]) {
                case 0: {
                    // COPY commit1 to commit1
                    Goldilocks::copy_avx512(bufferT_[buffTOffsetsStages[args[i_args]] + 2 * args[i_args + 1] + args[i_args + 2]], bufferT_[buffTOffsetsStages[args[i_args + 3]] + 2 * args[i_args + 4] + args[i_args + 5]]);
                    setStorePol(storePol, buffTOffsetsStages, args[i_args], args[i_args + 1], args[i_args + 2], 1);
                    i_args += 6;
                    break;
                }
                case 1: {
                    // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: commit1
                    Goldilocks::op_avx512(args[i_args], bufferT_[buffTOffsetsStages[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], bufferT_[buffTOffsetsStages[args[i_args + 4]] + 2 * args[i_args + 5] + args[i_args + 6]], bufferT_[buffTOffsetsStages[args[i_args + 7]] + 2 * args[i_args + 8] + args[i_args + 9]]);
                    setStorePol(storePol, buffTOffsetsStages, args[i_args + 1], args[i_args + 2], args[i_args + 3], 1);
                    i_args += 10;
                    break;
                }
                case 2: {
                    // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: tmp1
                    Goldilocks::op_avx512(args[i_args], bufferT_[buffTOffsetsStages[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], bufferT_[buffTOffsetsStages[args[i_args + 4]] + 2 * args[i_args + 5] + args[i_args + 6]], tmp1[args[i_args + 7]]);
                    setStorePol(storePol, buffTOffsetsStages, args[i_args + 1], args[i_args + 2], args[i_args + 3], 1);
                    i_args += 8;
                    break;
                }
                case 3: {
                    // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: public
                    Goldilocks::op_avx512(args[i_args], bufferT_[buffTOffsetsStages[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], bufferT_[buffTOffsetsStages[args[i_args + 4]] + 2 * args[i_args + 5] + args[i_args + 6]], publics[args[i_args + 7]]);
                    setStorePol(storePol, buffTOffsetsStages, args[i_args + 1], args[i_args + 2], args[i_args + 3], 1);
                    i_args += 8;
                    break;
                }
                case 4: {
                    // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: number
                    Goldilocks::op_avx512(args[i_args], bufferT_[buffTOffsetsStages[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], bufferT_[buffTOffsetsStages[args[i_args + 4]] + 2 * args[i_args + 5] + args[i_args + 6]], numbers_[args[i_args + 7]]);
                    setStorePol(storePol, buffTOffsetsStages, args[i_args + 1], args[i_args + 2], args[i_args + 3], 1);
                    i_args += 8;
                    break;
                }
                case 5: {
                    // COPY tmp1 to commit1
                    Goldilocks::copy_avx512(bufferT_[buffTOffsetsStages[args[i_args]] + 2 * args[i_args + 1] + args[i_args + 2]], tmp1[args[i_args + 3]]);
                    setStorePol(storePol, buffTOffsetsStages, args[i_args], args[i_args + 1], args[i_args + 2], 1);
                    i_args += 4;
                    break;
                }
                case 6: {
                    // OPERATION WITH DEST: commit1 - SRC0: tmp1 - SRC1: tmp1
                    Goldilocks::op_avx512(args[i_args], bufferT_[buffTOffsetsStages[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], tmp1[args[i_args + 4]], tmp1[args[i_args + 5]]);
                    setStorePol(storePol, buffTOffsetsStages, args[i_args + 1], args[i_args + 2], args[i_args + 3], 1);
                    i_args += 6;
                    break;
                }
                case 7: {
                    // OPERATION WITH DEST: commit1 - SRC0: tmp1 - SRC1: public
                    Goldilocks::op_avx512(args[i_args], bufferT_[buffTOffsetsStages[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], tmp1[args[i_args + 4]], publics[args[i_args + 5]]);
                    setStorePol(storePol, buffTOffsetsStages, args[i_args + 1], args[i_args + 2], args[i_args + 3], 1);
                    i_args += 6;
                    break;
                }
                case 8: {
                    // OPERATION WITH DEST: commit1 - SRC0: tmp1 - SRC1: number
                    Goldilocks::op_avx512(args[i_args], bufferT_[buffTOffsetsStages[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], tmp1[args[i_args + 4]], numbers_[args[i_args + 5]]);
                    setStorePol(storePol, buffTOffsetsStages, args[i_args + 1], args[i_args + 2], args[i_args + 3], 1);
                    i_args += 6;
                    break;
                }
                case 9: {
                    // COPY public to commit1
                    Goldilocks::copy_avx512(bufferT_[buffTOffsetsStages[args[i_args]] + 2 * args[i_args + 1] + args[i_args + 2]], publics[args[i_args + 3]]);
                    setStorePol(storePol, buffTOffsetsStages, args[i_args], args[i_args + 1], args[i_args + 2], 1);
                    i_args += 4;
                    break;
                }
                case 10: {
                    // OPERATION WITH DEST: commit1 - SRC0: public - SRC1: public
                    Goldilocks::op_avx512(args[i_args], bufferT_[buffTOffsetsStages[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], publics[args[i_args + 4]], publics[args[i_args + 5]]);
                    setStorePol(storePol, buffTOffsetsStages, args[i_args + 1], args[i_args + 2], args[i_args + 3], 1);
                    i_args += 6;
                    break;
                }
                case 11: {
                    // OPERATION WITH DEST: commit1 - SRC0: public - SRC1: number
                    Goldilocks::op_avx512(args[i_args], bufferT_[buffTOffsetsStages[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], publics[args[i_args + 4]], numbers_[args[i_args + 5]]);
                    setStorePol(storePol, buffTOffsetsStages, args[i_args + 1], args[i_args + 2], args[i_args + 3], 1);
                    i_args += 6;
                    break;
                }
                case 12: {
                    // COPY x to commit1
                        Goldilocks::load_avx512(tmp1_0, x[i], x.offset());
                    Goldilocks::copy_avx512(bufferT_[buffTOffsetsStages[args[i_args]] + 2 * args[i_args + 1] + args[i_args + 2]], tmp1_0);
                    setStorePol(storePol, buffTOffsetsStages, args[i_args], args[i_args + 1], args[i_args + 2], 1);
                    i_args += 3;
                    break;
                }
                case 13: {
                    // COPY number to commit1
                    Goldilocks::copy_avx512(bufferT_[buffTOffsetsStages[args[i_args]] + 2 * args[i_args + 1] + args[i_args + 2]], numbers_[args[i_args + 3]]);
                    setStorePol(storePol, buffTOffsetsStages, args[i_args], args[i_args + 1], args[i_args + 2], 1);
                    i_args += 4;
                    break;
                }
                case 14: {
                    // OPERATION WITH DEST: commit1 - SRC0: number - SRC1: number
                    Goldilocks::op_avx512(args[i_args], bufferT_[buffTOffsetsStages[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], numbers_[args[i_args + 4]], numbers_[args[i_args + 5]]);
                    setStorePol(storePol, buffTOffsetsStages, args[i_args + 1], args[i_args + 2], args[i_args + 3], 1);
                    i_args += 6;
                    break;
                }
                case 15: {
                    // COPY commit1 to tmp1
                    Goldilocks::copy_avx512(tmp1[args[i_args]], bufferT_[buffTOffsetsStages[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 16: {
                    // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: commit1
                    Goldilocks::op_avx512(args[i_args], tmp1[args[i_args + 1]], bufferT_[buffTOffsetsStages[args[i_args + 2]] + 2 * args[i_args + 3] + args[i_args + 4]], bufferT_[buffTOffsetsStages[args[i_args + 5]] + 2 * args[i_args + 6] + args[i_args + 7]]);
                    i_args += 8;
                    break;
                }
                case 17: {
                    // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: tmp1
                    Goldilocks::op_avx512(args[i_args], tmp1[args[i_args + 1]], bufferT_[buffTOffsetsStages[args[i_args + 2]] + 2 * args[i_args + 3] + args[i_args + 4]], tmp1[args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 18: {
                    // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: public
                    Goldilocks::op_avx512(args[i_args], tmp1[args[i_args + 1]], bufferT_[buffTOffsetsStages[args[i_args + 2]] + 2 * args[i_args + 3] + args[i_args + 4]], publics[args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 19: {
                    // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: number
                    Goldilocks::op_avx512(args[i_args], tmp1[args[i_args + 1]], bufferT_[buffTOffsetsStages[args[i_args + 2]] + 2 * args[i_args + 3] + args[i_args + 4]], numbers_[args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 20: {
                    // COPY tmp1 to tmp1
                    Goldilocks::copy_avx512(tmp1[args[i_args]], tmp1[args[i_args + 1]]);
                    i_args += 2;
                    break;
                }
                case 21: {
                    // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: tmp1
                    Goldilocks::op_avx512(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], tmp1[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 22: {
                    // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: public
                    Goldilocks::op_avx512(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], publics[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 23: {
                    // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: number
                    Goldilocks::op_avx512(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], numbers_[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 24: {
                    // COPY public to tmp1
                    Goldilocks::copy_avx512(tmp1[args[i_args]], publics[args[i_args + 1]]);
                    i_args += 2;
                    break;
                }
                case 25: {
                    // OPERATION WITH DEST: tmp1 - SRC0: public - SRC1: public
                    Goldilocks::op_avx512(args[i_args], tmp1[args[i_args + 1]], publics[args[i_args + 2]], publics[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 26: {
                    // OPERATION WITH DEST: tmp1 - SRC0: public - SRC1: number
                    Goldilocks::op_avx512(args[i_args], tmp1[args[i_args + 1]], publics[args[i_args + 2]], numbers_[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 27: {
                    // COPY x to tmp1
                        Goldilocks::load_avx512(tmp1_0, x[i], x.offset());
                    Goldilocks::copy_avx512(tmp1[args[i_args]], tmp1_0);
                    i_args += 1;
                    break;
                }
                case 28: {
                    // COPY number to tmp1
                    Goldilocks::copy_avx512(tmp1[args[i_args]], numbers_[args[i_args + 1]]);
                    i_args += 2;
                    break;
                }
                case 29: {
                    // OPERATION WITH DEST: tmp1 - SRC0: number - SRC1: number
                    Goldilocks::op_avx512(args[i_args], tmp1[args[i_args + 1]], numbers_[args[i_args + 2]], numbers_[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 30: {
                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: commit1
                    Goldilocks3::op_31_avx512(args[i_args], &bufferT_[buffTOffsetsStages[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &bufferT_[buffTOffsetsStages[args[i_args + 4]] + 2 * args[i_args + 5] + args[i_args + 6]], 2, 
                        bufferT_[buffTOffsetsStages[args[i_args + 7]] + 2 * args[i_args + 8] + args[i_args + 9]]);
                    setStorePol(storePol, buffTOffsetsStages, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 10;
                    break;
                }
                case 31: {
                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: tmp1
                    Goldilocks3::op_31_avx512(args[i_args], &bufferT_[buffTOffsetsStages[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &bufferT_[buffTOffsetsStages[args[i_args + 4]] + 2 * args[i_args + 5] + args[i_args + 6]], 2, 
                        tmp1[args[i_args + 7]]);
                    setStorePol(storePol, buffTOffsetsStages, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 8;
                    break;
                }
                case 32: {
                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: public
                    Goldilocks3::op_31_avx512(args[i_args], &bufferT_[buffTOffsetsStages[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &bufferT_[buffTOffsetsStages[args[i_args + 4]] + 2 * args[i_args + 5] + args[i_args + 6]], 2, 
                        publics[args[i_args + 7]]);
                    setStorePol(storePol, buffTOffsetsStages, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 8;
                    break;
                }
                case 33: {
                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: x
                    Goldilocks::load_avx512(tmp1_1, x[i], x.offset());
                    Goldilocks3::op_31_avx512(args[i_args], &bufferT_[buffTOffsetsStages[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &bufferT_[buffTOffsetsStages[args[i_args + 4]] + 2 * args[i_args + 5] + args[i_args + 6]], 2, 
                        tmp1_1);
                    setStorePol(storePol, buffTOffsetsStages, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 7;
                    break;
                }
                case 34: {
                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: number
                    Goldilocks3::op_31_avx512(args[i_args], &bufferT_[buffTOffsetsStages[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &bufferT_[buffTOffsetsStages[args[i_args + 4]] + 2 * args[i_args + 5] + args[i_args + 6]], 2, 
                        numbers_[args[i_args + 7]]);
                    setStorePol(storePol, buffTOffsetsStages, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 8;
                    break;
                }
                case 35: {
                    // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: commit1
                    Goldilocks3::op_31_avx512(args[i_args], &bufferT_[buffTOffsetsStages[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &(tmp3[args[i_args + 4]][0]), 1, 
                        bufferT_[buffTOffsetsStages[args[i_args + 5]] + 2 * args[i_args + 6] + args[i_args + 7]]);
                    setStorePol(storePol, buffTOffsetsStages, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 8;
                    break;
                }
                case 36: {
                    // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: tmp1
                    Goldilocks3::op_31_avx512(args[i_args], &bufferT_[buffTOffsetsStages[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &(tmp3[args[i_args + 4]][0]), 1, 
                        tmp1[args[i_args + 5]]);
                    setStorePol(storePol, buffTOffsetsStages, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 6;
                    break;
                }
                case 37: {
                    // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: public
                    Goldilocks3::op_31_avx512(args[i_args], &bufferT_[buffTOffsetsStages[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &(tmp3[args[i_args + 4]][0]), 1, 
                        publics[args[i_args + 5]]);
                    setStorePol(storePol, buffTOffsetsStages, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 6;
                    break;
                }
                case 38: {
                    // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: x
                    Goldilocks::load_avx512(tmp1_1, x[i], x.offset());
                    Goldilocks3::op_31_avx512(args[i_args], &bufferT_[buffTOffsetsStages[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &(tmp3[args[i_args + 4]][0]), 1, 
                        tmp1_1);
                    setStorePol(storePol, buffTOffsetsStages, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 5;
                    break;
                }
                case 39: {
                    // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: number
                    Goldilocks3::op_31_avx512(args[i_args], &bufferT_[buffTOffsetsStages[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &(tmp3[args[i_args + 4]][0]), 1, 
                        numbers_[args[i_args + 5]]);
                    setStorePol(storePol, buffTOffsetsStages, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 6;
                    break;
                }
                case 40: {
                    // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: commit1
                    Goldilocks3::op_31_avx512(args[i_args], &bufferT_[buffTOffsetsStages[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &(challenges[args[i_args + 4]][0]), 1, 
                        bufferT_[buffTOffsetsStages[args[i_args + 5]] + 2 * args[i_args + 6] + args[i_args + 7]]);
                    setStorePol(storePol, buffTOffsetsStages, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 8;
                    break;
                }
                case 41: {
                    // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: tmp1
                    Goldilocks3::op_31_avx512(args[i_args], &bufferT_[buffTOffsetsStages[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &(challenges[args[i_args + 4]][0]), 1, 
                        tmp1[args[i_args + 5]]);
                    setStorePol(storePol, buffTOffsetsStages, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 6;
                    break;
                }
                case 42: {
                    // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: public
                    Goldilocks3::op_31_avx512(args[i_args], &bufferT_[buffTOffsetsStages[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &(challenges[args[i_args + 4]][0]), 1, 
                        publics[args[i_args + 5]]);
                    setStorePol(storePol, buffTOffsetsStages, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 6;
                    break;
                }
                case 43: {
                    // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: x
                    Goldilocks::load_avx512(tmp1_1, x[i], x.offset());
                    Goldilocks3::op_31_avx512(args[i_args], &bufferT_[buffTOffsetsStages[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &(challenges[args[i_args + 4]][0]), 1, 
                        tmp1_1);
                    setStorePol(storePol, buffTOffsetsStages, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 5;
                    break;
                }
                case 44: {
                    // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: number
                    Goldilocks3::op_31_avx512(args[i_args], &bufferT_[buffTOffsetsStages[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &(challenges[args[i_args + 4]][0]), 1, 
                        numbers_[args[i_args + 5]]);
                    setStorePol(storePol, buffTOffsetsStages, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 6;
                    break;
                }
                case 45: {
                    // COPY commit3 to commit3
                    Goldilocks3::copy_avx512(&bufferT_[buffTOffsetsStages[args[i_args]] + 2 * args[i_args + 1] + args[i_args + 2]], 2, 
                        &bufferT_[buffTOffsetsStages[args[i_args + 3]] + 2 * args[i_args + 4] + args[i_args + 5]], 2);
                    setStorePol(storePol, buffTOffsetsStages, args[i_args], args[i_args + 1], args[i_args + 2], FIELD_EXTENSION);
                    i_args += 6;
                    break;
                }
                case 46: {
                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: commit3
                    Goldilocks3::op_avx512(args[i_args], &bufferT_[buffTOffsetsStages[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &bufferT_[buffTOffsetsStages[args[i_args + 4]] + 2 * args[i_args + 5] + args[i_args + 6]], 2, 
                        &bufferT_[buffTOffsetsStages[args[i_args + 7]] + 2 * args[i_args + 8] + args[i_args + 9]], 2);
                    setStorePol(storePol, buffTOffsetsStages, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 10;
                    break;
                }
                case 47: {
                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: tmp3
                    Goldilocks3::op_avx512(args[i_args], &bufferT_[buffTOffsetsStages[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &bufferT_[buffTOffsetsStages[args[i_args + 4]] + 2 * args[i_args + 5] + args[i_args + 6]], 2, 
                        &(tmp3[args[i_args + 7]][0]), 1);
                    setStorePol(storePol, buffTOffsetsStages, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 8;
                    break;
                }
                case 48: {
                    // MULTIPLICATION WITH DEST: commit3 - SRC0: commit3 - SRC1: challenge
                    Goldilocks3::mul_avx512(&bufferT_[buffTOffsetsStages[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &bufferT_[buffTOffsetsStages[args[i_args + 4]] + 2 * args[i_args + 5] + args[i_args + 6]], 2, 
                        challenges[args[i_args + 7]], challenges_ops[args[i_args + 7]]);
                    setStorePol(storePol, buffTOffsetsStages, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 8;
                    break;
                }
                case 49: {
                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: challenge
                    Goldilocks3::op_avx512(args[i_args], &bufferT_[buffTOffsetsStages[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &bufferT_[buffTOffsetsStages[args[i_args + 4]] + 2 * args[i_args + 5] + args[i_args + 6]], 2, 
                        &(challenges[args[i_args + 7]][0]), 1);
                    setStorePol(storePol, buffTOffsetsStages, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 8;
                    break;
                }
                case 50: {
                    // COPY tmp3 to commit3
                    Goldilocks3::copy_avx512(&bufferT_[buffTOffsetsStages[args[i_args]] + 2 * args[i_args + 1] + args[i_args + 2]], 2, 
                        &(tmp3[args[i_args + 3]][0]), 1);
                    setStorePol(storePol, buffTOffsetsStages, args[i_args], args[i_args + 1], args[i_args + 2], FIELD_EXTENSION);
                    i_args += 4;
                    break;
                }
                case 51: {
                    // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: tmp3
                    Goldilocks3::op_avx512(args[i_args], &bufferT_[buffTOffsetsStages[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &(tmp3[args[i_args + 4]][0]), 1, 
                        &(tmp3[args[i_args + 5]][0]), 1);
                    setStorePol(storePol, buffTOffsetsStages, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 6;
                    break;
                }
                case 52: {
                    // MULTIPLICATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: challenge
                    Goldilocks3::mul_avx512(&bufferT_[buffTOffsetsStages[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &(tmp3[args[i_args + 4]][0]), 1, 
                        challenges[args[i_args + 5]], challenges_ops[args[i_args + 5]]);
                    setStorePol(storePol, buffTOffsetsStages, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 6;
                    break;
                }
                case 53: {
                    // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: challenge
                    Goldilocks3::op_avx512(args[i_args], &bufferT_[buffTOffsetsStages[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &(tmp3[args[i_args + 4]][0]), 1, 
                        &(challenges[args[i_args + 5]][0]), 1);
                    setStorePol(storePol, buffTOffsetsStages, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 6;
                    break;
                }
                case 54: {
                    // MULTIPLICATION WITH DEST: commit3 - SRC0: challenge - SRC1: challenge
                    Goldilocks3::mul_avx512(&bufferT_[buffTOffsetsStages[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &(challenges[args[i_args + 4]][0]), 1, 
                        challenges[args[i_args + 5]], challenges_ops[args[i_args + 5]]);
                    setStorePol(storePol, buffTOffsetsStages, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 6;
                    break;
                }
                case 55: {
                    // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: challenge
                    Goldilocks3::op_avx512(args[i_args], &bufferT_[buffTOffsetsStages[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2, 
                        &(challenges[args[i_args + 4]][0]), 1, 
                        &(challenges[args[i_args + 5]][0]), 1);
                    setStorePol(storePol, buffTOffsetsStages, args[i_args + 1], args[i_args + 2], args[i_args + 3], FIELD_EXTENSION);
                    i_args += 6;
                    break;
                }
                case 56: {
                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: commit1
                    Goldilocks3::op_31_avx512(args[i_args], &(tmp3[args[i_args + 1]][0]), 1, 
                        &bufferT_[buffTOffsetsStages[args[i_args + 2]] + 2 * args[i_args + 3] + args[i_args + 4]], 2, 
                        bufferT_[buffTOffsetsStages[args[i_args + 5]] + 2 * args[i_args + 6] + args[i_args + 7]]);
                    i_args += 8;
                    break;
                }
                case 57: {
                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: tmp1
                    Goldilocks3::op_31_avx512(args[i_args], &(tmp3[args[i_args + 1]][0]), 1, 
                        &bufferT_[buffTOffsetsStages[args[i_args + 2]] + 2 * args[i_args + 3] + args[i_args + 4]], 2, 
                        tmp1[args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 58: {
                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: public
                    Goldilocks3::op_31_avx512(args[i_args], &(tmp3[args[i_args + 1]][0]), 1, 
                        &bufferT_[buffTOffsetsStages[args[i_args + 2]] + 2 * args[i_args + 3] + args[i_args + 4]], 2, 
                        publics[args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 59: {
                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: x
                    Goldilocks::load_avx512(tmp1_1, x[i], x.offset());
                    Goldilocks3::op_31_avx512(args[i_args], &(tmp3[args[i_args + 1]][0]), 1, 
                        &bufferT_[buffTOffsetsStages[args[i_args + 2]] + 2 * args[i_args + 3] + args[i_args + 4]], 2, 
                        tmp1_1);
                    i_args += 5;
                    break;
                }
                case 60: {
                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: number
                    Goldilocks3::op_31_avx512(args[i_args], &(tmp3[args[i_args + 1]][0]), 1, 
                        &bufferT_[buffTOffsetsStages[args[i_args + 2]] + 2 * args[i_args + 3] + args[i_args + 4]], 2, 
                        numbers_[args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 61: {
                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: commit1
                    Goldilocks3::op_31_avx512(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], bufferT_[buffTOffsetsStages[args[i_args + 3]] + 2 * args[i_args + 4] + args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 62: {
                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: tmp1
                    Goldilocks3::op_31_avx512(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], tmp1[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 63: {
                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: public
                    Goldilocks3::op_31_avx512(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], publics[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 64: {
                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: x
                    Goldilocks::load_avx512(tmp1_1, x[i], x.offset());
                    Goldilocks3::op_31_avx512(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], tmp1_1);
                    i_args += 3;
                    break;
                }
                case 65: {
                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: number
                    Goldilocks3::op_31_avx512(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], numbers_[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 66: {
                    // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: commit1
                    Goldilocks3::op_31_avx512(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], bufferT_[buffTOffsetsStages[args[i_args + 3]] + 2 * args[i_args + 4] + args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 67: {
                    // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: tmp1
                    Goldilocks3::op_31_avx512(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], tmp1[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 68: {
                    // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: public
                    Goldilocks3::op_31_avx512(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], publics[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 69: {
                    // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: x
                    Goldilocks::load_avx512(tmp1_1, x[i], x.offset());
                    Goldilocks3::op_31_avx512(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], tmp1_1);
                    i_args += 3;
                    break;
                }
                case 70: {
                    // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: number
                    Goldilocks3::op_31_avx512(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], numbers_[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 71: {
                    // COPY commit3 to tmp3
                    Goldilocks3::copy_avx512(&(tmp3[args[i_args]][0]), 1, 
                        &bufferT_[buffTOffsetsStages[args[i_args + 1]] + 2 * args[i_args + 2] + args[i_args + 3]], 2);
                    i_args += 4;
                    break;
                }
                case 72: {
                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: commit3
                    Goldilocks3::op_avx512(args[i_args], &(tmp3[args[i_args + 1]][0]), 1, 
                        &bufferT_[buffTOffsetsStages[args[i_args + 2]] + 2 * args[i_args + 3] + args[i_args + 4]], 2, 
                        &bufferT_[buffTOffsetsStages[args[i_args + 5]] + 2 * args[i_args + 6] + args[i_args + 7]], 2);
                    i_args += 8;
                    break;
                }
                case 73: {
                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: tmp3
                    Goldilocks3::op_avx512(args[i_args], &(tmp3[args[i_args + 1]][0]), 1, 
                        &bufferT_[buffTOffsetsStages[args[i_args + 2]] + 2 * args[i_args + 3] + args[i_args + 4]], 2, 
                        &(tmp3[args[i_args + 5]][0]), 1);
                    i_args += 6;
                    break;
                }
                case 74: {
                    // MULTIPLICATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: challenge
                    Goldilocks3::mul_avx512(&(tmp3[args[i_args + 1]][0]), 1, 
                        &bufferT_[buffTOffsetsStages[args[i_args + 2]] + 2 * args[i_args + 3] + args[i_args + 4]], 2, 
                        challenges[args[i_args + 5]], challenges_ops[args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 75: {
                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: challenge
                    Goldilocks3::op_avx512(args[i_args], &(tmp3[args[i_args + 1]][0]), 1, 
                        &bufferT_[buffTOffsetsStages[args[i_args + 2]] + 2 * args[i_args + 3] + args[i_args + 4]], 2, 
                        &(challenges[args[i_args + 5]][0]), 1);
                    i_args += 6;
                    break;
                }
                case 76: {
                    // COPY tmp3 to tmp3
                    Goldilocks3::copy_avx512(tmp3[args[i_args]], tmp3[args[i_args + 1]]);
                    i_args += 2;
                    break;
                }
                case 77: {
                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: tmp3
                    Goldilocks3::op_avx512(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], tmp3[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 78: {
                    // MULTIPLICATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: challenge
                    Goldilocks3::mul_avx512(tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 79: {
                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: challenge
                    Goldilocks3::op_avx512(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], challenges[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 80: {
                    // MULTIPLICATION WITH DEST: tmp3 - SRC0: challenge - SRC1: challenge
                    Goldilocks3::mul_avx512(tmp3[args[i_args + 1]], challenges[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 81: {
                    // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: challenge
                    Goldilocks3::op_avx512(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], challenges[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 82: {
                    // COPY eval to tmp3
                    Goldilocks3::copy_avx512(tmp3[args[i_args]], evals[args[i_args + 1]]);
                    i_args += 2;
                    break;
                }
                case 83: {
                    // MULTIPLICATION WITH DEST: tmp3 - SRC0: eval - SRC1: challenge
                    Goldilocks3::mul_avx512(tmp3[args[i_args + 1]], evals[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 84: {
                    // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: eval
                    Goldilocks3::op_avx512(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], evals[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 85: {
                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: eval
                    Goldilocks3::op_avx512(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], evals[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 86: {
                    // OPERATION WITH DEST: tmp3 - SRC0: eval - SRC1: commit1
                    Goldilocks3::op_31_avx512(args[i_args], tmp3[args[i_args + 1]], evals[args[i_args + 2]], bufferT_[buffTOffsetsStages[args[i_args + 3]] + 2 * args[i_args + 4] + args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 87: {
                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: eval
                    Goldilocks3::op_avx512(args[i_args], &(tmp3[args[i_args + 1]][0]), 1, 
                        &bufferT_[buffTOffsetsStages[args[i_args + 2]] + 2 * args[i_args + 3] + args[i_args + 4]], 2, 
                        &(evals[args[i_args + 5]][0]), 1);
                    i_args += 6;
                    break;
                }
                case 88: {
                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: xDivXSubXi
                    Goldilocks3::load_avx512(tmp3_1, params.xDivXSubXi[i + args[i_args + 3]*domainSize], uint64_t(FIELD_EXTENSION));
                    Goldilocks3::op_avx512(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], tmp3_1);
                    i_args += 4;
                    break;
                }
                case 89: {
                    // OPERATION WITH DEST: q - SRC0: tmp3 - SRC1: Zi
                    Goldilocks::load_avx512(tmp1_1, params.zi[i], params.zi.offset());
                    Goldilocks3::op_31_avx512(2, tmp3_, tmp3[args[i_args]], tmp1_1);
                    Goldilocks3::store_avx512(&params.q_2ns[i*FIELD_EXTENSION], uint64_t(FIELD_EXTENSION), tmp3_);
                    i_args += 1;
                    break;
                }
                case 90: {
                    // OPERATION WITH DEST: f - SRC0: tmp3 - SRC1: tmp3
                    Goldilocks3::op_avx512(args[i_args], tmp3_, tmp3[args[i_args + 1]], tmp3[args[i_args + 2]]);
                    Goldilocks3::store_avx512(&params.f_2ns[i*FIELD_EXTENSION], uint64_t(FIELD_EXTENSION), tmp3_);
                    i_args += 3;
                    break;
                }
                    default: {
                        std::cout << " Wrong operation!" << std::endl;
                        exit(1);
                    }
                }
            }
            storePolinomials(starkInfo, params, bufferT_, storePol, i, nrowsPack, domainExtended);
            if (i_args != parserParams.nArgs) std::cout << " " << i_args << " - " << parserParams.nArgs << std::endl;
            assert(i_args == parserParams.nArgs);
        }
    }
};

#endif