#include "chelpers_steps.hpp"


class AllSteps : public CHelpersSteps {
public:
    vector<uint64_t> nColsStages;
    vector<uint64_t> nColsStagesAcc;
    vector<uint64_t> offsetsStages;

    inline void setBufferTInfo(StarkInfo& starkInfo, uint64_t stage) {
       bool domainExtended = stage <= 3 ? false : true;
       nColsStagesAcc.resize(10);
       nColsStages.resize(5);
       offsetsStages.resize(5);

       nColsStages[0] = starkInfo.nConstants;
       offsetsStages[0] = 0;

       for(uint64_t s = 1; s <= 3; ++s) {
           nColsStages[s] = starkInfo.mapSectionsN.section[string2section("cm" + to_string(s) + "_n")];
           if(domainExtended) {
               offsetsStages[s] = starkInfo.mapOffsets.section[string2section("cm" + to_string(s) + "_2ns")];
           } else {
               offsetsStages[s] = starkInfo.mapOffsets.section[string2section("cm" + to_string(s) + "_n")];
           }
       }
       if(domainExtended) {
           nColsStages[4] = starkInfo.mapSectionsN.section[eSection::cm4_2ns];
           offsetsStages[4] = starkInfo.mapOffsets.section[eSection::cm4_2ns];
       } else {
           nColsStages[4] = starkInfo.mapSectionsN.section[eSection::tmpExp_n];
           offsetsStages[4] = starkInfo.mapOffsets.section[eSection::tmpExp_n];
       }
       for(uint64_t o = 0; o < 2; ++o) {
           for(uint64_t s = 0; s < 5; ++s) {
               if(s == 0) {
                   if(o == 0) {
                       nColsStagesAcc[0] = 0;
                   } else {
                       nColsStagesAcc[5*o] = nColsStagesAcc[5*o - 1] + nColsStages[4];
                   }
               } else {
                   nColsStagesAcc[5*o + s] = nColsStagesAcc[5*o + (s - 1)] + nColsStages[(s - 1)];
               }
           }
       }
    }

    inline void storePolinomial(StarkInfo& starkInfo, Goldilocks::Element *pols, __m256i *bufferT, uint64_t row, uint64_t nrowsPack, bool domainExtended, uint64_t stage, uint64_t stagePos, uint64_t openingPointIndex, uint64_t dim) {
        uint64_t domainSize = domainExtended ? 1 << starkInfo.starkStruct.nBitsExt : 1 << starkInfo.starkStruct.nBits;
        uint64_t nextStride = domainExtended ?  1 << (starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits) : 1;
        std::vector<uint64_t> nextStrides = {0, nextStride};
        bool isTmpPol = !domainExtended && stage == 4;
        bool const needModule = row + nrowsPack + nextStride >= domainSize;
        __m256i *buffT = &bufferT[(nColsStagesAcc[5* openingPointIndex + stage] + stagePos)];
        if(needModule) {
            uint64_t offsetsDest[nrowsPack];
            uint64_t nextStrideOffset = row + nextStrides[openingPointIndex];
            if(isTmpPol) {
                uint64_t stepOffset = offsetsStages[stage] + stagePos * domainSize;
                for(uint64_t i = 0; i < nrowsPack; ++i) {
                    offsetsDest[i] = stepOffset + ((nextStrideOffset + i) % domainSize) * dim;
                }
                if(dim == 1) {
                    Goldilocks::store_avx(&pols[0], offsetsDest, buffT[0]);
                } else {
                    Goldilocks3::store_avx(&pols[0], offsetsDest, buffT);
                }
            } else {
                uint64_t stepOffset = offsetsStages[stage] + stagePos;
                for(uint64_t i = 0; i < nrowsPack; ++i) {
                    offsetsDest[i] = stepOffset + ((nextStrideOffset + i) % domainSize) * nColsStages[stage];
                }
                Goldilocks::store_avx(&pols[0], offsetsDest, buffT[0]);
            }
        } else {
            if(isTmpPol) {
                if(dim == 1) {
                        Goldilocks::store_avx(&pols[offsetsStages[stage] + stagePos * domainSize + (row + nextStrides[openingPointIndex])], uint64_t(1), buffT[0]);
                } else {
                        Goldilocks3::store_avx(&pols[offsetsStages[stage] + stagePos * domainSize + (row + nextStrides[openingPointIndex]) * FIELD_EXTENSION], uint64_t(FIELD_EXTENSION), buffT);
                }
            } else {
                Goldilocks::store_avx(&pols[offsetsStages[stage] + stagePos + (row + nextStrides[openingPointIndex]) * nColsStages[stage]], nColsStages[stage], buffT[0]);
            }
        }
    }

    inline void storePolinomials(StarkInfo &starkInfo, StepsParams &params, __m256i *bufferT_, vector<uint64_t> &storePol, uint64_t row, uint64_t nrowsPack, uint64_t domainExtended) {
        uint64_t nStages = 3;
        for(uint64_t s = 2; s <= nStages + 1; ++s) {
            for(uint64_t k = 0; k < nColsStages[s]; ++k) {
                for(uint64_t o = 0; o < 2; ++o) {
                    if(storePol[nColsStagesAcc[5*o + s] + k]) {
                        storePolinomial(starkInfo, params.pols, bufferT_, row, nrowsPack, domainExtended, s, k, o, storePol[nColsStagesAcc[5*o + s] + k]);
                    }
                }
            }
        }
    }

    inline void setStorePol(std::vector<uint64_t> &storePol, uint64_t stage, uint64_t stagePos, uint64_t dim) {
        if(stage == 4 || stage == 9) {
            storePol[nColsStagesAcc[stage] + stagePos] = dim;
        } else {
            if(dim == 1) {
                storePol[nColsStagesAcc[stage] + stagePos] = 1;
            } else {
                storePol[nColsStagesAcc[stage] + stagePos] = 1;
                storePol[nColsStagesAcc[stage] + stagePos + 1] = 1;
                storePol[nColsStagesAcc[stage] + stagePos + 2] = 1;
            }
        }
    }

    inline void loadPolinomials(StarkInfo &starkInfo, StepsParams &params, __m256i *bufferT_, uint64_t row, uint64_t stage, uint64_t nrowsPack, uint64_t domainExtended) {
        Goldilocks::Element bufferT[2*nrowsPack];
        ConstantPolsStarks *constPols = domainExtended ? params.pConstPols2ns : params.pConstPols;
        uint64_t domainSize = domainExtended ? 1 << starkInfo.starkStruct.nBitsExt : 1 << starkInfo.starkStruct.nBits;
        uint64_t nStages = 3;
        uint64_t nextStride = domainExtended ?  1 << (starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits) : 1;
        std::vector<uint64_t> nextStrides = {0, nextStride};
        for(uint64_t k = 0; k < nColsStages[0]; ++k) {
            for(uint64_t o = 0; o < 2; ++o) {
                for(uint64_t j = 0; j < nrowsPack; ++j) {
                    uint64_t l = (row + j + nextStrides[o]) % domainSize;
                    bufferT[nrowsPack*o + j] = ((Goldilocks::Element *)constPols->address())[l * nColsStages[0] + k];
                }
                Goldilocks::load_avx(bufferT_[nColsStagesAcc[5*o] + k], &bufferT[nrowsPack*o]);
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
                    Goldilocks::load_avx(bufferT_[nColsStagesAcc[5*o + s] + k], &bufferT[nrowsPack*o]);
                }
            }
        }
        if(stage == 5) {
           for(uint64_t k = 0; k < nColsStages[nStages + 1]; ++k) {
               for(uint64_t o = 0; o < 2; ++o) {
                   for(uint64_t j = 0; j < nrowsPack; ++j) {
                       uint64_t l = (row + j + nextStrides[o]) % domainSize;
                       bufferT[nrowsPack*o + j] = params.pols[offsetsStages[nStages + 1] + l * nColsStages[nStages + 1] + k];
                   }
                   Goldilocks::load_avx(bufferT_[nColsStagesAcc[5*o + nStages + 1] + k], &bufferT[nrowsPack*o]);
               }
           }
       }
    }

    void calculateExpressions(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams) {
        uint32_t nrowsPack =  4;
        bool domainExtended = parserParams.stage > 3 ? true : false;
        uint64_t domainSize = domainExtended ? 1 << starkInfo.starkStruct.nBitsExt : 1 << starkInfo.starkStruct.nBits;
        Polinomial &x = domainExtended ? params.x_2ns : params.x_n;
        uint8_t *ops = &parserArgs.ops[parserParams.opsOffset];
        uint16_t *args = &parserArgs.args[parserParams.argsOffset];
        uint64_t *numbers = &parserArgs.numbers[parserParams.numbersOffset];

        setBufferTInfo(starkInfo, parserParams.stage);
        uint64_t nCols = nColsStages[nColsStages.size() - 1] + nColsStagesAcc[nColsStagesAcc.size() - 1];

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
        for (uint64_t i = 0; i < domainSize; i+= nrowsPack) {
            uint64_t i_args = 0;

            std::vector<uint64_t> storePol(2*nCols, 0);

            __m256i bufferT_[2*nCols];

            __m256i tmp1[parserParams.nTemp1];
            __m256i tmp1_1;
    

            Goldilocks3::Element_avx tmp3[parserParams.nTemp3];
            Goldilocks3::Element_avx tmp3_;
            Goldilocks3::Element_avx tmp3_1;

            loadPolinomials(starkInfo, params, bufferT_, i, parserParams.stage, nrowsPack, domainExtended);

            for (uint64_t kk = 0; kk < parserParams.nOps; ++kk) {
                switch (ops[kk]) {
                case 0: {
                    // COPY commit1 to tmp1
                    Goldilocks::copy_avx(tmp1[args[i_args]], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]]);
                    i_args += 3;
                    break;
                }
                case 1: {
                    // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: commit1
                    Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], bufferT_[nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 2: {
                    // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: tmp1
                    Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], tmp1[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 3: {
                    // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: public
                    Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], publics[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 4: {
                    // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: number
                    Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], numbers_[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 5: {
                    // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: tmp1
                    Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], tmp1[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 6: {
                    // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: number
                    Goldilocks::op_avx(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], numbers_[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 7: {
                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: tmp3
                    Goldilocks3::op_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], tmp3[args[i_args + 5]]);
                    setStorePol(storePol, args[i_args + 1], args[i_args + 2], FIELD_EXTENSION);
                    i_args += 6;
                    break;
                }
                case 8: {
                    // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: tmp3
                    Goldilocks3::op_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], tmp3[args[i_args + 3]], tmp3[args[i_args + 4]]);
                    setStorePol(storePol, args[i_args + 1], args[i_args + 2], FIELD_EXTENSION);
                    i_args += 5;
                    break;
                }
                case 9: {
                    // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: challenge
                    Goldilocks3::op_avx(args[i_args], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], tmp3[args[i_args + 3]], challenges[args[i_args + 4]]);
                    setStorePol(storePol, args[i_args + 1], args[i_args + 2], FIELD_EXTENSION);
                    i_args += 5;
                    break;
                }
                case 10: {
                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: number
                    Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], numbers_[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 11: {
                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: commit1
                    Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 12: {
                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: tmp1
                    Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], tmp1[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 13: {
                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: x
                    Goldilocks::load_avx(tmp1_1, x[i], x.offset());
                    Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], tmp1_1);
                    i_args += 3;
                    break;
                }
                case 14: {
                    // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: commit1
                    Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 15: {
                    // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: tmp1
                    Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], tmp1[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 16: {
                    // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: x
                    Goldilocks::load_avx(tmp1_1, x[i], x.offset());
                    Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], tmp1_1);
                    i_args += 3;
                    break;
                }
                case 17: {
                    // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: number
                    Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], numbers_[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 18: {
                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: commit3
                    Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 19: {
                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: tmp3
                    Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], tmp3[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 20: {
                    // MULTIPLICATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: challenge
                    Goldilocks3::mul_avx(tmp3[args[i_args + 1]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], challenges[args[i_args + 4]], challenges_ops[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 21: {
                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: challenge
                    Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], challenges[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 22: {
                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: tmp3
                    Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], tmp3[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 23: {
                    // MULTIPLICATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: challenge
                    Goldilocks3::mul_avx(tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 24: {
                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: challenge
                    Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], challenges[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 25: {
                    // OPERATION WITH DEST: tmp3 - SRC0: eval - SRC1: commit1
                    Goldilocks3::op_31_avx(args[i_args], tmp3[args[i_args + 1]], evals[args[i_args + 2]], bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 26: {
                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: eval
                    Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], evals[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 27: {
                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: xDivXSubXi
                    Goldilocks3::load_avx(tmp3_1, params.xDivXSubXi[i + args[i_args + 3]*domainSize], uint64_t(FIELD_EXTENSION));
                    Goldilocks3::op_avx(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], tmp3_1);
                    i_args += 4;
                    break;
                }
                case 28: {
                    // OPERATION WITH DEST: q - SRC0: tmp3 - SRC1: Zi
                    Goldilocks::load_avx(tmp1_1, params.zi[i], params.zi.offset());
                    Goldilocks3::op_31_avx(2, tmp3_, tmp3[args[i_args]], tmp1_1);
                    Goldilocks3::store_avx(&params.q_2ns[i*FIELD_EXTENSION], uint64_t(FIELD_EXTENSION), tmp3_);
                    i_args += 1;
                    break;
                }
                case 29: {
                    // OPERATION WITH DEST: f - SRC0: tmp3 - SRC1: tmp3
                    Goldilocks3::op_avx(args[i_args], tmp3_, tmp3[args[i_args + 1]], tmp3[args[i_args + 2]]);
                    Goldilocks3::store_avx(&params.f_2ns[i*FIELD_EXTENSION], uint64_t(FIELD_EXTENSION), tmp3_);
                    i_args += 3;
                    break;
                }
                    default: {
                        std::cout << " Wrong operation!" << std::endl;
                        exit(1);
                    }
                }
            }
            if(!domainExtended) storePolinomials(starkInfo, params, bufferT_, storePol, i, nrowsPack, domainExtended);
            if (i_args != parserParams.nArgs) std::cout << " " << i_args << " - " << parserParams.nArgs << std::endl;
            assert(i_args == parserParams.nArgs);
        }
    }
};