#ifndef CHELPERS_STEPS_AVX512_HPP
#define CHELPERS_STEPS_AVX512_HPP
#include "chelpers.hpp"
#include "chelpers_steps.hpp"
#include "steps.hpp"

class CHelpersStepsAvx512 : public CHelpersSteps {
public:
    uint64_t nrowsPack = 8;
    uint64_t nCols;
    vector<uint64_t> nColsStages;
    vector<uint64_t> nColsStagesAcc;
    vector<uint64_t> offsetsStages;

    inline virtual void setBufferTInfo(StarkInfo& starkInfo, uint64_t stage) {
        bool domainExtended = stage <= 3 ? false : true;
        nColsStagesAcc.resize(10 + 2);
        nColsStages.resize(10 + 2);
        offsetsStages.resize(10 + 2);

        nColsStages[0] = starkInfo.nConstants + 2;
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
        nColsStagesAcc[10] = nColsStagesAcc[9] + nColsStages[4]; // Polinomials f & q
        if(stage == 4) {
            offsetsStages[10] = starkInfo.mapOffsets.section[eSection::q_2ns];
            nColsStages[10] = starkInfo.qDim;
        } else if(stage == 5) {
            offsetsStages[10] = starkInfo.mapOffsets.section[eSection::f_2ns];
            nColsStages[10] = 3;
        }
        nColsStagesAcc[11] = nColsStagesAcc[10] + nColsStages[10]; // xDivXSubXi
        nCols = nColsStagesAcc[11] + 6; // 3 for xDivXSubXi and 3 for xDivXSubWxi
    }

    inline virtual void storePolinomials(StarkInfo &starkInfo, StepsParams &params, __m512i *bufferT_, uint8_t* storePol, uint64_t row, uint64_t nrowsPack, uint64_t domainExtended) {
        if(domainExtended) {
            // Store either polinomial f or polinomial q
            for(uint64_t k = 0; k < nColsStages[10]; ++k) {
                __m512i *buffT = &bufferT_[(nColsStagesAcc[10] + k)];
                Goldilocks::store_avx512(&params.pols[offsetsStages[10] + k + row * nColsStages[10]], nColsStages[10], buffT[0]);
            }
        } else {
            uint64_t nStages = 3;
            uint64_t domainSize = domainExtended ? 1 << starkInfo.starkStruct.nBitsExt : 1 << starkInfo.starkStruct.nBits;
            for(uint64_t s = 2; s <= nStages + 1; ++s) {
                bool isTmpPol = !domainExtended && s == 4;
                for(uint64_t k = 0; k < nColsStages[s]; ++k) {
                    uint64_t dim = storePol[nColsStagesAcc[s] + k];
                    if(storePol[nColsStagesAcc[s] + k]) {
                        __m512i *buffT = &bufferT_[(nColsStagesAcc[s] + k)];
                        if(isTmpPol) {
                            for(uint64_t i = 0; i < dim; ++i) {
                                Goldilocks::store_avx512(&params.pols[offsetsStages[s] + k * domainSize + row * dim + i], uint64_t(dim), buffT[i]);
                            }
                        } else {
                            Goldilocks::store_avx512(&params.pols[offsetsStages[s] + k + row * nColsStages[s]], nColsStages[s], buffT[0]);
                        }
                    }
                }
            }
        }
    }

    inline virtual void loadPolinomials(StarkInfo &starkInfo, StepsParams &params, __m512i *bufferT_, uint64_t row, uint64_t stage, uint64_t nrowsPack, uint64_t domainExtended) {
        Goldilocks::Element bufferT[2*nrowsPack];
        ConstantPolsStarks *constPols = domainExtended ? params.pConstPols2ns : params.pConstPols;
        Polinomial &x = domainExtended ? params.x_2ns : params.x_n;
        uint64_t domainSize = domainExtended ? 1 << starkInfo.starkStruct.nBitsExt : 1 << starkInfo.starkStruct.nBits;
        uint64_t nStages = 3;
        uint64_t nextStride = domainExtended ?  1 << (starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits) : 1;
        std::vector<uint64_t> nextStrides = {0, nextStride};
        for(uint64_t k = 0; k < starkInfo.nConstants; ++k) {
            for(uint64_t o = 0; o < 2; ++o) {
                for(uint64_t j = 0; j < nrowsPack; ++j) {
                    uint64_t l = (row + j + nextStrides[o]) % domainSize;
                    bufferT[nrowsPack*o + j] = ((Goldilocks::Element *)constPols->address())[l * starkInfo.nConstants + k];
                }
                Goldilocks::load_avx512(bufferT_[nColsStagesAcc[5*o] + k], &bufferT[nrowsPack*o]);
            }
        }

        // Load x and Zi
        for(uint64_t j = 0; j < nrowsPack; ++j) {
            bufferT[j] = x[row + j][0];
        }
        Goldilocks::load_avx512(bufferT_[starkInfo.nConstants], &bufferT[0]);
        for(uint64_t j = 0; j < nrowsPack; ++j) {
            bufferT[j] = params.zi[row + j][0];
        }

        Goldilocks::load_avx512(bufferT_[starkInfo.nConstants + 1], &bufferT[0]);

        for(uint64_t s = 1; s <= nStages; ++s) {
            if(stage < s) break;
            for(uint64_t k = 0; k < nColsStages[s]; ++k) {
                for(uint64_t o = 0; o < 2; ++o) {
                    for(uint64_t j = 0; j < nrowsPack; ++j) {
                        uint64_t l = (row + j + nextStrides[o]) % domainSize;
                        bufferT[nrowsPack*o + j] = params.pols[offsetsStages[s] + l * nColsStages[s] + k];
                    }
                    Goldilocks::load_avx512(bufferT_[nColsStagesAcc[5*o + s] + k], &bufferT[nrowsPack*o]);
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
                   Goldilocks::load_avx512(bufferT_[nColsStagesAcc[5*o + nStages + 1] + k], &bufferT[nrowsPack*o]);
               }
           }

           // Load xDivXSubXi & xDivXSubWXi
           for(uint64_t d = 0; d < 2; ++d) {
               for(uint64_t i = 0; i < FIELD_EXTENSION; ++i) {
                   for(uint64_t j = 0; j < nrowsPack; ++j) {
                       bufferT[j] = params.xDivXSubXi[d*domainSize + row + j][i];
                   }
                   Goldilocks::load_avx512(bufferT_[nColsStagesAcc[11] + FIELD_EXTENSION*d + i], &bufferT[0]);
               }
           }
       }
    }

    virtual void calculateExpressions(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams) {
        assert(nrowsPack == 8);
        bool domainExtended = parserParams.stage > 3 ? true : false;
        uint64_t domainSize = domainExtended ? 1 << starkInfo.starkStruct.nBitsExt : 1 << starkInfo.starkStruct.nBits;
        uint8_t *ops = &parserArgs.ops[parserParams.opsOffset];
        uint16_t *args = &parserArgs.args[parserParams.argsOffset];
        uint64_t *numbers = &parserArgs.numbers[parserParams.numbersOffset];
        uint8_t *storePol = &parserArgs.storePols[parserParams.storePolsOffset];

        setBufferTInfo(starkInfo, parserParams.stage);
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

            __m512i bufferT_[2*nCols];
            __m512i tmp1[parserParams.nTemp1];
            Goldilocks3::Element_avx512 tmp3[parserParams.nTemp3];

            loadPolinomials(starkInfo, params, bufferT_, i, parserParams.stage, nrowsPack, domainExtended);

            for (uint64_t kk = 0; kk < parserParams.nOps; ++kk) {
                switch (ops[kk]) {
                case 0: {
                    // COPY commit1 to commit1
                    Goldilocks::copy_avx512(bufferT_[nColsStagesAcc[args[i_args]] + args[i_args + 1]], bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 1: {
                    // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: commit1
                    Goldilocks::op_avx512(args[i_args], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], bufferT_[nColsStagesAcc[args[i_args + 5]] + args[i_args + 6]]);
                    i_args += 7;
                    break;
                }
                case 2: {
                    // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: tmp1
                    Goldilocks::op_avx512(args[i_args], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], tmp1[args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 3: {
                    // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: public
                    Goldilocks::op_avx512(args[i_args], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], publics[args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 4: {
                    // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: number
                    Goldilocks::op_avx512(args[i_args], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], numbers_[args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 5: {
                    // COPY tmp1 to commit1
                    Goldilocks::copy_avx512(bufferT_[nColsStagesAcc[args[i_args]] + args[i_args + 1]], tmp1[args[i_args + 2]]);
                    i_args += 3;
                    break;
                }
                case 6: {
                    // OPERATION WITH DEST: commit1 - SRC0: tmp1 - SRC1: tmp1
                    Goldilocks::op_avx512(args[i_args], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], tmp1[args[i_args + 3]], tmp1[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 7: {
                    // OPERATION WITH DEST: commit1 - SRC0: tmp1 - SRC1: public
                    Goldilocks::op_avx512(args[i_args], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], tmp1[args[i_args + 3]], publics[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 8: {
                    // OPERATION WITH DEST: commit1 - SRC0: tmp1 - SRC1: number
                    Goldilocks::op_avx512(args[i_args], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], tmp1[args[i_args + 3]], numbers_[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 9: {
                    // COPY public to commit1
                    Goldilocks::copy_avx512(bufferT_[nColsStagesAcc[args[i_args]] + args[i_args + 1]], publics[args[i_args + 2]]);
                    i_args += 3;
                    break;
                }
                case 10: {
                    // OPERATION WITH DEST: commit1 - SRC0: public - SRC1: public
                    Goldilocks::op_avx512(args[i_args], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], publics[args[i_args + 3]], publics[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 11: {
                    // OPERATION WITH DEST: commit1 - SRC0: public - SRC1: number
                    Goldilocks::op_avx512(args[i_args], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], publics[args[i_args + 3]], numbers_[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 12: {
                    // COPY number to commit1
                    Goldilocks::copy_avx512(bufferT_[nColsStagesAcc[args[i_args]] + args[i_args + 1]], numbers_[args[i_args + 2]]);
                    i_args += 3;
                    break;
                }
                case 13: {
                    // OPERATION WITH DEST: commit1 - SRC0: number - SRC1: number
                    Goldilocks::op_avx512(args[i_args], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], numbers_[args[i_args + 3]], numbers_[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 14: {
                    // COPY commit1 to tmp1
                    Goldilocks::copy_avx512(tmp1[args[i_args]], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]]);
                    i_args += 3;
                    break;
                }
                case 15: {
                    // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: commit1
                    Goldilocks::op_avx512(args[i_args], tmp1[args[i_args + 1]], bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], bufferT_[nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 16: {
                    // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: tmp1
                    Goldilocks::op_avx512(args[i_args], tmp1[args[i_args + 1]], bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], tmp1[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 17: {
                    // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: public
                    Goldilocks::op_avx512(args[i_args], tmp1[args[i_args + 1]], bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], publics[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 18: {
                    // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: number
                    Goldilocks::op_avx512(args[i_args], tmp1[args[i_args + 1]], bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], numbers_[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 19: {
                    // COPY tmp1 to tmp1
                    Goldilocks::copy_avx512(tmp1[args[i_args]], tmp1[args[i_args + 1]]);
                    i_args += 2;
                    break;
                }
                case 20: {
                    // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: tmp1
                    Goldilocks::op_avx512(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], tmp1[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 21: {
                    // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: public
                    Goldilocks::op_avx512(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], publics[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 22: {
                    // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: number
                    Goldilocks::op_avx512(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], numbers_[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 23: {
                    // COPY public to tmp1
                    Goldilocks::copy_avx512(tmp1[args[i_args]], publics[args[i_args + 1]]);
                    i_args += 2;
                    break;
                }
                case 24: {
                    // OPERATION WITH DEST: tmp1 - SRC0: public - SRC1: public
                    Goldilocks::op_avx512(args[i_args], tmp1[args[i_args + 1]], publics[args[i_args + 2]], publics[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 25: {
                    // OPERATION WITH DEST: tmp1 - SRC0: public - SRC1: number
                    Goldilocks::op_avx512(args[i_args], tmp1[args[i_args + 1]], publics[args[i_args + 2]], numbers_[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 26: {
                    // COPY number to tmp1
                    Goldilocks::copy_avx512(tmp1[args[i_args]], numbers_[args[i_args + 1]]);
                    i_args += 2;
                    break;
                }
                case 27: {
                    // OPERATION WITH DEST: tmp1 - SRC0: number - SRC1: number
                    Goldilocks::op_avx512(args[i_args], tmp1[args[i_args + 1]], numbers_[args[i_args + 2]], numbers_[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 28: {
                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: commit1
                    Goldilocks3::op_31_avx512(args[i_args], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], bufferT_[nColsStagesAcc[args[i_args + 5]] + args[i_args + 6]]);
                    i_args += 7;
                    break;
                }
                case 29: {
                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: tmp1
                    Goldilocks3::op_31_avx512(args[i_args], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], tmp1[args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 30: {
                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: public
                    Goldilocks3::op_31_avx512(args[i_args], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], publics[args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 31: {
                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: number
                    Goldilocks3::op_31_avx512(args[i_args], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], numbers_[args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 32: {
                    // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: commit1
                    Goldilocks3::op_31_avx512(args[i_args], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], tmp3[args[i_args + 3]], bufferT_[nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 33: {
                    // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: tmp1
                    Goldilocks3::op_31_avx512(args[i_args], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], tmp3[args[i_args + 3]], tmp1[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 34: {
                    // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: public
                    Goldilocks3::op_31_avx512(args[i_args], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], tmp3[args[i_args + 3]], publics[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 35: {
                    // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: number
                    Goldilocks3::op_31_avx512(args[i_args], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], tmp3[args[i_args + 3]], numbers_[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 36: {
                    // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: commit1
                    Goldilocks3::op_31_avx512(args[i_args], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], challenges[args[i_args + 3]], bufferT_[nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 37: {
                    // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: tmp1
                    Goldilocks3::op_31_avx512(args[i_args], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], challenges[args[i_args + 3]], tmp1[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 38: {
                    // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: public
                    Goldilocks3::op_31_avx512(args[i_args], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], challenges[args[i_args + 3]], publics[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 39: {
                    // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: number
                    Goldilocks3::op_31_avx512(args[i_args], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], challenges[args[i_args + 3]], numbers_[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 40: {
                    // COPY commit3 to commit3
                    Goldilocks3::copy_avx512((Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args]] + args[i_args + 1]], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 41: {
                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: commit3
                    Goldilocks3::op_avx512(args[i_args], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 5]] + args[i_args + 6]]);
                    i_args += 7;
                    break;
                }
                case 42: {
                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: tmp3
                    Goldilocks3::op_avx512(args[i_args], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], tmp3[args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 43: {
                    // MULTIPLICATION WITH DEST: commit3 - SRC0: commit3 - SRC1: challenge
                    Goldilocks3::mul_avx512((Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], challenges[args[i_args + 5]], challenges_ops[args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 44: {
                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: challenge
                    Goldilocks3::op_avx512(args[i_args], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]], challenges[args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 45: {
                    // COPY tmp3 to commit3
                    Goldilocks3::copy_avx512((Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args]] + args[i_args + 1]], tmp3[args[i_args + 2]]);
                    i_args += 3;
                    break;
                }
                case 46: {
                    // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: tmp3
                    Goldilocks3::op_avx512(args[i_args], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], tmp3[args[i_args + 3]], tmp3[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 47: {
                    // MULTIPLICATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: challenge
                    Goldilocks3::mul_avx512((Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], tmp3[args[i_args + 3]], challenges[args[i_args + 4]], challenges_ops[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 48: {
                    // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: challenge
                    Goldilocks3::op_avx512(args[i_args], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], tmp3[args[i_args + 3]], challenges[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 49: {
                    // MULTIPLICATION WITH DEST: commit3 - SRC0: challenge - SRC1: challenge
                    Goldilocks3::mul_avx512((Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], challenges[args[i_args + 3]], challenges[args[i_args + 4]], challenges_ops[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 50: {
                    // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: challenge
                    Goldilocks3::op_avx512(args[i_args], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]], challenges[args[i_args + 3]], challenges[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 51: {
                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: commit1
                    Goldilocks3::op_31_avx512(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], bufferT_[nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 52: {
                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: tmp1
                    Goldilocks3::op_31_avx512(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], tmp1[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 53: {
                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: public
                    Goldilocks3::op_31_avx512(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], publics[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 54: {
                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: number
                    Goldilocks3::op_31_avx512(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], numbers_[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 55: {
                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: commit1
                    Goldilocks3::op_31_avx512(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 56: {
                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: tmp1
                    Goldilocks3::op_31_avx512(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], tmp1[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 57: {
                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: public
                    Goldilocks3::op_31_avx512(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], publics[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 58: {
                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: number
                    Goldilocks3::op_31_avx512(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], numbers_[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 59: {
                    // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: commit1
                    Goldilocks3::op_31_avx512(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 60: {
                    // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: tmp1
                    Goldilocks3::op_31_avx512(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], tmp1[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 61: {
                    // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: public
                    Goldilocks3::op_31_avx512(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], publics[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 62: {
                    // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: number
                    Goldilocks3::op_31_avx512(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], numbers_[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 63: {
                    // COPY commit3 to tmp3
                    Goldilocks3::copy_avx512(tmp3[args[i_args]], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]]);
                    i_args += 3;
                    break;
                }
                case 64: {
                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: commit3
                    Goldilocks3::op_avx512(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]]);
                    i_args += 6;
                    break;
                }
                case 65: {
                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: tmp3
                    Goldilocks3::op_avx512(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], tmp3[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 66: {
                    // MULTIPLICATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: challenge
                    Goldilocks3::mul_avx512(tmp3[args[i_args + 1]], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], challenges[args[i_args + 4]], challenges_ops[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 67: {
                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: challenge
                    Goldilocks3::op_avx512(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], challenges[args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 68: {
                    // COPY tmp3 to tmp3
                    Goldilocks3::copy_avx512(tmp3[args[i_args]], tmp3[args[i_args + 1]]);
                    i_args += 2;
                    break;
                }
                case 69: {
                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: tmp3
                    Goldilocks3::op_avx512(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], tmp3[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 70: {
                    // MULTIPLICATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: challenge
                    Goldilocks3::mul_avx512(tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 71: {
                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: challenge
                    Goldilocks3::op_avx512(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], challenges[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 72: {
                    // MULTIPLICATION WITH DEST: tmp3 - SRC0: challenge - SRC1: challenge
                    Goldilocks3::mul_avx512(tmp3[args[i_args + 1]], challenges[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 73: {
                    // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: challenge
                    Goldilocks3::op_avx512(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], challenges[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 74: {
                    // COPY eval to tmp3
                    Goldilocks3::copy_avx512(tmp3[args[i_args]], evals[args[i_args + 1]]);
                    i_args += 2;
                    break;
                }
                case 75: {
                    // MULTIPLICATION WITH DEST: tmp3 - SRC0: eval - SRC1: challenge
                    Goldilocks3::mul_avx512(tmp3[args[i_args + 1]], evals[args[i_args + 2]], challenges[args[i_args + 3]], challenges_ops[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 76: {
                    // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: eval
                    Goldilocks3::op_avx512(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], evals[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 77: {
                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: eval
                    Goldilocks3::op_avx512(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], evals[args[i_args + 3]]);
                    i_args += 4;
                    break;
                }
                case 78: {
                    // OPERATION WITH DEST: tmp3 - SRC0: eval - SRC1: commit1
                    Goldilocks3::op_31_avx512(args[i_args], tmp3[args[i_args + 1]], evals[args[i_args + 2]], bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]]);
                    i_args += 5;
                    break;
                }
                case 79: {
                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: eval
                    Goldilocks3::op_avx512(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], evals[args[i_args + 4]]);
                    i_args += 5;
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