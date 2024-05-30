#ifndef CHELPERS_STEPS_PACK_HPP
#define CHELPERS_STEPS_PACK_HPP
#include "chelpers.hpp"
#include "chelpers_steps.hpp"
#include "steps.hpp"

class CHelpersStepsPack : public CHelpersSteps {
public:
    vector<uint64_t> nColsStages;
    vector<uint64_t> nColsStagesAcc;
    vector<uint64_t> offsetsStages;

    inline virtual void setBufferTInfo(StarkInfo& starkInfo, uint64_t stage) {
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

    inline virtual void storePolinomial(StarkInfo& starkInfo, Goldilocks::Element *pols, Goldilocks::Element *bufferT, uint64_t row, uint64_t nrowsPack, bool domainExtended, uint64_t stage, uint64_t stagePos, uint64_t openingPointIndex, uint64_t dim) {
        uint64_t domainSize = domainExtended ? 1 << starkInfo.starkStruct.nBitsExt : 1 << starkInfo.starkStruct.nBits;
        uint64_t nextStride = domainExtended ?  1 << (starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits) : 1;
        std::vector<uint64_t> nextStrides = {0, nextStride};
        bool isTmpPol = !domainExtended && stage == 4;
        bool const needModule = row + nrowsPack + nextStride >= domainSize;
        Goldilocks::Element *buffT = &bufferT[(nColsStagesAcc[5* openingPointIndex + stage] + stagePos)* nrowsPack];
        if(needModule) {
            uint64_t offsetsDest[nrowsPack];
            uint64_t nextStrideOffset = row + nextStrides[openingPointIndex];
            if(isTmpPol) {
                uint64_t stepOffset = offsetsStages[stage] + stagePos * domainSize;
                for(uint64_t i = 0; i < nrowsPack; ++i) {
                    offsetsDest[i] = stepOffset + ((nextStrideOffset + i) % domainSize) * dim;
                }
                for(uint64_t i = 0; i < dim; ++i) {
                    Goldilocks::copy_pack(nrowsPack, &pols[i], offsetsDest, &buffT[i*nrowsPack]);
                }
            } else {
                uint64_t stepOffset = offsetsStages[stage] + stagePos;
                for(uint64_t i = 0; i < nrowsPack; ++i) {
                    offsetsDest[i] = stepOffset + ((nextStrideOffset + i) % domainSize) * nColsStages[stage];
                }
                Goldilocks::copy_pack(nrowsPack, &pols[0], offsetsDest, buffT);
            }
        } else {
            if(isTmpPol) {
                for(uint64_t i = 0; i < dim; ++i) {
                    Goldilocks::copy_pack(nrowsPack, &pols[offsetsStages[stage] + stagePos * domainSize + (row + nextStrides[openingPointIndex]) * FIELD_EXTENSION + i], uint64_t(dim), &buffT[i*nrowsPack]);
                }
            } else {
                Goldilocks::copy_pack(nrowsPack, &pols[offsetsStages[stage] + stagePos + (row + nextStrides[openingPointIndex]) * nColsStages[stage]], nColsStages[stage], buffT);
            }
        }
    }

    inline virtual void storePolinomials(StarkInfo &starkInfo, StepsParams &params, Goldilocks::Element *bufferT_, vector<uint64_t> &storePol, uint64_t row, uint64_t nrowsPack, uint64_t domainExtended) {
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

    inline virtual void setStorePol(std::vector<uint64_t> &storePol, uint64_t stage, uint64_t stagePos, uint64_t dim) {
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

    inline virtual void loadPolinomials(StarkInfo &starkInfo, StepsParams &params, Goldilocks::Element *bufferT_, uint64_t row, uint64_t stage, uint64_t nrowsPack, uint64_t domainExtended) {
        ConstantPolsStarks *constPols = domainExtended ? params.pConstPols2ns : params.pConstPols;
        uint64_t domainSize = domainExtended ? 1 << starkInfo.starkStruct.nBitsExt : 1 << starkInfo.starkStruct.nBits;
        uint64_t nStages = 3;
        uint64_t nextStride = domainExtended ?  1 << (starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits) : 1;
        std::vector<uint64_t> nextStrides = {0, nextStride};
        for(uint64_t k = 0; k < nColsStages[0]; ++k) {
            for(uint64_t o = 0; o < 2; ++o) {
                for(uint64_t j = 0; j < nrowsPack; ++j) {
                    uint64_t l = (row + j + nextStrides[o]) % domainSize;
                    bufferT_[(nColsStagesAcc[5*o] + k)*nrowsPack + j] = ((Goldilocks::Element *)constPols->address())[l * nColsStages[0] + k];
                }
            }
        }
        for(uint64_t s = 1; s <= nStages; ++s) {
            for(uint64_t k = 0; k < nColsStages[s]; ++k) {
                for(uint64_t o = 0; o < 2; ++o) {
                    for(uint64_t j = 0; j < nrowsPack; ++j) {
                        uint64_t l = (row + j + nextStrides[o]) % domainSize;
                        bufferT_[(nColsStagesAcc[5*o + s] + k)*nrowsPack + j] = params.pols[offsetsStages[s] + l * nColsStages[s] + k];
                    }
                }
            }
        }
        if(stage == 5) {
            for(uint64_t k = 0; k < nColsStages[nStages + 1]; ++k) {
                for(uint64_t o = 0; o < 2; ++o) {
                    for(uint64_t j = 0; j < nrowsPack; ++j) {
                        uint64_t l = (row + j + nextStrides[o]) % domainSize;
                        bufferT_[(nColsStagesAcc[5*o + nStages + 1] + k)*nrowsPack + j] = params.pols[offsetsStages[nStages + 1] + l * nColsStages[nStages + 1] + k];
                    }
                }
            }
        }
    }

    virtual void calculateExpressions(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams) {
        uint32_t nrowsPack =  4;
        bool domainExtended = parserParams.stage > 3 ? true : false;
        uint64_t domainSize = domainExtended ? 1 << starkInfo.starkStruct.nBitsExt : 1 << starkInfo.starkStruct.nBits;
        Polinomial &x = domainExtended ? params.x_2ns : params.x_n;
        uint8_t *ops = &parserArgs.ops[parserParams.opsOffset];
        uint16_t *args = &parserArgs.args[parserParams.argsOffset];
        uint64_t *numbers = &parserArgs.numbers[parserParams.numbersOffset];

        setBufferTInfo(starkInfo, parserParams.stage);
        uint64_t nCols = nColsStages[nColsStages.size() - 1] + nColsStagesAcc[nColsStagesAcc.size() - 1];

        Goldilocks::Element challenges[params.challenges.degree()*FIELD_EXTENSION*nrowsPack];
        Goldilocks::Element challenges_ops[params.challenges.degree()*FIELD_EXTENSION*nrowsPack];
        for(uint64_t i = 0; i < params.challenges.degree(); ++i) {
            for(uint64_t j = 0; j < nrowsPack; ++j) {
                challenges[(i*FIELD_EXTENSION)*nrowsPack + j] = params.challenges[i][0];
                challenges[(i*FIELD_EXTENSION + 1)*nrowsPack + j] = params.challenges[i][1];
                challenges[(i*FIELD_EXTENSION + 2)*nrowsPack + j] = params.challenges[i][2];
                challenges_ops[(i*FIELD_EXTENSION)*nrowsPack + j] = params.challenges[i][0] + params.challenges[i][1];
                challenges_ops[(i*FIELD_EXTENSION + 1)*nrowsPack + j] = params.challenges[i][0] + params.challenges[i][2];
                challenges_ops[(i*FIELD_EXTENSION + 2)*nrowsPack + j] = params.challenges[i][1] + params.challenges[i][2];
            }
        }

        Goldilocks::Element numbers_[parserParams.nNumbers*nrowsPack];
        for(uint64_t i = 0; i < parserParams.nNumbers; ++i) {
            for(uint64_t j = 0; j < nrowsPack; ++j) {
                numbers_[i*nrowsPack + j] = Goldilocks::fromU64(numbers[i]);
            }
        }

        Goldilocks::Element publics[starkInfo.nPublics*nrowsPack];
        for(uint64_t i = 0; i < starkInfo.nPublics; ++i) {
            for(uint64_t j = 0; j < nrowsPack; ++j) {
                publics[i*nrowsPack + j] = params.publicInputs[i];
            }
        }

        Goldilocks::Element evals[params.evals.degree()*FIELD_EXTENSION*nrowsPack];
        for(uint64_t i = 0; i < params.evals.degree(); ++i) {
            for(uint64_t j = 0; j < nrowsPack; ++j) {
                evals[(i*FIELD_EXTENSION)*nrowsPack + j] = params.evals[i][0];
                evals[(i*FIELD_EXTENSION + 1)*nrowsPack + j] = params.evals[i][1];
                evals[(i*FIELD_EXTENSION + 2)*nrowsPack + j] = params.evals[i][2];
            }
        }

    #pragma omp parallel for
        for (uint64_t i = 0; i < domainSize; i+= nrowsPack) {
            uint64_t i_args = 0;

            std::vector<uint64_t> storePol(2*nCols, 0);

            Goldilocks::Element bufferT_[2*nCols*nrowsPack];

            Goldilocks::Element tmp1[parserParams.nTemp1*nrowsPack];
            Goldilocks::Element tmp3[parserParams.nTemp3*nrowsPack*FIELD_EXTENSION];
            Goldilocks::Element tmp3_[nrowsPack*FIELD_EXTENSION];
            Goldilocks::Element tmp3_1[nrowsPack*FIELD_EXTENSION];
            loadPolinomials(starkInfo, params, bufferT_, i, parserParams.stage, nrowsPack, domainExtended);

            for (uint64_t kk = 0; kk < parserParams.nOps; ++kk) {
                switch (ops[kk]) {
                case 0: {
                    // COPY commit1 to commit1
                    Goldilocks::copy_pack(nrowsPack, &bufferT_[(nColsStagesAcc[args[i_args]] + args[i_args + 1]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack]);
                    setStorePol(storePol, args[i_args], args[i_args + 1], 1);
                    i_args += 4;
                    break;
                }
                case 1: {
                    // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: commit1
                    Goldilocks::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 5]] + args[i_args + 6]) * nrowsPack]);
                    setStorePol(storePol, args[i_args + 1], args[i_args + 2], 1);
                    i_args += 7;
                    break;
                }
                case 2: {
                    // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: tmp1
                    Goldilocks::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack], &tmp1[args[i_args + 5] * nrowsPack]);
                    setStorePol(storePol, args[i_args + 1], args[i_args + 2], 1);
                    i_args += 6;
                    break;
                }
                case 3: {
                    // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: public
                    Goldilocks::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack], &publics[args[i_args + 5] * nrowsPack]);
                    setStorePol(storePol, args[i_args + 1], args[i_args + 2], 1);
                    i_args += 6;
                    break;
                }
                case 4: {
                    // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: number
                    Goldilocks::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack], &numbers_[args[i_args + 5]*nrowsPack]);
                    setStorePol(storePol, args[i_args + 1], args[i_args + 2], 1);
                    i_args += 6;
                    break;
                }
                case 5: {
                    // COPY tmp1 to commit1
                    Goldilocks::copy_pack(nrowsPack, &bufferT_[(nColsStagesAcc[args[i_args]] + args[i_args + 1]) * nrowsPack], &tmp1[args[i_args + 2] * nrowsPack]);
                    setStorePol(storePol, args[i_args], args[i_args + 1], 1);
                    i_args += 3;
                    break;
                }
                case 6: {
                    // OPERATION WITH DEST: commit1 - SRC0: tmp1 - SRC1: tmp1
                    Goldilocks::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &tmp1[args[i_args + 3] * nrowsPack], &tmp1[args[i_args + 4] * nrowsPack]);
                    setStorePol(storePol, args[i_args + 1], args[i_args + 2], 1);
                    i_args += 5;
                    break;
                }
                case 7: {
                    // OPERATION WITH DEST: commit1 - SRC0: tmp1 - SRC1: public
                    Goldilocks::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &tmp1[args[i_args + 3] * nrowsPack], &publics[args[i_args + 4] * nrowsPack]);
                    setStorePol(storePol, args[i_args + 1], args[i_args + 2], 1);
                    i_args += 5;
                    break;
                }
                case 8: {
                    // OPERATION WITH DEST: commit1 - SRC0: tmp1 - SRC1: number
                    Goldilocks::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &tmp1[args[i_args + 3] * nrowsPack], &numbers_[args[i_args + 4]*nrowsPack]);
                    setStorePol(storePol, args[i_args + 1], args[i_args + 2], 1);
                    i_args += 5;
                    break;
                }
                case 9: {
                    // COPY public to commit1
                    Goldilocks::copy_pack(nrowsPack, &bufferT_[(nColsStagesAcc[args[i_args]] + args[i_args + 1]) * nrowsPack], &publics[args[i_args + 2] * nrowsPack]);
                    setStorePol(storePol, args[i_args], args[i_args + 1], 1);
                    i_args += 3;
                    break;
                }
                case 10: {
                    // OPERATION WITH DEST: commit1 - SRC0: public - SRC1: public
                    Goldilocks::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &publics[args[i_args + 3] * nrowsPack], &publics[args[i_args + 4] * nrowsPack]);
                    setStorePol(storePol, args[i_args + 1], args[i_args + 2], 1);
                    i_args += 5;
                    break;
                }
                case 11: {
                    // OPERATION WITH DEST: commit1 - SRC0: public - SRC1: number
                    Goldilocks::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &publics[args[i_args + 3] * nrowsPack], &numbers_[args[i_args + 4]*nrowsPack]);
                    setStorePol(storePol, args[i_args + 1], args[i_args + 2], 1);
                    i_args += 5;
                    break;
                }
                case 12: {
                    // COPY x to commit1
                    Goldilocks::copy_pack(nrowsPack, &bufferT_[(nColsStagesAcc[args[i_args]] + args[i_args + 1]) * nrowsPack], x[i]);
                    setStorePol(storePol, args[i_args], args[i_args + 1], 1);
                    i_args += 2;
                    break;
                }
                case 13: {
                    // COPY number to commit1
                    Goldilocks::copy_pack(nrowsPack, &bufferT_[(nColsStagesAcc[args[i_args]] + args[i_args + 1]) * nrowsPack], &numbers_[args[i_args + 2]*nrowsPack]);
                    setStorePol(storePol, args[i_args], args[i_args + 1], 1);
                    i_args += 3;
                    break;
                }
                case 14: {
                    // OPERATION WITH DEST: commit1 - SRC0: number - SRC1: number
                    Goldilocks::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &numbers_[args[i_args + 3]*nrowsPack], &numbers_[args[i_args + 4]*nrowsPack]);
                    setStorePol(storePol, args[i_args + 1], args[i_args + 2], 1);
                    i_args += 5;
                    break;
                }
                case 15: {
                    // COPY commit1 to tmp1
                    Goldilocks::copy_pack(nrowsPack, &tmp1[args[i_args] * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack]);
                    i_args += 3;
                    break;
                }
                case 16: {
                    // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: commit1
                    Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]) * nrowsPack]);
                    i_args += 6;
                    break;
                }
                case 17: {
                    // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: tmp1
                    Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &tmp1[args[i_args + 4] * nrowsPack]);
                    i_args += 5;
                    break;
                }
                case 18: {
                    // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: public
                    Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &publics[args[i_args + 4] * nrowsPack]);
                    i_args += 5;
                    break;
                }
                case 19: {
                    // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: number
                    Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &numbers_[args[i_args + 4]*nrowsPack]);
                    i_args += 5;
                    break;
                }
                case 20: {
                    // COPY tmp1 to tmp1
                    Goldilocks::copy_pack(nrowsPack, &tmp1[args[i_args] * nrowsPack], &tmp1[args[i_args + 1] * nrowsPack]);
                    i_args += 2;
                    break;
                }
                case 21: {
                    // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: tmp1
                    Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &tmp1[args[i_args + 2] * nrowsPack], &tmp1[args[i_args + 3] * nrowsPack]);
                    i_args += 4;
                    break;
                }
                case 22: {
                    // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: public
                    Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &tmp1[args[i_args + 2] * nrowsPack], &publics[args[i_args + 3] * nrowsPack]);
                    i_args += 4;
                    break;
                }
                case 23: {
                    // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: number
                    Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &tmp1[args[i_args + 2] * nrowsPack], &numbers_[args[i_args + 3]*nrowsPack]);
                    i_args += 4;
                    break;
                }
                case 24: {
                    // COPY public to tmp1
                    Goldilocks::copy_pack(nrowsPack, &tmp1[args[i_args] * nrowsPack], &publics[args[i_args + 1] * nrowsPack]);
                    i_args += 2;
                    break;
                }
                case 25: {
                    // OPERATION WITH DEST: tmp1 - SRC0: public - SRC1: public
                    Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &publics[args[i_args + 2] * nrowsPack], &publics[args[i_args + 3] * nrowsPack]);
                    i_args += 4;
                    break;
                }
                case 26: {
                    // OPERATION WITH DEST: tmp1 - SRC0: public - SRC1: number
                    Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &publics[args[i_args + 2] * nrowsPack], &numbers_[args[i_args + 3]*nrowsPack]);
                    i_args += 4;
                    break;
                }
                case 27: {
                    // COPY x to tmp1
                    Goldilocks::copy_pack(nrowsPack, &tmp1[args[i_args] * nrowsPack], x[i]);
                    i_args += 1;
                    break;
                }
                case 28: {
                    // COPY number to tmp1
                    Goldilocks::copy_pack(nrowsPack, &tmp1[args[i_args] * nrowsPack], &numbers_[args[i_args + 1]*nrowsPack]);
                    i_args += 2;
                    break;
                }
                case 29: {
                    // OPERATION WITH DEST: tmp1 - SRC0: number - SRC1: number
                    Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &numbers_[args[i_args + 2]*nrowsPack], &numbers_[args[i_args + 3]*nrowsPack]);
                    i_args += 4;
                    break;
                }
                case 30: {
                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: commit1
                    Goldilocks3::op_31_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 5]] + args[i_args + 6]) * nrowsPack]);
                    setStorePol(storePol, args[i_args + 1], args[i_args + 2], FIELD_EXTENSION);
                    i_args += 7;
                    break;
                }
                case 31: {
                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: tmp1
                    Goldilocks3::op_31_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack], &tmp1[args[i_args + 5] * nrowsPack]);
                    setStorePol(storePol, args[i_args + 1], args[i_args + 2], FIELD_EXTENSION);
                    i_args += 6;
                    break;
                }
                case 32: {
                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: public
                    Goldilocks3::op_31_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack], &publics[args[i_args + 5] * nrowsPack]);
                    setStorePol(storePol, args[i_args + 1], args[i_args + 2], FIELD_EXTENSION);
                    i_args += 6;
                    break;
                }
                case 33: {
                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: x
                    Goldilocks3::op_31_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack], x[i]);
                    setStorePol(storePol, args[i_args + 1], args[i_args + 2], FIELD_EXTENSION);
                    i_args += 5;
                    break;
                }
                case 34: {
                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: number
                    Goldilocks3::op_31_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack], &numbers_[args[i_args + 5]*nrowsPack]);
                    setStorePol(storePol, args[i_args + 1], args[i_args + 2], FIELD_EXTENSION);
                    i_args += 6;
                    break;
                }
                case 35: {
                    // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: commit1
                    Goldilocks3::op_31_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &tmp3[args[i_args + 3] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]) * nrowsPack]);
                    setStorePol(storePol, args[i_args + 1], args[i_args + 2], FIELD_EXTENSION);
                    i_args += 6;
                    break;
                }
                case 36: {
                    // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: tmp1
                    Goldilocks3::op_31_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &tmp3[args[i_args + 3] * nrowsPack * FIELD_EXTENSION], &tmp1[args[i_args + 4] * nrowsPack]);
                    setStorePol(storePol, args[i_args + 1], args[i_args + 2], FIELD_EXTENSION);
                    i_args += 5;
                    break;
                }
                case 37: {
                    // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: public
                    Goldilocks3::op_31_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &tmp3[args[i_args + 3] * nrowsPack * FIELD_EXTENSION], &publics[args[i_args + 4] * nrowsPack]);
                    setStorePol(storePol, args[i_args + 1], args[i_args + 2], FIELD_EXTENSION);
                    i_args += 5;
                    break;
                }
                case 38: {
                    // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: x
                    Goldilocks3::op_31_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &tmp3[args[i_args + 3] * nrowsPack * FIELD_EXTENSION], x[i]);
                    setStorePol(storePol, args[i_args + 1], args[i_args + 2], FIELD_EXTENSION);
                    i_args += 4;
                    break;
                }
                case 39: {
                    // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: number
                    Goldilocks3::op_31_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &tmp3[args[i_args + 3] * nrowsPack * FIELD_EXTENSION], &numbers_[args[i_args + 4]*nrowsPack]);
                    setStorePol(storePol, args[i_args + 1], args[i_args + 2], FIELD_EXTENSION);
                    i_args += 5;
                    break;
                }
                case 40: {
                    // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: commit1
                    Goldilocks3::op_31_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &challenges[args[i_args + 3]*FIELD_EXTENSION*nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]) * nrowsPack]);
                    setStorePol(storePol, args[i_args + 1], args[i_args + 2], FIELD_EXTENSION);
                    i_args += 6;
                    break;
                }
                case 41: {
                    // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: tmp1
                    Goldilocks3::op_31_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &challenges[args[i_args + 3]*FIELD_EXTENSION*nrowsPack], &tmp1[args[i_args + 4] * nrowsPack]);
                    setStorePol(storePol, args[i_args + 1], args[i_args + 2], FIELD_EXTENSION);
                    i_args += 5;
                    break;
                }
                case 42: {
                    // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: public
                    Goldilocks3::op_31_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &challenges[args[i_args + 3]*FIELD_EXTENSION*nrowsPack], &publics[args[i_args + 4] * nrowsPack]);
                    setStorePol(storePol, args[i_args + 1], args[i_args + 2], FIELD_EXTENSION);
                    i_args += 5;
                    break;
                }
                case 43: {
                    // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: x
                    Goldilocks3::op_31_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &challenges[args[i_args + 3]*FIELD_EXTENSION*nrowsPack], x[i]);
                    setStorePol(storePol, args[i_args + 1], args[i_args + 2], FIELD_EXTENSION);
                    i_args += 4;
                    break;
                }
                case 44: {
                    // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: number
                    Goldilocks3::op_31_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &challenges[args[i_args + 3]*FIELD_EXTENSION*nrowsPack], &numbers_[args[i_args + 4]*nrowsPack]);
                    setStorePol(storePol, args[i_args + 1], args[i_args + 2], FIELD_EXTENSION);
                    i_args += 5;
                    break;
                }
                case 45: {
                    // COPY commit3 to commit3
                    Goldilocks3::copy_pack(nrowsPack, &bufferT_[(nColsStagesAcc[args[i_args]] + args[i_args + 1]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack]);
                    setStorePol(storePol, args[i_args], args[i_args + 1], FIELD_EXTENSION);
                    i_args += 4;
                    break;
                }
                case 46: {
                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: commit3
                    Goldilocks3::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 5]] + args[i_args + 6]) * nrowsPack]);
                    setStorePol(storePol, args[i_args + 1], args[i_args + 2], FIELD_EXTENSION);
                    i_args += 7;
                    break;
                }
                case 47: {
                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: tmp3
                    Goldilocks3::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack], &tmp3[args[i_args + 5] * nrowsPack * FIELD_EXTENSION]);
                    setStorePol(storePol, args[i_args + 1], args[i_args + 2], FIELD_EXTENSION);
                    i_args += 6;
                    break;
                }
                case 48: {
                    // MULTIPLICATION WITH DEST: commit3 - SRC0: commit3 - SRC1: challenge
                    Goldilocks3::mul_pack(nrowsPack, &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack], &challenges[args[i_args + 5]*FIELD_EXTENSION*nrowsPack], &challenges_ops[args[i_args + 5]*FIELD_EXTENSION*nrowsPack]);
                    setStorePol(storePol, args[i_args + 1], args[i_args + 2], FIELD_EXTENSION);
                    i_args += 6;
                    break;
                }
                case 49: {
                    // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: challenge
                    Goldilocks3::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack], &challenges[args[i_args + 5]*FIELD_EXTENSION*nrowsPack]);
                    setStorePol(storePol, args[i_args + 1], args[i_args + 2], FIELD_EXTENSION);
                    i_args += 6;
                    break;
                }
                case 50: {
                    // COPY tmp3 to commit3
                    Goldilocks3::copy_pack(nrowsPack, &bufferT_[(nColsStagesAcc[args[i_args]] + args[i_args + 1]) * nrowsPack], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION]);
                    setStorePol(storePol, args[i_args], args[i_args + 1], FIELD_EXTENSION);
                    i_args += 3;
                    break;
                }
                case 51: {
                    // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: tmp3
                    Goldilocks3::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &tmp3[args[i_args + 3] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 4] * nrowsPack * FIELD_EXTENSION]);
                    setStorePol(storePol, args[i_args + 1], args[i_args + 2], FIELD_EXTENSION);
                    i_args += 5;
                    break;
                }
                case 52: {
                    // MULTIPLICATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: challenge
                    Goldilocks3::mul_pack(nrowsPack, &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &tmp3[args[i_args + 3] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 4]*FIELD_EXTENSION*nrowsPack], &challenges_ops[args[i_args + 4]*FIELD_EXTENSION*nrowsPack]);
                    setStorePol(storePol, args[i_args + 1], args[i_args + 2], FIELD_EXTENSION);
                    i_args += 5;
                    break;
                }
                case 53: {
                    // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: challenge
                    Goldilocks3::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &tmp3[args[i_args + 3] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 4]*FIELD_EXTENSION*nrowsPack]);
                    setStorePol(storePol, args[i_args + 1], args[i_args + 2], FIELD_EXTENSION);
                    i_args += 5;
                    break;
                }
                case 54: {
                    // MULTIPLICATION WITH DEST: commit3 - SRC0: challenge - SRC1: challenge
                    Goldilocks3::mul_pack(nrowsPack, &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &challenges[args[i_args + 3]*FIELD_EXTENSION*nrowsPack], &challenges[args[i_args + 4]*FIELD_EXTENSION*nrowsPack], &challenges_ops[args[i_args + 4]*FIELD_EXTENSION*nrowsPack]);
                    setStorePol(storePol, args[i_args + 1], args[i_args + 2], FIELD_EXTENSION);
                    i_args += 5;
                    break;
                }
                case 55: {
                    // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: challenge
                    Goldilocks3::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &challenges[args[i_args + 3]*FIELD_EXTENSION*nrowsPack], &challenges[args[i_args + 4]*FIELD_EXTENSION*nrowsPack]);
                    setStorePol(storePol, args[i_args + 1], args[i_args + 2], FIELD_EXTENSION);
                    i_args += 5;
                    break;
                }
                case 56: {
                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: commit1
                    Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]) * nrowsPack]);
                    i_args += 6;
                    break;
                }
                case 57: {
                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: tmp1
                    Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &tmp1[args[i_args + 4] * nrowsPack]);
                    i_args += 5;
                    break;
                }
                case 58: {
                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: public
                    Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &publics[args[i_args + 4] * nrowsPack]);
                    i_args += 5;
                    break;
                }
                case 59: {
                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: x
                    Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], x[i]);
                    i_args += 4;
                    break;
                }
                case 60: {
                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: number
                    Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &numbers_[args[i_args + 4]*nrowsPack]);
                    i_args += 5;
                    break;
                }
                case 61: {
                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: commit1
                    Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack]);
                    i_args += 5;
                    break;
                }
                case 62: {
                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: tmp1
                    Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], &tmp1[args[i_args + 3] * nrowsPack]);
                    i_args += 4;
                    break;
                }
                case 63: {
                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: public
                    Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], &publics[args[i_args + 3] * nrowsPack]);
                    i_args += 4;
                    break;
                }
                case 64: {
                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: x
                    Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], x[i]);
                    i_args += 3;
                    break;
                }
                case 65: {
                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: number
                    Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], &numbers_[args[i_args + 3]*nrowsPack]);
                    i_args += 4;
                    break;
                }
                case 66: {
                    // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: commit1
                    Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION*nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack]);
                    i_args += 5;
                    break;
                }
                case 67: {
                    // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: tmp1
                    Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION*nrowsPack], &tmp1[args[i_args + 3] * nrowsPack]);
                    i_args += 4;
                    break;
                }
                case 68: {
                    // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: public
                    Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION*nrowsPack], &publics[args[i_args + 3] * nrowsPack]);
                    i_args += 4;
                    break;
                }
                case 69: {
                    // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: x
                    Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION*nrowsPack], x[i]);
                    i_args += 3;
                    break;
                }
                case 70: {
                    // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: number
                    Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION*nrowsPack], &numbers_[args[i_args + 3]*nrowsPack]);
                    i_args += 4;
                    break;
                }
                case 71: {
                    // COPY commit3 to tmp3
                    Goldilocks3::copy_pack(nrowsPack, &tmp3[args[i_args] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack]);
                    i_args += 3;
                    break;
                }
                case 72: {
                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: commit3
                    Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]) * nrowsPack]);
                    i_args += 6;
                    break;
                }
                case 73: {
                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: tmp3
                    Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &tmp3[args[i_args + 4] * nrowsPack * FIELD_EXTENSION]);
                    i_args += 5;
                    break;
                }
                case 74: {
                    // MULTIPLICATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: challenge
                    Goldilocks3::mul_pack(nrowsPack, &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &challenges[args[i_args + 4]*FIELD_EXTENSION*nrowsPack], &challenges_ops[args[i_args + 4]*FIELD_EXTENSION*nrowsPack]);
                    i_args += 5;
                    break;
                }
                case 75: {
                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: challenge
                    Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &challenges[args[i_args + 4]*FIELD_EXTENSION*nrowsPack]);
                    i_args += 5;
                    break;
                }
                case 76: {
                    // COPY tmp3 to tmp3
                    Goldilocks3::copy_pack(nrowsPack, &tmp3[args[i_args] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION]);
                    i_args += 2;
                    break;
                }
                case 77: {
                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: tmp3
                    Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 3] * nrowsPack * FIELD_EXTENSION]);
                    i_args += 4;
                    break;
                }
                case 78: {
                    // MULTIPLICATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: challenge
                    Goldilocks3::mul_pack(nrowsPack, &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 3]*FIELD_EXTENSION*nrowsPack], &challenges_ops[args[i_args + 3]*FIELD_EXTENSION*nrowsPack]);
                    i_args += 4;
                    break;
                }
                case 79: {
                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: challenge
                    Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 3]*FIELD_EXTENSION*nrowsPack]);
                    i_args += 4;
                    break;
                }
                case 80: {
                    // MULTIPLICATION WITH DEST: tmp3 - SRC0: challenge - SRC1: challenge
                    Goldilocks3::mul_pack(nrowsPack, &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION*nrowsPack], &challenges[args[i_args + 3]*FIELD_EXTENSION*nrowsPack], &challenges_ops[args[i_args + 3]*FIELD_EXTENSION*nrowsPack]);
                    i_args += 4;
                    break;
                }
                case 81: {
                    // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: challenge
                    Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION*nrowsPack], &challenges[args[i_args + 3]*FIELD_EXTENSION*nrowsPack]);
                    i_args += 4;
                    break;
                }
                case 82: {
                    // COPY eval to tmp3
                    Goldilocks3::copy_pack(nrowsPack, &tmp3[args[i_args] * nrowsPack * FIELD_EXTENSION], &evals[args[i_args + 1]*FIELD_EXTENSION*nrowsPack]);
                    i_args += 2;
                    break;
                }
                case 83: {
                    // MULTIPLICATION WITH DEST: tmp3 - SRC0: eval - SRC1: challenge
                    Goldilocks3::mul_pack(nrowsPack, &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &evals[args[i_args + 2]*FIELD_EXTENSION*nrowsPack], &challenges[args[i_args + 3]*FIELD_EXTENSION*nrowsPack], &challenges_ops[args[i_args + 3]*FIELD_EXTENSION*nrowsPack]);
                    i_args += 4;
                    break;
                }
                case 84: {
                    // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: eval
                    Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION*nrowsPack], &evals[args[i_args + 3]*FIELD_EXTENSION*nrowsPack]);
                    i_args += 4;
                    break;
                }
                case 85: {
                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: eval
                    Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], &evals[args[i_args + 3]*FIELD_EXTENSION*nrowsPack]);
                    i_args += 4;
                    break;
                }
                case 86: {
                    // OPERATION WITH DEST: tmp3 - SRC0: eval - SRC1: commit1
                    Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &evals[args[i_args + 2]*FIELD_EXTENSION*nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack]);
                    i_args += 5;
                    break;
                }
                case 87: {
                    // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: eval
                    Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &evals[args[i_args + 4]*FIELD_EXTENSION*nrowsPack]);
                    i_args += 5;
                    break;
                }
                case 88: {
                    // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: xDivXSubXi
                    Goldilocks3::load_pack(nrowsPack, tmp3_1, 1, params.xDivXSubXi[i + args[i_args + 3]*domainSize], uint64_t(FIELD_EXTENSION));
                    Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], tmp3_1);
                    i_args += 4;
                    break;
                }
                case 89: {
                    // OPERATION WITH DEST: q - SRC0: tmp3 - SRC1: Zi
                    Goldilocks3::op_31_pack(nrowsPack, 2, tmp3_, &tmp3[args[i_args] * nrowsPack * FIELD_EXTENSION], params.zi[i]);
                    Goldilocks3::store_pack(nrowsPack, &params.q_2ns[i*FIELD_EXTENSION], uint64_t(FIELD_EXTENSION), tmp3_, 1);
                    i_args += 1;
                    break;
                }
                case 90: {
                    // OPERATION WITH DEST: f - SRC0: tmp3 - SRC1: tmp3
                    Goldilocks3::op_pack(nrowsPack, args[i_args], tmp3_, &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION]);
                    Goldilocks3::store_pack(nrowsPack, &params.f_2ns[i*FIELD_EXTENSION], uint64_t(FIELD_EXTENSION), tmp3_, 1);
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

#endif