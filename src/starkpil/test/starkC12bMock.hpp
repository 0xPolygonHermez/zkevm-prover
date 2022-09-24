#ifndef STARK_C12_B_MOCK_HPP
#define STARK_C12_B_MOCK_HPP

#include "stark_info.hpp"
#include "commit_pols_basic_C12b.hpp"
#include "constant_pols_basic_C12b.hpp"
#include "transcriptBN128.hpp"
#include "zhInv.hpp"
#include "merklehash_goldilocks.hpp"
#include "polinomial.hpp"
#include "ntt_goldilocks.hpp"
#include "friProof.hpp"
#include "friProveC12.hpp"
#include "proof2zkinStark.hpp"
#include "compare_fe.hpp"
#include <fstream>
#include <iostream>
#include <math.h> /* floor */
#include "merkleTreeBN128.hpp"

#define BN128_ARITY 16

class StarkC12bMock
{
    const Config &config;
    StarkInfo starkInfo;
    void *pConstPolsAddress;
    ConstantPolsBasicC12b *pConstPols;
    void *pConstPolsAddress2ns;
    ConstantPolsBasicC12b *pConstPols2ns;
    void *pConstTreeAddress;
    ZhInv zi;
    uint64_t numCommited;
    uint64_t N;
    uint64_t NExtended;
    NTT_Goldilocks ntt;
    TranscriptBN128 transcript;
    Polinomial x_n;
    Polinomial x_2ns;
    Polinomial challenges;
    Polinomial xDivXSubXi;
    Polinomial xDivXSubWXi;
    Polinomial evals;
    MerkleTreeBN128 *trees[5];

public:
    StarkC12bMock(const Config &config);
    ~StarkC12bMock();

    uint64_t getConstTreeSize(uint64_t n, uint64_t pol)
    {
        uint n_tmp = n;
        uint64_t nextN = floor(((double)(n_tmp - 1) / BN128_ARITY) + 1);
        uint64_t acc = nextN * BN128_ARITY;
        while (n_tmp > 1)
        {
            // FIll with zeros if n nodes in the leve is not even
            n_tmp = nextN;
            nextN = floor((n_tmp - 1) / BN128_ARITY) + 1;
            if (n_tmp > 1)
            {
                acc += nextN * 16;
            }
            else
            {
                acc += 1;
            }
        }

        uint64_t numElements = n * pol;
        uint64_t total = numElements + acc * 4;
        return total * 8 + 16; // + HEADER
    }

    uint64_t getTreeSize(uint64_t n, uint64_t pol)
    {
        uint n_tmp = n;
        uint64_t nextN = floor(((double)(n_tmp - 1) / BN128_ARITY) + 1);
        uint64_t acc = nextN * BN128_ARITY;
        while (n_tmp > 1)
        {
            // FIll with zeros if n nodes in the leve is not even
            n_tmp = nextN;
            nextN = floor((n_tmp - 1) / BN128_ARITY) + 1;
            if (n_tmp > 1)
            {
                acc += nextN * 16;
            }
            else
            {
                acc += 1;
            }
        }

        uint64_t numElements = n * pol * sizeof(Goldilocks::Element);
        uint64_t total = numElements + acc * sizeof(RawFr::Element);
        return total + 16; // + HEADER
    }

    /* Returns the size of all the polynomials: committed, constant, etc. */
    uint64_t getTotalPolsSize(void) { return starkInfo.mapTotalN * sizeof(Goldilocks::Element); }

    /* Returns the size of the committed polynomials */
    uint64_t getCommitPolsSize(void) { return starkInfo.mapOffsets.section[cm2_n] * sizeof(Goldilocks::Element); }

    /* Generates a proof from the address to all polynomials memory area, and the committed pols */
    void genProof(void *pAddress, FRIProofC12 &proof, Goldilocks::Element publicInputs[8]);

    void step2prev_first(Goldilocks::Element *pols, const Goldilocks::Element *publicInputs, uint64_t i);
    void step2prev_i(Goldilocks::Element *pols, const Goldilocks::Element *publicInputs, uint64_t i);
    void step2prev_last(Goldilocks::Element *pols, const Goldilocks::Element *publicInputs, uint64_t i);

    void step3prev_first(Goldilocks::Element *pols, const Goldilocks::Element *publicInputs, uint64_t i);
    void step3prev_i(Goldilocks::Element *pols, const Goldilocks::Element *publicInputs, uint64_t i);
    void step3prev_last(Goldilocks::Element *pols, const Goldilocks::Element *publicInputs, uint64_t i);

    void step4_first(Goldilocks::Element *pols, const Goldilocks::Element *publicInputs, uint64_t i);
    void step4_i(Goldilocks::Element *pols, const Goldilocks::Element *publicInputs, uint64_t i);
    void step4_last(Goldilocks::Element *pols, const Goldilocks::Element *publicInputs, uint64_t i);

    void step42ns_first(Goldilocks::Element *pols, const Goldilocks::Element *publicInputs, uint64_t i);
    void step42ns_i(Goldilocks::Element *pols, const Goldilocks::Element *publicInputs, uint64_t i);
    void step42ns_last(Goldilocks::Element *pols, const Goldilocks::Element *publicInputs, uint64_t i);

    void step52ns_first(Goldilocks::Element *pols, const Goldilocks::Element *publicInputs, uint64_t i);
    void step52ns_i(Goldilocks::Element *pols, const Goldilocks::Element *publicInputs, uint64_t i);
    void step52ns_last(Goldilocks::Element *pols, const Goldilocks::Element *publicInputs, uint64_t i);
};
#endif