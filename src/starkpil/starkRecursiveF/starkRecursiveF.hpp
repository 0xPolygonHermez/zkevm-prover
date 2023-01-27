#ifndef STARK_RECURSIVE_FINAL_HPP
#define STARK_RECURSIVE_FINAL_HPP

#include "stark_info.hpp"
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
#include "constant_pols_starks.hpp"
#include "commit_pols_starks.hpp"
#include "steps.hpp"

#define BN128_ARITY 16
#define STARK_RECURSIVE_F_NUM_TREES 5

class StarkRecursiveF
{
    const Config &config;

public:
    StarkInfo starkInfo;

private:
    void *pConstPolsAddress;
    void *pConstPolsAddress2ns;
    ConstantPolsStarks *pConstPols;
    ConstantPolsStarks *pConstPols2ns;
    void *pConstTreeAddress;
    ZhInv zi;
    uint64_t N;
    uint64_t NExtended;
    NTT_Goldilocks ntt;
    NTT_Goldilocks nttExtended;
    Polinomial x_n;
    Polinomial x_2ns;
    uint64_t constPolsSize;
    uint64_t constPolsDegree;

    Goldilocks::Element *mem;

    Goldilocks::Element *p_cm1_2ns;
    Goldilocks::Element *p_cm1_n;
    Goldilocks::Element *p_cm2_2ns;
    Goldilocks::Element *p_cm2_n;
    Goldilocks::Element *p_cm3_2ns;
    Goldilocks::Element *p_cm3_n;
    Goldilocks::Element *cm4_2ns;
    Goldilocks::Element *p_q_2ns;
    Goldilocks::Element *p_f_2ns;

    Goldilocks::Element *pBuffer;

    void *pAddress;

public:
    StarkRecursiveF(const Config &config, void *_pAddress);
    ~StarkRecursiveF();

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
    void genProof(FRIProofC12 &proof, Goldilocks::Element publicInputs[8]);
};
#endif