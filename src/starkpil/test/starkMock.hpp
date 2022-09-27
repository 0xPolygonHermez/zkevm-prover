#ifndef STARK_MOCK_HPP
#define STARK_MOCK_HPP

#include "utils.hpp"
#include "stark_info.hpp"
#include "commit_pols_basic.hpp"
#include "constant_pols_basic.hpp"
#include "zhInv.hpp"
#include "ntt_goldilocks.hpp"
#include "transcript.hpp"
#include "timer.hpp"
#include "merklehash_goldilocks.hpp"
#include "friProof.hpp"
#include "friProve.hpp"
#include "proof2zkinStark.hpp"
#include <fstream>
#include <iostream>
#include "merkleTreeGL.hpp"

#define NUM_CHALLENGES 8
#define NUM_TREES 8
#define zkinFile "basic.proof.zkin.json"
#define starkFile "basic.prove.json"
#define publicFile "basic.public.json"

class StarkMock
{
    const Config &config;
    StarkInfo starkInfo;
    void *pConstPolsAddress;
    ConstantPolsBasic *pConstPols;
    void *pConstPolsAddress2ns;
    ConstantPolsBasic *pConstPols2ns;
    void *pConstTreeAddress;
    ZhInv zi;
    uint64_t numCommited;
    uint64_t N;
    uint64_t NExtended;
    NTT_Goldilocks ntt;
    Polinomial x_n;
    Polinomial x_2ns;
    Polinomial challenges;
    Polinomial xDivXSubXi;
    Polinomial xDivXSubWXi;
    Polinomial evals;
    Goldilocks::Element *trees[5];
    MerkleTreeGL *treesGL[5];

private:
    void calculateH1H2(Polinomial &h1, Polinomial &h2, Polinomial &fPol, Polinomial &tPol);
    void calculateZ(Polinomial &z, Polinomial &num, Polinomial &den);
    static inline void batchInverse(Polinomial &res, Polinomial &src);

public:
    StarkMock(const Config &config);
    ~StarkMock();

    /* Returns the size of all the polynomials: committed, constant, etc. */
    uint64_t getTotalPolsSize(void) { return starkInfo.mapTotalN * sizeof(Goldilocks::Element); }

    /* Returns the size of the committed polynomials */
    uint64_t getCommitPolsSize(void) { return starkInfo.mapOffsets.section[cm2_n] * sizeof(Goldilocks::Element); }

    /* Generates a proof from the address to all polynomials memory area, and the committed pols */
    void genProof(void *pAddress, FRIProof &proof);

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

    class CompareGL3
    {
    public:
        bool operator()(const vector<Goldilocks::Element> &a, const vector<Goldilocks::Element> &b) const
        {
            if (a.size() == 1)
            {
                return Goldilocks::toU64(a[0]) < Goldilocks::toU64(b[0]);
            }
            else if (Goldilocks::toU64(a[0]) != Goldilocks::toU64(b[0]))
            {
                return Goldilocks::toU64(a[0]) < Goldilocks::toU64(b[0]);
            }
            else if (Goldilocks::toU64(a[1]) != Goldilocks::toU64(b[1]))
            {
                return Goldilocks::toU64(a[1]) < Goldilocks::toU64(b[1]);
            }
            else
            {
                return Goldilocks::toU64(a[2]) < Goldilocks::toU64(b[2]);
            }
        }
    };
};
#endif