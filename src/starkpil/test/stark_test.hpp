#ifndef STARK_TEST_HPP
#define STARK_TEST_HPP

#include <iostream>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "utils.hpp"
#include "stark_info.hpp"
#include "transcript.hpp"
#include "zhInv.hpp"
#include "commit_pols_all.hpp"
#include "constant_pols_all.hpp"
#include "merklehash_goldilocks.hpp"
#include "timer.hpp"
#include "utils.hpp"
#include "polinomial.hpp"
#include "ntt_goldilocks.hpp"

#include "public_inputs_all.hpp"
#include "proof.hpp"
#include <vector>

// Test vectors files
#define starkInfo_File "all.starkinfo.json"
#define commited_file "all.commit"
#define constant_file "all.const"
#define constant_tree_file "all.consttree"

#define NUM_CHALLENGES_TEST 8

using namespace std;

void StarkTest(void);

class CompareGL3
{
public:
    bool operator()(const vector<Goldilocks::Element> &a, const vector<Goldilocks::Element> &b) const
    {
        return Goldilocks::toU64(a[1]) < Goldilocks::toU64(b[1]);
    }
};

class StarkTestMock
{
    const Config &config;
    StarkInfo starkInfo;
    void *pConstPolsAddress;
    const ConstantPolsAll *pConstPols;
    void *pConstPolsAddress2ns;
    const ConstantPolsAll *pConstPols2ns;
    void *pConstTreeAddress;
    ZhInv zi;
    uint64_t numCommited;
    uint64_t N;
    uint64_t NExtended;
    NTT_Goldilocks ntt;
    Transcript transcript;

private:
    void calculateH1H2(Polinomial &h1, Polinomial &h2, Polinomial &fPol, Polinomial &tPol);

public:
    Polinomial x_n;
    Polinomial x_2ns;
    Polinomial challenges;

    StarkTestMock(const Config &config);
    ~StarkTestMock();

    /* Returns the size of all the polynomials: committed, constant, etc. */
    uint64_t getTotalPolsSize(void) { return starkInfo.mapTotalN * sizeof(Goldilocks::Element); }

    /* Returns the size of the committed polynomials */
    uint64_t getCommitPolsSize(void) { return starkInfo.mapOffsets.section[cm2_n] * sizeof(Goldilocks::Element); }

    /* Generates a proof from the address to all polynomials memory area, and the committed pols */
    void genProof(void *pAddress, CommitPolsAll &cmPols, ConstantPolsAll &const_n, const PublicInputsAll &publicInputs, Proof &proof);
};

#endif