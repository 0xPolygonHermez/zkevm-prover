#ifndef PROVE_FRI
#define PROVE_FRI

#include "transcript.hpp"
#include "stark_info.hpp"
#include "proofFRI.hpp"
#include <cassert>
#include <vector>
#include "ntt_goldilocks.hpp"
#include "merklehash_goldilocks.hpp"

class ProveFRI
{
public:
    static void prove(FriProof &fproof, Goldilocks::Element **trees, Transcript transcript, Polinomial &friPol, uint64_t polBits, StarkInfo starkInfo);
    static void polMulAxi(Polinomial &pol, Goldilocks::Element init, Goldilocks::Element acc);
    static void evalPol(Polinomial &res, uint64_t res_idx, Polinomial &p, Polinomial &x);
    static void queryPol(FriProof &fproof, Goldilocks::Element **trees, uint64_t idx, uint64_t treeIdx);
    static void queryPol(FriProof &fproof, Goldilocks::Element *tree, uint64_t idx, uint64_t treeIdx);
};

#endif