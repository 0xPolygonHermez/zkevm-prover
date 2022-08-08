#ifndef FRI_PROVE
#define FRI_PROVE

#include "transcript.hpp"
#include "stark_info.hpp"
#include "friProof.hpp"
#include <cassert>
#include <vector>
#include "ntt_goldilocks.hpp"
#include "merklehash_goldilocks.hpp"

class FRIProve
{
public:
    static void prove(FRIProof &fproof, Goldilocks::Element **trees, Transcript transcript, Polinomial &friPol, uint64_t polBits, StarkInfo starkInfo);
    static void polMulAxi(Polinomial &pol, Goldilocks::Element init, Goldilocks::Element acc);
    static void evalPol(Polinomial &res, uint64_t res_idx, Polinomial &p, Polinomial &x);
    static void queryPol(FRIProof &fproof, Goldilocks::Element **trees, uint64_t idx, uint64_t treeIdx);
    static void queryPol(FRIProof &fproof, Goldilocks::Element *tree, uint64_t idx, uint64_t treeIdx);
};

#endif