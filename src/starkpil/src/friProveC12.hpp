#ifndef FRI_PROVE_C12
#define FRI_PROVE_C12

#include "transcriptBN128.hpp"
#include "merkleTreeBN128.hpp"
#include "stark_info.hpp"
#include "friProofC12.hpp"
#include <cassert>
#include <vector>
#include "ntt_goldilocks.hpp"
#include "merklehash_goldilocks.hpp"

class FRIProveC12
{
public:
    static void prove(FRIProofC12 &fproof, MerkleTreeBN128 **trees, TranscriptBN128 transcript, Polinomial &friPol, uint64_t polBits, StarkInfo starkInfo);
    static void polMulAxi(Polinomial &pol, Goldilocks::Element init, Goldilocks::Element acc);
    static void evalPol(Polinomial &res, uint64_t res_idx, Polinomial &p, Polinomial &x);
    static void getTransposed(Polinomial &aux, Polinomial &pol2_e, uint64_t trasposeBits);

    static void queryPol(FRIProofC12 &fproof, MerkleTreeBN128 **trees, uint64_t idx, uint64_t treeIdx);
    static void queryPol(FRIProofC12 &fproof, MerkleTreeBN128 *trees, uint64_t idx, uint64_t treeIdx);
};

#endif