#include "goldilocks_precomputed.hpp"
#include "zkmax.hpp"
#include "zkassert.hpp"

GoldilocksPrecomputed glp;


void GoldilocksPrecomputed::init (void)
{
    // Make sure it was not called before
    zkassert(!bInitialized);

    // Check that the size of the 2 precomputed arrays is a multiple of 8
    zkassert((GOLDILOCKS_PRECOMPUTED_MAX%8) == 0);

    // Init the positive inverse array
    invPos[0] = fr.zero();

#pragma omp parallel for
    for (uint64_t j=0; j<8; j++)
    {
        for (uint64_t i=zkmax(1,GOLDILOCKS_PRECOMPUTED_MAX*j/8) ; i<GOLDILOCKS_PRECOMPUTED_MAX*(j+1)/8; i++)
        {
            invPos[i] = fr.inv(fr.fromU64(i));
        }
    }

    // Init the negative inverse array
    invNeg[0] = fr.zero();

#pragma omp parallel for
    for (uint64_t j=0; j<8; j++)
    {
        for (uint64_t i=zkmax(1,GOLDILOCKS_PRECOMPUTED_MAX*j/8) ; i<GOLDILOCKS_PRECOMPUTED_MAX*(j+1)/8; i++)
        {
            invNeg[i] = fr.inv(fr.fromU64(GOLDILOCKS_PRIME - i));
        }
    }

    // Mark as initialized
    bInitialized = true;
}