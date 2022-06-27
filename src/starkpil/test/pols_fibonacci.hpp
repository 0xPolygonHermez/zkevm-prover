
#ifndef POLS_FIBONACCI
#define POLS_FIBONACCI

#include "commit_pols_fibonacci.hpp"
#include "constant_pols_fibonacci.hpp"
#include "fibonacciExps.hpp"

#include "ntt_goldilocks.hpp"

struct qs
{
    uint64_t idExp;
    uint64_t idQ;
};
struct starkStruct
{
    uint64_t nBits;
    uint64_t nBitsExt;
    uint64_t nQueries;
    uint64_t extendBits; // Computed
    uint64_t N;          // Computed
    uint64_t N_Extended; // Computed
};
struct starkInfo
{
    uint64_t nCm1;
    uint64_t nConst;
    uint64_t nQ1;
    uint64_t nQ2;
    uint64_t nQ3;
    uint64_t nQ4;
    qs *qs1;
};

class PolsFibonacci
{

public:
    FibCommitPols cm;
    FibConstantPols constP;
    FibonacciExps exps;

    starkStruct structStark;
    starkInfo infoStark;

    Goldilocks::Element shift = Goldilocks::fromU64(49);

    PolsFibonacci(void *pCommitedAddress, void *pConstantAddress, starkStruct _structStark, starkInfo _infoStark) : cm(pCommitedAddress, _structStark.N, _infoStark.nCm1), constP(pConstantAddress), exps(_structStark.N, _structStark.nQueries), structStark(_structStark), infoStark(_infoStark) {}

    void extendCms(Goldilocks::Element *cm2ns, Goldilocks::Element *exp2ns, uint64_t nQ, qs *_qs)
    {

        // Compute Commited 2ns
        std::memcpy(cm2ns, cm.Fibonacci.pData, structStark.N * infoStark.nCm1 * sizeof(Goldilocks::Element));
        extendPol(cm2ns, infoStark.nCm1);

        // Compute exp 2ns
        for (uint64_t i = 0; i < nQ; i++)
        {
            std::memcpy(&exp2ns[_qs[i].idExp * structStark.N_Extended], &exps[_qs[i].idExp * structStark.N], structStark.N * sizeof(Goldilocks::Element));
        }

        // Assuming al the exp2ns are ordered to be able to compute by columns
        extendPol(&exp2ns[_qs[0].idExp * structStark.N_Extended], nQ);

    };

    void extendPol(Goldilocks::Element *elements, uint64_t ncols)
    {
        NTT_Goldilocks ntt(structStark.N);
        NTT_Goldilocks ntt_extension(structStark.N_Extended);

        // TODO: Pre-compute r
        Goldilocks::Element *r;
        r = (Goldilocks::Element *)malloc(structStark.N * sizeof(Goldilocks::Element));
        r[0] = Goldilocks::one();

        for (uint64_t i = 1; i < structStark.N; i++)
        {
            Goldilocks::mul(r[i], r[i - 1], shift);
        }

        ntt.INTT_Block(elements, structStark.N, ncols);

#pragma omp parallel for
        for (uint64_t i = 0; i < structStark.N; i++)
        {
            for (uint j = 0; j < ncols; j++)
            {
                elements[i * ncols + j] = elements[ncols * i + j] * r[i];
            }
        }
#pragma omp parallel for schedule(static)
        for (uint64_t i = structStark.N * ncols; i < structStark.N_Extended * ncols; i++)
        {
            elements[i] = Goldilocks::zero();
        }
        ntt_extension.NTT_Block(elements, structStark.N_Extended, ncols);

        free(r);
    }
};

#endif