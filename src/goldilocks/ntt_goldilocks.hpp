#ifndef NTT_GOLDILOCKS
#define NTT_GOLDILOCKS

#include "goldilocks_base_field.hpp"
#include <cassert>
#include <gmp.h>
#include <omp.h>

#define CACHESIZE 1 << 18
#define NUM_PHASES 3
#define NUM_BLOCKS 1

class NTT_Goldilocks
{
private:
    u_int32_t s;
    u_int32_t nThreads;
    uint64_t nqr;
    Goldilocks::Element *roots;
    Goldilocks::Element *powTwoInv;

    static u_int32_t log2(u_int64_t size)
    {
        {
            assert(size != 0);
            u_int32_t res = 0;
            while (size != 1)
            {
                size >>= 1;
                res++;
            }
            return res;
        }
    }
    void NTT_iters(Goldilocks::Element *dst, Goldilocks::Element *src, u_int64_t size, u_int64_t offset_cols, u_int64_t ncols, u_int64_t ncols_all, u_int64_t nphase, Goldilocks::Element *aux);

public:
    NTT_Goldilocks(u_int64_t maxDomainSize, u_int32_t _nThreads = 0)
    {
        nThreads = _nThreads == 0 ? omp_get_max_threads() : _nThreads;

        u_int32_t domainPow = NTT_Goldilocks::log2(maxDomainSize);

        mpz_t m_qm1d2;
        mpz_t m_q;
        mpz_t m_nqr;
        mpz_t m_aux;
        mpz_init(m_qm1d2);
        mpz_init(m_q);
        mpz_init(m_nqr);
        mpz_init(m_aux);

        u_int64_t negone = GOLDILOCKS_PRIME - 1;

        mpz_import(m_aux, 1, 1, sizeof(u_int64_t), 0, 0, &negone);
        mpz_add_ui(m_q, m_aux, 1);
        mpz_fdiv_q_2exp(m_qm1d2, m_aux, 1);

        mpz_set_ui(m_nqr, 2);
        mpz_powm(m_aux, m_nqr, m_qm1d2, m_q);
        while (mpz_cmp_ui(m_aux, 1) == 0)
        {
            mpz_add_ui(m_nqr, m_nqr, 1);
            mpz_powm(m_aux, m_nqr, m_qm1d2, m_q);
        }

        s = 1;
        mpz_set(m_aux, m_qm1d2);
        while ((!mpz_tstbit(m_aux, 0)) && (s < domainPow))
        {
            mpz_fdiv_q_2exp(m_aux, m_aux, 1);
            s++;
        }

        nqr = mpz_get_ui(m_nqr);

        if (s < domainPow)
        {
            throw std::range_error("Domain size too big for the curve");
        }

        uint64_t nRoots = 1LL << s;

        roots = (Goldilocks::Element *)malloc(nRoots * sizeof(Goldilocks::Element));
        powTwoInv = (Goldilocks::Element *)malloc((s + 1) * sizeof(Goldilocks::Element));

        roots[0] = Goldilocks::one();
        powTwoInv[0] = Goldilocks::one();

        if (nRoots > 1)
        {
            mpz_powm(m_aux, m_nqr, m_aux, m_q);
            roots[1] = Goldilocks::fromU64(mpz_get_ui(m_aux));

            mpz_set_ui(m_aux, 2);
            mpz_invert(m_aux, m_aux, m_q);
            powTwoInv[1] = Goldilocks::fromU64(mpz_get_ui(m_aux));
        }

        for (uint64_t i = 2; i < nRoots; i++)
        {
            roots[i] = roots[i - 1] * roots[1];
        }

        Goldilocks::Element aux = roots[nRoots - 1] * roots[1];

        assert(Goldilocks::toU64(aux) == 1);

        for (uint64_t i = 2; i <= s; i++)
        {
            powTwoInv[i] = powTwoInv[i - 1] * powTwoInv[1];
        }

        mpz_clear(m_qm1d2);
        mpz_clear(m_q);
        mpz_clear(m_nqr);
        mpz_clear(m_aux);
    };
    ~NTT_Goldilocks()
    {
        free(roots);
        free(powTwoInv);
    }

    void NTT(Goldilocks::Element *dst, Goldilocks::Element *src, u_int64_t size, u_int64_t ncols = 1, u_int64_t nphase = NUM_PHASES, u_int64_t nblock = NUM_BLOCKS);
    void INTT(Goldilocks::Element *dst, Goldilocks::Element *src, u_int64_t size, u_int64_t ncols = 1, u_int64_t nphase = NUM_PHASES, u_int64_t nblock = NUM_BLOCKS);
    void reversePermutation(Goldilocks::Element *dst, Goldilocks::Element *src, u_int64_t size, u_int64_t offset_cols, u_int64_t ncols, u_int64_t ncols_all);
    inline Goldilocks::Element &root(u_int32_t domainPow, u_int64_t idx)
    {
        return roots[idx << (s - domainPow)];
    }
    void extendPol(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t N, uint64_t N_Extended, uint64_t ncols);
};

#endif
