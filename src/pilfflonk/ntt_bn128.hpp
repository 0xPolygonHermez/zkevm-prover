#ifndef NTT_BN128
#define NTT_BN128

#include <cassert>
#include <gmp.h>
#include <omp.h>
#include <alt_bn128.hpp>

class NTT_AltBn128
{

#define NUM_PHASES 3
#define NUM_BLOCKS 1

private:
    AltBn128::Engine &E;

    u_int32_t s = 0;
    u_int32_t nThreads;
    AltBn128::FrElement nqr;
    AltBn128::FrElement *roots;
    AltBn128::FrElement *powTwoInv;
    AltBn128::FrElement *rrrrr;
    AltBn128::FrElement *rrrrr_;
    int extension;

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

    void NTT_iters(AltBn128::FrElement *dst, AltBn128::FrElement *src, u_int64_t size, u_int64_t offset_cols, u_int64_t ncols, u_int64_t ncols_all, u_int64_t nphase, AltBn128::FrElement *aux, bool inverse);

    inline int intt_idx(int i, int N)
    {
        int ind1 = N - i;
        if (ind1 == N)
        {
            ind1 = 0;
        }
        return ind1;
    }

public:
    NTT_AltBn128(AltBn128::Engine &E, u_int64_t maxDomainSize, u_int32_t _nThreads = 0, int extension_ = 1) : E(E)
    {
        if (maxDomainSize == 0) return;
        nThreads = _nThreads == 0 ? omp_get_max_threads() : _nThreads;
        extension = extension_;

        u_int32_t domainPow = NTT_AltBn128::log2(maxDomainSize);

        mpz_t m_qm1d2;
        mpz_t m_q;
        mpz_t m_nqr;
        mpz_t m_aux;
        mpz_init(m_qm1d2);
        mpz_init(m_q);
        mpz_init(m_nqr);
        mpz_init(m_aux);

        E.fr.toMpz(m_aux, E.fr.negOne());

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

        E.fr.fromMpz(nqr, m_nqr);

        if (s < domainPow)
        {
            throw std::range_error("Domain size too big for the curve");
        }

        uint64_t nRoots = 1LL << s;

        roots = (AltBn128::FrElement *)malloc(nRoots * sizeof(AltBn128::FrElement));
        powTwoInv = (AltBn128::FrElement *)malloc((s + 1) * sizeof(AltBn128::FrElement));

        roots[0] = E.fr.one();
        powTwoInv[0] = E.fr.one();

        if (nRoots > 1)
        {
            mpz_powm(m_aux, m_nqr, m_aux, m_q);
            E.fr.fromMpz(roots[1], m_aux);

            mpz_set_ui(m_aux, 2);
            mpz_invert(m_aux, m_aux, m_q);
            E.fr.fromMpz(powTwoInv[1], m_aux);
        }

        // calculate the rest of roots of unity
        for (uint64_t i = 2; i < nRoots; i++)
        {
            roots[i] = E.fr.mul(roots[i - 1], roots[1]);
        }

        AltBn128::FrElement aux = E.fr.mul(roots[nRoots - 1], roots[1]);
        assert(E.fr.eq(aux, E.fr.one()));
        
        for (uint64_t i = 2; i <= s; i++)
        {
            powTwoInv[i] = E.fr.mul(powTwoInv[i - 1], powTwoInv[1]);
        }

        mpz_clear(m_qm1d2);
        mpz_clear(m_q);
        mpz_clear(m_nqr);
        mpz_clear(m_aux);
    };
    ~NTT_AltBn128()
    {
        if (s != 0)
        {
            free(roots);
            free(powTwoInv);
        }
    }

    void NTT(AltBn128::FrElement *dst, AltBn128::FrElement *src, u_int64_t size, u_int64_t ncols = 1, AltBn128::FrElement *buffer = NULL, u_int64_t nphase = NUM_PHASES, u_int64_t nblock = NUM_BLOCKS, bool inverse = false);

    void INTT(AltBn128::FrElement *dst, AltBn128::FrElement *src, u_int64_t size, u_int64_t ncols = 1, AltBn128::FrElement *buffer = NULL, u_int64_t nphase = NUM_PHASES, u_int64_t nblock = NUM_BLOCKS);

    void reversePermutation(AltBn128::FrElement *dst, AltBn128::FrElement *src, u_int64_t size, u_int64_t offset_cols, u_int64_t ncols, u_int64_t ncols_all);

    inline AltBn128::FrElement &root(u_int32_t domainPow, u_int64_t idx)
    {
        return roots[idx << (s - domainPow)];
    }

    void extendPol(AltBn128::FrElement *output, AltBn128::FrElement *input, uint64_t N_Extended, uint64_t N, uint64_t ncols, AltBn128::FrElement *buffer = NULL, u_int64_t nphase = NUM_PHASES, u_int64_t nblock = NUM_BLOCKS);
};

#endif
