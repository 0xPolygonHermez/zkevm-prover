#include "fft.hpp"
#include <cassert>
#include <thread>
#include <vector>
#include <omp.h>
#include <stdexcept>
#include <iostream>

uint32_t FFT::log2(uint64_t n)
{
    assert(n != 0);
    uint32_t res = 0;
    while (n != 1)
    {
        n >>= 1;
        res++;
    }
    return res;
}

static inline uint64_t BR(uint64_t x, uint64_t domainPow)
{
    x = (x >> 16) | (x << 16);
    x = ((x & 0xFF00FF00) >> 8) | ((x & 0x00FF00FF) << 8);
    x = ((x & 0xF0F0F0F0) >> 4) | ((x & 0x0F0F0F0F) << 4);
    x = ((x & 0xCCCCCCCC) >> 2) | ((x & 0x33333333) << 2);
    return (((x & 0xAAAAAAAA) >> 1) | ((x & 0x55555555) << 1)) >> (32 - domainPow);
}

#define ROOT(s, j) (rootsOfUnit[(1 << (s)) + (j)])

FFT::FFT(RawFr *_field, uint64_t maxDomainSize, uint32_t _nThreads)
{
    nThreads = _nThreads == 0 ? omp_get_max_threads() : _nThreads;
    field = _field;
    // f = Field::field;

    uint32_t domainPow = log2(maxDomainSize);

    mpz_t m_qm1d2;
    mpz_t m_q;
    mpz_t m_nqr;
    mpz_t m_aux;
    mpz_init(m_qm1d2);
    mpz_init(m_q);
    mpz_init(m_nqr);
    mpz_init(m_aux);

    field->toMpz(m_aux, field->negOne());

    mpz_add_ui(m_q, m_aux, 1);
    mpz_fdiv_q_2exp(m_qm1d2, m_aux, 1);

    mpz_set_ui(m_nqr, 2);
    mpz_powm(m_aux, m_nqr, m_qm1d2, m_q);
    while (mpz_cmp_ui(m_aux, 1) == 0)
    {
        mpz_add_ui(m_nqr, m_nqr, 1);
        mpz_powm(m_aux, m_nqr, m_qm1d2, m_q);
    }

    field->fromMpz(nqr, m_nqr);

    // std::std::cout << "nqr: " << field->toString(nqr) << std::std::endl;

    s = 1;
    mpz_set(m_aux, m_qm1d2);
    while ((!mpz_tstbit(m_aux, 0)) && (s < domainPow))
    {
        mpz_fdiv_q_2exp(m_aux, m_aux, 1);
        s++;
    }

    if (s < domainPow)
    {
        throw std::range_error("Domain size too big for the curve");
    }

    uint64_t nRoots = 1LL << s;

    roots = new RawFr::RawFr::Element[nRoots];
    powTwoInv = new RawFr::RawFr::Element[s + 1];

    field->copy(roots[0], field->one());
    field->copy(powTwoInv[0], field->one());
    if (nRoots > 1)
    {
        mpz_powm(m_aux, m_nqr, m_aux, m_q);
        field->fromMpz(roots[1], m_aux);

        mpz_set_ui(m_aux, 2);
        mpz_invert(m_aux, m_aux, m_q);
        field->fromMpz(powTwoInv[1], m_aux);
    }
#pragma omp parallel
    {
        int idThread = omp_get_thread_num();
        int nThreads = omp_get_num_threads();
        uint64_t increment = nRoots / nThreads;
        uint64_t start = idThread == 0 ? 2 : idThread * increment;
        uint64_t end = idThread == nThreads - 1 ? nRoots : (idThread + 1) * increment;
        if (end > start)
        {
            field->exp(roots[start], roots[1], (uint8_t *)(&start), sizeof(start));
        }
        for (uint64_t i = start + 1; i < end; i++)
        {
            field->mul(roots[i], roots[i - 1], roots[1]);
        }
    }
    RawFr::RawFr::Element aux;
    field->mul(aux, roots[nRoots - 1], roots[1]);
    assert(field->eq(aux, field->one()));

    for (uint64_t i = 2; i <= s; i++)
    {
        field->mul(powTwoInv[i], powTwoInv[i - 1], powTwoInv[1]);
    }

    mpz_clear(m_qm1d2);
    mpz_clear(m_q);
    mpz_clear(m_nqr);
    mpz_clear(m_aux);
}

FFT::~FFT()
{
    delete[] roots;
    delete[] powTwoInv;
}

void FFT::reversePermutation(RawFr::Element *a, uint64_t n) {
    int domainPow = log2(n);
    #pragma omp parallel for
    for (uint64_t i=0; i<n; i++) {
        RawFr::Element tmp;
        uint64_t r = BR(i, domainPow);
        if (i>r) {
            field->copy(tmp, a[i]);
            field->copy(a[i], a[r]);
            field->copy(a[r], tmp);
        }
    }
}



void FFT::fft(RawFr::Element *a, uint64_t n) {
    reversePermutation(a, n);
    uint64_t domainPow =log2(n);
    assert(((uint64_t)1 << domainPow) == n);
    for (u_int32_t s=1; s<=domainPow; s++) {
        uint64_t m = 1 << s;
        uint64_t mdiv2 = m >> 1;
        #pragma omp parallel for
        for (uint64_t i=0; i< (n>>1); i++) {
            RawFr::Element t;
            RawFr::Element u;
            uint64_t k=(i/mdiv2)*m;
            uint64_t j=i%mdiv2;

            field->mul(t, root(s, j), a[k+j+mdiv2]);
            field->copy(u,a[k+j]);
            field->add(a[k+j], t, u);
            field->sub(a[k+j+mdiv2], u, t);
        }
    }
}


void FFT::ifft(RawFr::Element *a, uint64_t n ) {
    fft(a, n);
    uint64_t domainPow =log2(n);
    uint64_t nDiv2= n >> 1; 
    #pragma omp parallel for
    for (uint64_t i=1; i<nDiv2; i++) {
        RawFr::Element tmp;
        uint64_t r = n-i;
        field->copy(tmp, a[i]);
        field->mul(a[i], a[r], powTwoInv[domainPow]);
        field->mul(a[r], tmp, powTwoInv[domainPow]);
    } 
    field->mul(a[0], a[0], powTwoInv[domainPow]);
    field->mul(a[n >> 1], a[n >> 1], powTwoInv[domainPow]);
}




void FFT::printVector(RawFr::Element *a, uint64_t n ) {
    std::cout << "[" << std::endl;
    for (uint64_t i=0; i<n; i++) {
        std::cout << field->toString(a[i]) << std::endl;
    }
    std::cout << "]" << std::endl;
}

