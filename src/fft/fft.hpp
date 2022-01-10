#ifndef FFT_HPP
#define FFT_HPP

#include "ffiasm/fr.hpp"
#include <cstdint>

using namespace std;

class FFT
{
    RawFr *field;
    uint32_t s;
    RawFr::Element nqr;
    RawFr::Element *roots;
    RawFr::Element *powTwoInv;
    uint32_t nThreads;

    void reversePermutationInnerLoop(RawFr::Element *a, u_int64_t from, u_int64_t to, u_int32_t domainPow);
    void reversePermutation(RawFr::Element *a, u_int64_t n);
    void fftInnerLoop(RawFr::Element *a, u_int64_t from, u_int64_t to, u_int32_t s);
    void finalInverseInner(RawFr::Element *a, u_int64_t from, u_int64_t to, u_int32_t domainPow);

public:
    FFT(RawFr *_field, uint64_t maxDomainSize, uint32_t _nThreads = 0);
    ~FFT();
    void fft(RawFr::Element *a, uint64_t n);
    void ifft(RawFr::Element *a, uint64_t n);

    uint32_t log2(uint64_t n);
    inline RawFr::Element &root(uint32_t domainPow, uint64_t idx) { return roots[idx << (s - domainPow)]; }

    void printVector(RawFr::Element *a, uint64_t n);
};

#endif // FFT_HPP