#ifndef POLS2NS_FIBONACCI
#define POLS2NS_FIBONACCI

#include "commit_pols_fibonacci.hpp"

class FibExp2ns
{
    Goldilocks::Element *pData;

public:
    FibExp2ns()
    {
        pData = (Goldilocks::Element *)malloc(8 * 1024 * sizeof(Goldilocks::Element));
    };
    ~FibExp2ns()
    {
        delete pData;
    };
    Goldilocks::Element &operator[](int i) { return pData[i]; };
    Goldilocks::Element *operator=(Goldilocks::Element *pAddress)
    {
        pData = pAddress;
        return pData;
    };
};

class Pols2nsFibonacci
{

public:
    FibCommitPols cm;
    FibonacciExps exps;

    Goldilocks::Element *q;

    starkStruct structStark;
    starkInfo infoStark;
    Pols2nsFibonacci(void *pCommitedAddress, starkStruct _structStark, starkInfo _infoStark) : cm(pCommitedAddress, _structStark.N_Extended, _infoStark.nCm1), exps(_structStark.N_Extended, _structStark.nQueries), structStark(_structStark), infoStark(_infoStark)
    {
        uint64_t qSize = _infoStark.nQ1 + _infoStark.nQ2 + _infoStark.nQ3 + _infoStark.nQ4;
        q = (Goldilocks::Element *)malloc(_structStark.N_Extended * qSize * sizeof(Goldilocks::Element));
    }
    ~Pols2nsFibonacci() {
        free(q);
    }
};

#endif