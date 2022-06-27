
#ifndef POLS_FIBONACCI
#define POLS_FIBONACCI

#include "commit_pols_fibonacci.hpp"

class FibExp
{
    Goldilocks::Element *pData;

public:
    FibExp()
    {
        pData = (Goldilocks::Element *)malloc(8 * 1024 * sizeof(Goldilocks::Element));
    };
    ~FibExp()
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

class PolsFibonacci
{

public:
    FibCommitPols cm;
    FibConstantPols constP;
    FibExp exps;

    PolsFibonacci(void *pCommitedAddress, void *pConstantAddress) : cm(pCommitedAddress), constP(pConstantAddress)
    {
    }
};

#endif