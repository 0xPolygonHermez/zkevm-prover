#ifndef POLS2NS_FIBONACCI
#define POLS2NS_FIBONACCI

#include "commit_pols_fibonacci.hpp"

class Fib2nsCommitPols
{
    Goldilocks::Element *pData;

public:
    Fib2nsCommitPols()
    {
        pData = (Goldilocks::Element *)malloc(2 * 1024 * sizeof(Goldilocks::Element));
    };
    ~Fib2nsCommitPols()
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
    Fib2nsCommitPols cm;
    Pols2nsFibonacci(FibCommitPols &cm){

    };
};

#endif