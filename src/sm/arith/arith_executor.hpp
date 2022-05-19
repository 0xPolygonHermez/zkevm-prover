#ifndef ARITH_SM_HPP
#define ARITH_SM_HPP

#include "config.hpp"
#include "arith_action.hpp"
#include "utils.hpp"
#include "commit_pols.hpp"
#include "ffiasm/fec.hpp"
#include "scalar.hpp"

class ArithExecutor
{
private:
    FiniteField &fr;
    RawFec fec;
    const Config &config;
    mpz_class pFec;

public:
    ArithExecutor (FiniteField &fr, const Config &config) : fr(fr), config(config)
    {
        // Calculate the prime number
        fec2scalar(fec, fec.negOne(), pFec);
        pFec++;
    }
    ~ArithExecutor ()
    {
    }
    void execute (vector<ArithAction> &action, ArithCommitPols &pols);
};

#endif