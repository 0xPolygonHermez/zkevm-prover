#ifndef ARITH_SM_HPP
#define ARITH_SM_HPP

#include "config.hpp"
#include "arith_action.hpp"
#include "ff/ff.hpp"
#include "utils.hpp"

class ArithExecutor
{
private:
    FiniteField &fr;
    const Config &config;
    json pilJson;
    uint64_t polSize;

public:
    ArithExecutor (FiniteField &fr, const Config &config) : fr(fr), config(config)
    {
        // Set pol size
        polSize = 1<<22;

        // Parse PIL json file into memory
        file2json(config.binaryPilFile, pilJson);
    }
    ~ArithExecutor ()
    {
    }
    void execute (vector<ArithAction> &action);
};

#endif