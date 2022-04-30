#ifndef BINARY_SM_HPP
#define BINARY_SM_HPP

#include "config.hpp"
#include "binary_action.hpp"
#include "binary_const_pols.hpp"
#include "ff/ff.hpp"
#include "utils.hpp"

class BinaryExecutor
{
private:
    FiniteField &fr;
    const Config &config;
    json pilJson;
    BinaryConstPols constPols;
    uint64_t polSize;

public:
    BinaryExecutor (FiniteField &fr, const Config &config) : fr(fr), config(config), constPols(config)
    {
        // Set pol size
        polSize = 1<<16;

        // Parse PIL json file into memory
        file2json(config.binaryPilFile, pilJson);

        // Allocate constant polynomials
        constPols.alloc(polSize, pilJson);
    }
    ~BinaryExecutor ()
    {
        // Deallocate constant polynomials
        constPols.dealloc();
    }
    void execute (vector<BinaryAction> &action);
};

#endif