#ifndef BINARY_SM_HPP
#define BINARY_SM_HPP

#include "config.hpp"
#include "binary_action.hpp"
#include "ff/ff.hpp"
#include "utils.hpp"
#include "commit_pols.hpp"

class BinaryExecutor
{
private:
    FiniteField &fr;
    const Config &config;
    uint64_t N;
    vector<vector<uint64_t>> FACTOR;
    vector<uint64_t> RESET;

public:
    BinaryExecutor (FiniteField &fr, const Config &config) : fr(fr), config(config)
    {
        // Set pol size
        N = BinaryCommitPols::degree();

        buildFactors();

        buildReset();
    }
    
    void execute (vector<BinaryAction> &action, BinaryCommitPols &pols);

    void execute (vector<BinaryAction> &action); // Only for testing purposes

private:
    void buildFactors (void);
    void buildReset (void);
};

#endif