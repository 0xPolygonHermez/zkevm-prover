#ifndef BINARY_SM_HPP
#define BINARY_SM_HPP

#include "config.hpp"
#include "goldilocks_base_field.hpp"
#include "binary_action.hpp"
#include "utils.hpp"
#include "commit_pols.hpp"

class BinaryExecutor
{
private:
    Goldilocks &fr;
    const Config &config;
    const uint64_t N;
    vector<vector<uint64_t>> FACTOR;
    vector<uint64_t> RESET;

public:
    BinaryExecutor (Goldilocks &fr, const Config &config) :
        fr(fr),
        config(config),
        N(BinaryCommitPols::pilDegree())
    {
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