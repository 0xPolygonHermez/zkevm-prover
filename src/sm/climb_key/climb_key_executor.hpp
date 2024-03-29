#ifndef CLIMB_KEY_SM_HPP
#define CLIMB_KEY_SM_HPP

#include <gmpxx.h>
#include "definitions.hpp"
#include "config.hpp"
#include "goldilocks_base_field.hpp"
#include "sm/pols_generated/commit_pols.hpp"

USING_PROVER_FORK_NAMESPACE;

class ClimbKeyAction
{
public:
    Goldilocks::Element key[4];
    uint16_t level;
    uint8_t bit;
};


class ClimbKeyHelper
{
    public:
        static bool calculate(Goldilocks &fr, const Goldilocks::Element &current, bool bit, Goldilocks::Element &result);
};

class ClimbKeyExecutor
{
    Goldilocks &fr;
    const Config &config;
    const uint64_t N;

public:
    ClimbKeyExecutor (Goldilocks &fr, const Config &config) :
        fr(fr),
        config(config),
        N(PROVER_FORK_NAMESPACE::ClimbKeyCommitPols::pilDegree()) {}
    void execute (vector<ClimbKeyAction> &input, PROVER_FORK_NAMESPACE::ClimbKeyCommitPols &pols);

    // To be used only for testing, since it allocates a lot of memory
    void execute (vector<ClimbKeyAction> &action);

    static bool check(const Goldilocks::Element &current, const Goldilocks::Element &expected, bool bit);
    static bool calculate(const Goldilocks::Element &current, Goldilocks::Element &expected, bool bit);
};

#endif