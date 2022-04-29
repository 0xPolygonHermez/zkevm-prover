#ifndef STORAGE_SM_HPP
#define STORAGE_SM_HPP

#include "config.hpp"
#include "smt_action.hpp"
#include "smt_action_context.hpp"
#include "ff/ff.hpp"
#include "poseidon_opt/poseidon_goldilocks.hpp"

class StorageExecutor
{
    FiniteField &fr;
    Poseidon_goldilocks &poseidon;
    const Config &config;
public:
    StorageExecutor (FiniteField &fr, Poseidon_goldilocks &poseidon, const Config &config) : fr(fr), poseidon(poseidon), config(config) {;}
    void execute (vector<SmtAction> &action);
};

#endif