#ifndef MAIN_EXEC_C_HPP_fork_8
#define MAIN_EXEC_C_HPP_fork_8

#include <string>
#include "main_sm/fork_8/main/main_executor.hpp"
#include "fec.hpp"
#include "fnec.hpp"
#include "poseidon_goldilocks.hpp"

namespace fork_8
{

class MainExecutorC
{
    // Finite field data
    Goldilocks fr; // Finite field reference

    // RawFec instance
    RawFec fec;

    // RawFnec instance
    RawFnec fnec;

    MainExecutor &mainExecutor;

    // Poseidon instance
    PoseidonGoldilocks &poseidon;

    const Config &config;

public:
    MainExecutorC(MainExecutor &mainExecutor) : mainExecutor(mainExecutor), poseidon(mainExecutor.poseidon), config(mainExecutor.config)
    {
    };

    void execute (ProverRequest &proverRequest);
};

}

#endif

