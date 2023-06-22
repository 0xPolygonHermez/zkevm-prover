#ifndef MAIN_EXEC_C_HPP_fork_5
#define MAIN_EXEC_C_HPP_fork_5

#include <string>
#include "main_sm/fork_5/main/main_executor.hpp"
#include "main_sm/fork_5/main_exec_c/context_c.hpp"

namespace fork_5
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

