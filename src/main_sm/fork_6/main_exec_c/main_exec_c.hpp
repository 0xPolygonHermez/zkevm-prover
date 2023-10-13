#ifndef MAIN_EXEC_C_HPP_fork_6
#define MAIN_EXEC_C_HPP_fork_6

#include <string>
#include "main_sm/fork_6/main/main_executor.hpp"
#include "fec.hpp"
#include "fnec.hpp"
#include "poseidon_goldilocks.hpp"

namespace fork_6
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

