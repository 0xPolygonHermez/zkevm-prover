#ifndef STORAGE_SM_HPP
#define STORAGE_SM_HPP

#include "definitions.hpp"
#include "config.hpp"
#include "storage_rom.hpp"
#include "smt_action.hpp"
#include "smt_action_context.hpp"
#include "goldilocks_base_field.hpp"
#include "poseidon_goldilocks.hpp"
#include "utils.hpp"
#include "sm/pols_generated/commit_pols.hpp"

USING_PROVER_FORK_NAMESPACE;

class StorageExecutor
{
    Goldilocks &fr;
    PoseidonGoldilocks &poseidon;
    const Config &config;
    const uint64_t N;
    StorageRom rom;

public:
    StorageExecutor (Goldilocks &fr, PoseidonGoldilocks &poseidon, const Config &config) :
        fr(fr),
        poseidon(poseidon),
        config(config),
        N(PROVER_FORK_NAMESPACE::StorageCommitPols::pilDegree())
    {
        // Init rom from file
        json romJson;
        file2json(config.storageRomFile, romJson);
        rom.load(romJson);
    }

    // To be used by prover
    void execute (vector<SmtAction> &action, PROVER_FORK_NAMESPACE::StorageCommitPols &pols, vector<array<Goldilocks::Element, 17>> &required);

    // To be used only for testing, since it allocates a lot of memory
    void execute (vector<SmtAction> &action);
};

#endif