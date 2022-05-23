#ifndef STORAGE_SM_HPP
#define STORAGE_SM_HPP

#include "config.hpp"
#include "storage_rom.hpp"
#include "smt_action.hpp"
#include "smt_action_context.hpp"
#include "ff/ff.hpp"
#include "poseidon_opt/poseidon_goldilocks.hpp"
#include "utils.hpp"

class StorageExecutor
{
    FiniteField &fr;
    Poseidon_goldilocks &poseidon;
    const Config &config;
    StorageRom rom;
    json pilJson;

public:
    StorageExecutor (FiniteField &fr, Poseidon_goldilocks &poseidon, const Config &config) : fr(fr), poseidon(poseidon), config(config)
    {
        // Init rom from file
        json romJson;
        file2json(config.storageRomFile, romJson);
        rom.load(romJson);

        // Parse PIL json file into memory
        file2json(config.storagePilFile, pilJson);
    }

    // To be used by prover
    void execute (vector<SmtAction> &action, StorageCommitPols &pols, vector<array<FieldElement, 16>> &required);

    // To be used only for testing, since it allocates a lot of memory
    void execute (vector<SmtAction> &action);
};

#endif