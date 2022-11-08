#ifndef EXECUTOR_HPP
#define EXECUTOR_HPP

#include "sm/pols_generated/commit_pols.hpp"
#include "config.hpp"
#include "goldilocks_base_field.hpp"
#include "sm/main/main_executor.hpp"
#include "sm/storage/storage_executor.hpp"
#include "sm/memory/memory_executor.hpp"
#include "sm/binary/binary_executor.hpp"
#include "sm/arith/arith_executor.hpp"
#include "sm/padding_kk/padding_kk_executor.hpp"
#include "sm/padding_kkbit/padding_kkbit_executor.hpp"
#include "sm/nine2one/nine2one_executor.hpp"
#include "sm/keccak_f/keccak_f_executor.hpp"
#include "sm/byte4/byte4_executor.hpp"
#include "sm/padding_pg/padding_pg_executor.hpp"
#include "sm/poseidon_g/poseidon_g_executor.hpp"
#include "sm/mem_align/mem_align_executor.hpp"
#include "prover_request.hpp"

class Executor
{
public:
    Goldilocks &fr;
    const Config &config;
    
    MainExecutor mainExecutor;
    StorageExecutor storageExecutor;
    MemoryExecutor memoryExecutor;
    BinaryExecutor binaryExecutor;
    ArithExecutor arithExecutor;
    PaddingKKExecutor paddingKKExecutor;
    PaddingKKBitExecutor paddingKKBitExecutor;
    Nine2OneExecutor nine2OneExecutor;
    KeccakFExecutor keccakFExecutor;
    Byte4Executor byte4Executor;
    PaddingPGExecutor paddingPGExecutor;
    PoseidonGExecutor poseidonGExecutor;
    MemAlignExecutor memAlignExecutor;

    Executor(Goldilocks &fr, const Config &config, PoseidonGoldilocks &poseidon) :
        fr(fr),
        config(config),
        mainExecutor(fr, poseidon, config),
        storageExecutor(fr, poseidon, config),
        memoryExecutor(fr, config),
        binaryExecutor(fr, config),
        arithExecutor(fr, config),
        paddingKKExecutor(fr),
        paddingKKBitExecutor(fr),
        nine2OneExecutor(fr),
        keccakFExecutor(fr, config),
        byte4Executor(fr),
        paddingPGExecutor(fr, poseidon),
        poseidonGExecutor(fr, poseidon),
        memAlignExecutor(fr, config)
        {};

    // Full version: all polynomials are evaluated, in all evaluations
    void execute (ProverRequest &proverRequest, CommitPols & commitPols);

    // Reduced version: only 2 evaluations are allocated, and assert is disabled
    void process_batch (ProverRequest &proverRequest);
};

#endif