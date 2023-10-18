#ifndef EXECUTOR_HPP
#define EXECUTOR_HPP

#include "config.hpp"
#include "goldilocks_base_field.hpp"
#include "main_sm/fork_1/main/main_executor.hpp"
#include "main_sm/fork_2/main/main_executor.hpp"
#include "main_sm/fork_3/main/main_executor.hpp"
#include "main_sm/fork_4/main/main_executor.hpp"
#include "main_sm/fork_5/main/main_executor.hpp"
#include "main_sm/fork_6/main/main_executor.hpp"
#include "main_sm/fork_5/main_exec_c/main_exec_c.hpp"
#include "main_sm/fork_6/main_exec_c/main_exec_c.hpp"
#include "sm/storage/storage_executor.hpp"
#include "sm/memory/memory_executor.hpp"
#include "sm/binary/binary_executor.hpp"
#include "sm/arith/arith_executor.hpp"
#include "sm/padding_kk/padding_kk_executor.hpp"
#include "sm/padding_kkbit/padding_kkbit_executor.hpp"
#include "sm/bits2field/bits2field_executor.hpp"
#include "sm/keccak_f/keccak_f_executor.hpp"
#include "sm/padding_pg/padding_pg_executor.hpp"
#include "sm/poseidon_g/poseidon_g_executor.hpp"
#include "sm/mem_align/mem_align_executor.hpp"
#include "prover_request.hpp"

class Executor
{
public:
    Goldilocks &fr;
    const Config &config;
    
    fork_1::MainExecutor mainExecutor_fork_1;
    fork_2::MainExecutor mainExecutor_fork_2;
    fork_3::MainExecutor mainExecutor_fork_3;
    fork_4::MainExecutor mainExecutor_fork_4;
    fork_5::MainExecutor mainExecutor_fork_5;
    fork_6::MainExecutor mainExecutor_fork_6;
    fork_5::MainExecutorC mainExecutorC_fork_5;
    fork_6::MainExecutorC mainExecutorC_fork_6;
    StorageExecutor storageExecutor;
    MemoryExecutor memoryExecutor;
    BinaryExecutor binaryExecutor;
    ArithExecutor arithExecutor;
    PaddingKKExecutor paddingKKExecutor;
    PaddingKKBitExecutor paddingKKBitExecutor;
    Bits2FieldExecutor bits2FieldExecutor;
    KeccakFExecutor keccakFExecutor;
    PaddingPGExecutor paddingPGExecutor;
    PoseidonGExecutor poseidonGExecutor;
    MemAlignExecutor memAlignExecutor;

    Executor(Goldilocks &fr, const Config &config, PoseidonGoldilocks &poseidon) :
        fr(fr),
        config(config),
        mainExecutor_fork_1(fr, poseidon, config),
        mainExecutor_fork_2(fr, poseidon, config),
        mainExecutor_fork_3(fr, poseidon, config),
        mainExecutor_fork_4(fr, poseidon, config),
        mainExecutor_fork_5(fr, poseidon, config),
        mainExecutor_fork_6(fr, poseidon, config),
        mainExecutorC_fork_5(mainExecutor_fork_5),
        mainExecutorC_fork_6(mainExecutor_fork_6),
        storageExecutor(fr, poseidon, config),
        memoryExecutor(fr, config),
        binaryExecutor(fr, config),
        arithExecutor(fr, config),
        paddingKKExecutor(fr),
        paddingKKBitExecutor(fr),
        bits2FieldExecutor(fr),
        keccakFExecutor(fr, config),
        paddingPGExecutor(fr, poseidon),
        poseidonGExecutor(fr, poseidon),
        memAlignExecutor(fr, config)
        {};

    // Full version: all polynomials are evaluated, in all evaluations
    void execute (ProverRequest &proverRequest, PROVER_FORK_NAMESPACE::CommitPols & commitPols);

    // Reduced version: only 2 evaluations are allocated, and assert is disabled
    void process_batch (ProverRequest &proverRequest);
};

#endif