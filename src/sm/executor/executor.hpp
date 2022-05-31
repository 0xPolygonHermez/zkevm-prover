#ifndef EXECUTOR_HPP
#define EXECUTOR_HPP

#include "sm/pil/commit_pols.hpp"
#include "config.hpp"
#include "ff/ff.hpp"
#include "sm/main/main_executor.hpp"
#include "sm/storage/storage_executor.hpp"
#include "sm/memory/memory_executor.hpp"
#include "sm/binary/binary_executor.hpp"
#include "sm/arith/arith_executor.hpp"
#include "sm/padding_kk/padding_kk_executor.hpp"
#include "sm/padding_kkbit/padding_kkbit_executor.hpp"
#include "sm/nine2one/nine2one_executor.hpp"
#include "sm/keccak_f/keccak_f_executor.hpp"
#include "sm/norm_gate9/norm_gate9_executor.hpp"
#include "sm/byte4/byte4_executor.hpp"
#include "sm/padding_pg/padding_pg_executor.hpp"
#include "sm/poseidon_g/poseidon_g_executor.hpp"

class Executor
{
public:
    FiniteField &fr;
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
    NormGate9Executor normGate9Executor;
    Byte4Executor byte4Executor;
    PaddingPGExecutor paddingPGExecutor;
    PoseidonGExecutor poseidonGExecutor;

    Executor(FiniteField &fr, const Config &config, Poseidon_goldilocks &poseidon, const Rom &rom) :
        fr(fr),
        config(config),
        mainExecutor(fr, poseidon, rom, config),
        storageExecutor(fr, poseidon, config),
        memoryExecutor(fr, config),
        binaryExecutor(fr, config),
        arithExecutor(fr, config),
        paddingKKExecutor(fr),
        nine2OneExecutor(fr),
        keccakFExecutor(config),
        paddingPGExecutor(fr, poseidon),
        poseidonGExecutor(fr, poseidon)
        {};

    // Full version: all polynomials are evaluated, in all evaluations
    void execute(const Input &input, CommitPols & commitPols, Database &db, Counters &counters);

    // Fast version: only 2 evaluations are allocated, and only MainCommitPols are evaluated
    void execute_fast(const Input &input, Database &db, Counters &counters );
};

#endif