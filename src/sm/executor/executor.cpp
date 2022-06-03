#include "executor.hpp"
#include "utils.hpp"
#include "sm/generated/main_exec_generated.hpp"
#include "sm/generated/main_exec_generated_fast.hpp"
#include "timer.hpp"

// Fast version: only 2 evaluations are allocated, and only MainCommitPols are evaluated
void Executor::execute_fast (const Input &input, Database &db, Counters &counters )
{
    // Execute the Main State Machine
    TimerStart(EXECUTOR_EXECUTE_FAST);
    if (config.useMainExecGenerated)
    {
        MainExecGeneratedFast(fr, input, db, counters);
    }
    else
    {
        // Allocate an area of memory, to store the main and byte4 committed polynomials,
        // and create them using the allocated address
        void * pMainAddress = malloc(MainCommitPols::size()*2);
        zkassert(pMainAddress!=NULL);
        memset(pMainAddress, 0, MainCommitPols::size()*2);
        MainCommitPols mainCommitPols(pMainAddress,2);

        // This instance will store all data required to execute the rest of State Machines
        MainExecRequired required;

        mainExecutor.execute(input, mainCommitPols, db, counters, required, true);

        // Free committed polynomials address space
        free(pMainAddress);
    }
    TimerStopAndLog(EXECUTOR_EXECUTE_FAST);
}

class ExecutorContext
{
public:
    Executor * pExecutor;
    MainExecRequired * pRequired;
    CommitPols * pCommitPols;
};

void* Byte4Thread (void* arg)
{
    // Get the context
    ExecutorContext * pExecutorContext = (ExecutorContext *)arg;
    
    // Execute the Byte4 State Machine
    TimerStart(BYTE4_SM_EXECUTE_THREAD);
    pExecutorContext->pExecutor->byte4Executor.execute(pExecutorContext->pRequired->Byte4, pExecutorContext->pCommitPols->Byte4);
    TimerStopAndLog(BYTE4_SM_EXECUTE_THREAD);

    return NULL;
}

void* BinaryThread (void* arg)
{
    // Get the context
    ExecutorContext * pExecutorContext = (ExecutorContext *)arg;
    
    // Execute the Binary State Machine
    TimerStart(BINARY_SM_EXECUTE_THREAD);
    pExecutorContext->pExecutor->binaryExecutor.execute(pExecutorContext->pRequired->Binary, pExecutorContext->pCommitPols->Binary);
    TimerStopAndLog(BINARY_SM_EXECUTE_THREAD);

    return NULL;
}

void* MemAlignThread (void* arg)
{
    // Get the context
    ExecutorContext * pExecutorContext = (ExecutorContext *)arg;
    
    // Execute the MemAlign State Machine
    TimerStart(MEM_ALIGN_SM_EXECUTE_THREAD);
    pExecutorContext->pExecutor->memAlignExecutor.execute(pExecutorContext->pRequired->MemAlign, pExecutorContext->pCommitPols->MemAlign);
    TimerStopAndLog(MEM_ALIGN_SM_EXECUTE_THREAD);

    return NULL;
}

void* MemoryThread (void* arg)
{
    // Get the context
    ExecutorContext * pExecutorContext = (ExecutorContext *)arg;
    
    // Execute the Binary State Machine
    TimerStart(MEMORY_SM_EXECUTE_THREAD);
    pExecutorContext->pExecutor->memoryExecutor.execute(pExecutorContext->pRequired->Memory, pExecutorContext->pCommitPols->Mem);
    TimerStopAndLog(MEMORY_SM_EXECUTE_THREAD);

    return NULL;
}

void* ArithThread (void* arg)
{
    // Get the context
    ExecutorContext * pExecutorContext = (ExecutorContext *)arg;
    
    // Execute the Binary State Machine
    TimerStart(ARITH_SM_EXECUTE_THREAD);
    pExecutorContext->pExecutor->arithExecutor.execute(pExecutorContext->pRequired->Arith, pExecutorContext->pCommitPols->Arith);
    TimerStopAndLog(ARITH_SM_EXECUTE_THREAD);

    return NULL;
}

void* PoseidonThread (void* arg)
{
    // Get the context
    ExecutorContext * pExecutorContext = (ExecutorContext *)arg;
    
    // Execute the Padding PG State Machine
    TimerStart(PADDING_PG_SM_EXECUTE_THREAD);
    pExecutorContext->pExecutor->paddingPGExecutor.execute(pExecutorContext->pRequired->PaddingPG, pExecutorContext->pCommitPols->PaddingPG, pExecutorContext->pRequired->PoseidonG);
    TimerStopAndLog(PADDING_PG_SM_EXECUTE_THREAD);
    
    // Execute the Storage State Machine
    TimerStart(STORAGE_SM_EXECUTE_THREAD);
    pExecutorContext->pExecutor->storageExecutor.execute(pExecutorContext->pRequired->Storage, pExecutorContext->pCommitPols->Storage, pExecutorContext->pRequired->PoseidonG);
    TimerStopAndLog(STORAGE_SM_EXECUTE_THREAD);
    
    // Execute the Poseidon G State Machine
    TimerStart(POSEIDON_G_SM_EXECUTE_THREAD);
    pExecutorContext->pExecutor->poseidonGExecutor.execute(pExecutorContext->pRequired->PoseidonG, pExecutorContext->pCommitPols->PoseidonG);
    TimerStopAndLog(POSEIDON_G_SM_EXECUTE_THREAD);

    return NULL;
}

void* KeccakThread (void* arg)
{
    // Get the context
    ExecutorContext * pExecutorContext = (ExecutorContext *)arg;
    
    // Execute the Padding KK State Machine
    TimerStart(PADDING_KK_SM_EXECUTE_THREAD);
    pExecutorContext->pExecutor->paddingKKExecutor.execute(pExecutorContext->pRequired->PaddingKK, pExecutorContext->pCommitPols->PaddingKK, pExecutorContext->pRequired->PaddingKKBit);
    TimerStopAndLog(PADDING_KK_SM_EXECUTE_THREAD);
    
    // Execute the PaddingKKBit State Machine
    TimerStart(PADDING_KK_BIT_SM_EXECUTE_THREAD);
    pExecutorContext->pExecutor->paddingKKBitExecutor.execute(pExecutorContext->pRequired->PaddingKKBit, pExecutorContext->pCommitPols->PaddingKKBit, pExecutorContext->pRequired->Nine2One);
    TimerStopAndLog(PADDING_KK_BIT_SM_EXECUTE_THREAD);
    
    // Execute the Poseidon G State Machine
    TimerStart(NINE2ONE_SM_EXECUTE_THREAD);
    pExecutorContext->pExecutor->nine2OneExecutor.execute(pExecutorContext->pRequired->Nine2One, pExecutorContext->pCommitPols->Nine2One, pExecutorContext->pRequired->KeccakF);
    TimerStopAndLog(NINE2ONE_SM_EXECUTE_THREAD);
    
    // Execute the Keccak F State Machine
    TimerStart(KECCAK_F_SM_EXECUTE_THREAD);
    pExecutorContext->pExecutor->keccakFExecutor.execute(pExecutorContext->pRequired->KeccakF, pExecutorContext->pCommitPols->KeccakF, pExecutorContext->pRequired->NormGate9);
    TimerStopAndLog(KECCAK_F_SM_EXECUTE_THREAD);
    
    // Execute the NormGate9 State Machine
    TimerStart(NORM_GATE_9_SM_EXECUTE_THREAD);
    pExecutorContext->pExecutor->normGate9Executor.execute(pExecutorContext->pRequired->NormGate9, pExecutorContext->pCommitPols->NormGate9);
    TimerStopAndLog(NORM_GATE_9_SM_EXECUTE_THREAD);

    return NULL;
}

// Full version: all polynomials are evaluated, in all evaluations
void Executor::execute (const Input &input, CommitPols & commitPols, Database &db, Counters &counters)
{
    if (!config.executeInParallel)
    {
        // This instance will store all data required to execute the rest of State Machines
        MainExecRequired required;

        // Execute the Main State Machine
        TimerStart(MAIN_EXECUTOR_EXECUTE);
        if (config.useMainExecGenerated)
        {
            MainExecGenerated(fr, input, commitPols.Main, db, counters, required);
        }
        else
        {
            mainExecutor.execute(input, commitPols.Main, db, counters, required);
        }
        TimerStopAndLog(MAIN_EXECUTOR_EXECUTE);

        // Execute the Padding PG State Machine
        TimerStart(PADDING_PG_SM_EXECUTE);
        paddingPGExecutor.execute(required.PaddingPG, commitPols.PaddingPG, required.PoseidonG);
        TimerStopAndLog(PADDING_PG_SM_EXECUTE);

        // Execute the Storage State Machine
        TimerStart(STORAGE_SM_EXECUTE);
        storageExecutor.execute(required.Storage, commitPols.Storage, required.PoseidonG);
        TimerStopAndLog(STORAGE_SM_EXECUTE);

        // Execute the Byte4 State Machine
        TimerStart(BYTE4_SM_EXECUTE);
        byte4Executor.execute(required.Byte4, commitPols.Byte4);
        TimerStopAndLog(BYTE4_SM_EXECUTE);

        // Execute the Arith State Machine
        TimerStart(ARITH_SM_EXECUTE);
        arithExecutor.execute(required.Arith, commitPols.Arith);
        TimerStopAndLog(ARITH_SM_EXECUTE);

        // Execute the Binary State Machine
        TimerStart(BINARY_SM_EXECUTE);
        binaryExecutor.execute(required.Binary, commitPols.Binary);
        TimerStopAndLog(BINARY_SM_EXECUTE);

        // Execute the MemAlign State Machine
        TimerStart(MEM_ALIGN_SM_EXECUTE);
        memAlignExecutor.execute(required.MemAlign, commitPols.MemAlign);
        TimerStopAndLog(MEM_ALIGN_SM_EXECUTE);
        
        // Execute the Memory State Machine
        TimerStart(MEMORY_SM_EXECUTE);
        memoryExecutor.execute(required.Memory, commitPols.Mem);
        TimerStopAndLog(MEMORY_SM_EXECUTE);

        // Execute the PaddingKK State Machine
        TimerStart(PADDING_KK_SM_EXECUTE);
        paddingKKExecutor.execute(required.PaddingKK, commitPols.PaddingKK, required.PaddingKKBit);
        TimerStopAndLog(PADDING_KK_SM_EXECUTE);

        // Execute the PaddingKKBit State Machine
        TimerStart(PADDING_KK_BIT_SM_EXECUTE);
        paddingKKBitExecutor.execute(required.PaddingKKBit, commitPols.PaddingKKBit, required.Nine2One);
        TimerStopAndLog(PADDING_KK_BIT_SM_EXECUTE);

        // Execute the Nine2One State Machine
        TimerStart(NINE2ONE_SM_EXECUTE);
        nine2OneExecutor.execute(required.Nine2One, commitPols.Nine2One, required.KeccakF);
        TimerStopAndLog(NINE2ONE_SM_EXECUTE);

        // Execute the Keccak F State Machine
        TimerStart(KECCAK_F_SM_EXECUTE);
        keccakFExecutor.execute(required.KeccakF, commitPols.KeccakF, required.NormGate9);
        TimerStopAndLog(KECCAK_F_SM_EXECUTE);

        // Execute the NormGate9 State Machine
        TimerStart(NORM_GATE_9_SM_EXECUTE);
        normGate9Executor.execute(required.NormGate9, commitPols.NormGate9);
        TimerStopAndLog(NORM_GATE_9_SM_EXECUTE);

        // Execute the PoseidonG State Machine
        TimerStart(POSEIDON_G_SM_EXECUTE);
        poseidonGExecutor.execute(required.PoseidonG, commitPols.PoseidonG);
        TimerStopAndLog(POSEIDON_G_SM_EXECUTE);
    }
    else
    {

        // This instance will store all data required to execute the rest of State Machines
        MainExecRequired required;
        ExecutorContext executorContext;
        executorContext.pExecutor = this;
        executorContext.pCommitPols = &commitPols;
        executorContext.pRequired = &required;

        // Execute the Main State Machine
        TimerStart(MAIN_EXECUTOR_EXECUTE);
        mainExecutor.execute(input, commitPols.Main, db, counters, required);
        TimerStopAndLog(MAIN_EXECUTOR_EXECUTE);

        // Execute the Padding PG, Storage and Poseidon G State Machines
        pthread_t poseidonThread;
        pthread_create(&poseidonThread, NULL, PoseidonThread, &executorContext);

        // Execute the Byte4 State Machine, in parallel
        pthread_t byte4Thread;
        pthread_create(&byte4Thread, NULL, Byte4Thread, &executorContext);

        // Execute the Arith State Machine, in parallel
        pthread_t arithThread;
        pthread_create(&arithThread, NULL, ArithThread, &executorContext);

        // Execute the Binary State Machine, in parallel
        pthread_t binaryThread;
        pthread_create(&binaryThread, NULL, BinaryThread, &executorContext);

        // Execute the Mem Align State Machine, in parallel
        pthread_t memAlignThread;
        pthread_create(&memAlignThread, NULL, MemAlignThread, &executorContext);
        
        // Execute the Memory State Machine, in parallel
        pthread_t memoryThread;
        pthread_create(&memoryThread, NULL, MemoryThread, &executorContext);

        // Execute the PaddingKK, PaddingKKBit, Nine2One, Keccak F and NormGate9 State Machines
        pthread_t keccakThread;
        pthread_create(&keccakThread, NULL, KeccakThread, &executorContext);

        // Wait for the parallel SM threads
        pthread_join(byte4Thread, NULL);
        pthread_join(binaryThread, NULL);
        pthread_join(memAlignThread, NULL);
        pthread_join(memoryThread, NULL);
        pthread_join(arithThread, NULL);
        pthread_join(poseidonThread, NULL);
        pthread_join(keccakThread, NULL);
    }
}