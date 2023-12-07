#include "executor.hpp"
#include "utils.hpp"
#include "main_sm/fork_1/main_exec_generated/main_exec_generated.hpp"
#include "main_sm/fork_1/main_exec_generated/main_exec_generated_fast.hpp"
#include "main_sm/fork_2/main_exec_generated/main_exec_generated.hpp"
#include "main_sm/fork_2/main_exec_generated/main_exec_generated_fast.hpp"
#include "main_sm/fork_3/main_exec_generated/main_exec_generated.hpp"
#include "main_sm/fork_3/main_exec_generated/main_exec_generated_fast.hpp"
#include "main_sm/fork_4/main_exec_generated/main_exec_generated.hpp"
#include "main_sm/fork_4/main_exec_generated/main_exec_generated_fast.hpp"
#include "main_sm/fork_5/main_exec_generated/main_exec_generated.hpp"
#include "main_sm/fork_5/main_exec_generated/main_exec_generated_fast.hpp"
#include "main_sm/fork_5/main_exec_c/main_exec_c.hpp"
#include "main_sm/fork_6/main_exec_generated/main_exec_generated.hpp"
#include "main_sm/fork_6/main_exec_generated/main_exec_generated_fast.hpp"
#include "main_sm/fork_6/main_exec_c/main_exec_c.hpp"
#include "timer.hpp"
#include "zklog.hpp"

// Reduced version: only 1 evaluation is allocated, and some asserts are disabled
void Executor::process_batch (ProverRequest &proverRequest)
{
    // Execute the Main State Machine
    switch (proverRequest.input.publicInputsExtended.publicInputs.forkID)
    {
        case 1: // fork_1
        {
            /*if (config.useMainExecGenerated) // Generated code has been disabled in old forks
            {
                fork_1::main_exec_generated_fast(mainExecutor_fork_1, proverRequest);
            }
            else*/
            {
                //zklog.info("Executor::process_batch() fork 1 native");

                // Allocate committed polynomials for only 1 evaluation
                void * pAddress = calloc(fork_1::CommitPols::numPols()*sizeof(Goldilocks::Element), 1);
                if (pAddress == NULL)
                {
                    zklog.error("Executor::process_batch() failed calling calloc(" + to_string(fork_1::CommitPols::pilSize()) + ")");
                    exitProcess();
                }
                fork_1::CommitPols commitPols(pAddress,1);

                // This instance will store all data required to execute the rest of State Machines
                fork_1::MainExecRequired required;

                mainExecutor_fork_1.execute(proverRequest, commitPols.Main, required);

                // Free committed polynomials address space
                free(pAddress);
            }
            return;
        }
        case 2: // fork_2
        {
            /*if (config.useMainExecGenerated) // Generated code has been disabled in old forks
            {
                fork_2::main_exec_generated_fast(mainExecutor_fork_2, proverRequest);
            }
            else*/
            {
                //zklog.info("Executor::process_batch() fork 2 native");

                // Allocate committed polynomials for only 1 evaluation
                void * pAddress = calloc(fork_2::CommitPols::numPols()*sizeof(Goldilocks::Element), 1);
                if (pAddress == NULL)
                {
                    zklog.error("Executor::process_batch() failed calling calloc(" + to_string(fork_2::CommitPols::pilSize()) + ")");
                    exitProcess();
                }
                fork_2::CommitPols commitPols(pAddress,1);

                // This instance will store all data required to execute the rest of State Machines
                fork_2::MainExecRequired required;

                mainExecutor_fork_2.execute(proverRequest, commitPols.Main, required);

                // Free committed polynomials address space
                free(pAddress);
            }
            return;
        }
        case 3: // fork_3
        {
            /*if (config.useMainExecGenerated) // Generated code has been disabled in old forks
            {
                fork_3::main_exec_generated_fast(mainExecutor_fork_3, proverRequest);
            }
            else*/
            {
                //zklog.info("Executor::process_batch() fork 3 native");

                // Allocate committed polynomials for only 1 evaluation
                void * pAddress = calloc(fork_3::CommitPols::numPols()*sizeof(Goldilocks::Element), 1);
                if (pAddress == NULL)
                {
                    zklog.error("Executor::process_batch() failed calling calloc(" + to_string(fork_3::CommitPols::pilSize()) + ")");
                    exitProcess();
                }
                fork_3::CommitPols commitPols(pAddress,1);

                // This instance will store all data required to execute the rest of State Machines
                fork_3::MainExecRequired required;

                mainExecutor_fork_3.execute(proverRequest, commitPols.Main, required);

                // Free committed polynomials address space
                free(pAddress);
            }
            return;
        }
        case 4: // fork_4
        {
            if (config.useMainExecGenerated)
            {
                //zklog.info("Executor::process_batch() fork 4 generated");
                fork_4::main_exec_generated_fast(mainExecutor_fork_4, proverRequest);
            }
            else
            {
                //zklog.info("Executor::process_batch() fork 4 native");

                // Allocate committed polynomials for only 1 evaluation
                void * pAddress = calloc(fork_4::CommitPols::numPols()*sizeof(Goldilocks::Element), 1);
                if (pAddress == NULL)
                {
                    zklog.error("Executor::process_batch() failed calling calloc(" + to_string(fork_4::CommitPols::pilSize()) + ")");
                    exitProcess();
                }
                fork_4::CommitPols commitPols(pAddress,1);

                // This instance will store all data required to execute the rest of State Machines
                fork_4::MainExecRequired required;

                mainExecutor_fork_4.execute(proverRequest, commitPols.Main, required);

                // Free committed polynomials address space
                free(pAddress);
            }
            return;
        }
        case 5: // fork_5
        {            
            if (config.useMainExecC) // Do not use in production; under development
            {
                //zklog.info("Executor::process_batch() fork 5 C");
                mainExecutorC_fork_5.execute(proverRequest);
            }
            else if (config.useMainExecGenerated)
            {
                //zklog.info("Executor::process_batch() fork 5 generated");
                fork_5::main_exec_generated_fast(mainExecutor_fork_5, proverRequest);
            }
            else
            {
                //zklog.info("Executor::process_batch() fork 5 native");

                // Allocate committed polynomials for only 1 evaluation
                void * pAddress = calloc(fork_5::CommitPols::numPols()*sizeof(Goldilocks::Element), 1);
                if (pAddress == NULL)
                {
                    zklog.error("Executor::process_batch() failed calling calloc(" + to_string(fork_5::CommitPols::pilSize()) + ")");
                    exitProcess();
                }
                fork_5::CommitPols commitPols(pAddress,1);

                // This instance will store all data required to execute the rest of State Machines
                fork_5::MainExecRequired required;

                mainExecutor_fork_5.execute(proverRequest, commitPols.Main, required);

                // Free committed polynomials address space
                free(pAddress);
            }
            return;
        }
        case 6: // fork_6
        {
            if (config.useMainExecC) // Do not use in production; under development
            {
                //zklog.info("Executor::process_batch() fork 6 C");
                mainExecutorC_fork_6.execute(proverRequest);
            }
            else if (config.useMainExecGenerated)
            {
                //zklog.info("Executor::process_batch() fork 5 generated");
                fork_6::main_exec_generated_fast(mainExecutor_fork_6, proverRequest);
            }
            else
            {
                //zklog.info("Executor::process_batch() fork 6 native");

                // Allocate committed polynomials for only 1 evaluation
                void * pAddress = calloc(fork_6::CommitPols::numPols()*sizeof(Goldilocks::Element), 1);
                if (pAddress == NULL)
                {
                    zklog.error("Executor::process_batch() failed calling calloc(" + to_string(fork_6::CommitPols::pilSize()) + ")");
                    exitProcess();
                }
                fork_6::CommitPols commitPols(pAddress,1);

                // This instance will store all data required to execute the rest of State Machines
                fork_6::MainExecRequired required;

                mainExecutor_fork_6.execute(proverRequest, commitPols.Main, required);

                // Free committed polynomials address space
                free(pAddress);
            }
            return;
        }
        default:
        {
            zklog.error("Executor::process_batch() got invalid fork ID=" + to_string(proverRequest.input.publicInputsExtended.publicInputs.forkID));
            proverRequest.result = ZKR_SM_MAIN_INVALID_FORK_ID;
            return;
        }
    }
}

class ExecutorContext
{
public:
    Executor * pExecutor;
    PROVER_FORK_NAMESPACE::MainExecRequired * pRequired;
    PROVER_FORK_NAMESPACE::CommitPols * pCommitPols;
};

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
    pExecutorContext->pExecutor->paddingKKBitExecutor.execute(pExecutorContext->pRequired->PaddingKKBit, pExecutorContext->pCommitPols->PaddingKKBit, pExecutorContext->pRequired->Bits2Field);
    TimerStopAndLog(PADDING_KK_BIT_SM_EXECUTE_THREAD);
    
    // Execute the Poseidon G State Machine
    TimerStart(BITS2FIELD_SM_EXECUTE_THREAD);
    pExecutorContext->pExecutor->bits2FieldExecutor.execute(pExecutorContext->pRequired->Bits2Field, pExecutorContext->pCommitPols->Bits2Field, pExecutorContext->pRequired->KeccakF);
    TimerStopAndLog(BITS2FIELD_SM_EXECUTE_THREAD);
    
    // Execute the Keccak F State Machine
    TimerStart(KECCAK_F_SM_EXECUTE_THREAD);
    pExecutorContext->pExecutor->keccakFExecutor.execute(pExecutorContext->pRequired->KeccakF, pExecutorContext->pCommitPols->KeccakF);
    TimerStopAndLog(KECCAK_F_SM_EXECUTE_THREAD);

    return NULL;
}

// Full version: all polynomials are evaluated, in all evaluations
void Executor::execute (ProverRequest &proverRequest, PROVER_FORK_NAMESPACE::CommitPols & commitPols)
{
    if (!config.executeInParallel)
    {
        // This instance will store all data required to execute the rest of State Machines
        PROVER_FORK_NAMESPACE::MainExecRequired required;

        // Execute the Main State Machine
        TimerStart(MAIN_EXECUTOR_EXECUTE);
        if (proverRequest.input.publicInputsExtended.publicInputs.forkID == PROVER_FORK_ID)
        {
            if (config.useMainExecGenerated)
            {
                PROVER_FORK_NAMESPACE::main_exec_generated(mainExecutor_fork_6, proverRequest, commitPols.Main, required);
            }
            else
            {
                mainExecutor_fork_6.execute(proverRequest, commitPols.Main, required);
            }
            
            // Save input to <timestamp>.input.json after execution including dbReadLog
            if (config.saveDbReadsToFile)
            {
                json inputJsonEx;
                proverRequest.input.save(inputJsonEx, *proverRequest.dbReadLog);
                json2file(inputJsonEx, proverRequest.inputDbFile());
            }
        }
        else
        {
            zklog.error("Executor::execute() got invalid fork ID=" + to_string(proverRequest.input.publicInputsExtended.publicInputs.forkID));
            proverRequest.result = ZKR_SM_MAIN_INVALID_FORK_ID;
        }
        TimerStopAndLog(MAIN_EXECUTOR_EXECUTE);

        if (proverRequest.result != ZKR_SUCCESS)
        {
            return;
        }

        // Execute the Padding PG State Machine
        TimerStart(PADDING_PG_SM_EXECUTE);
        paddingPGExecutor.execute(required.PaddingPG, commitPols.PaddingPG, required.PoseidonG);
        TimerStopAndLog(PADDING_PG_SM_EXECUTE);

        // Execute the Storage State Machine
        TimerStart(STORAGE_SM_EXECUTE);
        storageExecutor.execute(required.Storage, commitPols.Storage, required.PoseidonG);
        TimerStopAndLog(STORAGE_SM_EXECUTE);

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
        paddingKKBitExecutor.execute(required.PaddingKKBit, commitPols.PaddingKKBit, required.Bits2Field);
        TimerStopAndLog(PADDING_KK_BIT_SM_EXECUTE);

        // Execute the Bits2Field State Machine
        TimerStart(BITS2FIELD_SM_EXECUTE);
        bits2FieldExecutor.execute(required.Bits2Field, commitPols.Bits2Field, required.KeccakF);
        TimerStopAndLog(BITS2FIELD_SM_EXECUTE);

        // Execute the Keccak F State Machine
        TimerStart(KECCAK_F_SM_EXECUTE);
        keccakFExecutor.execute(required.KeccakF, commitPols.KeccakF);
        TimerStopAndLog(KECCAK_F_SM_EXECUTE);

        // Execute the PoseidonG State Machine
        TimerStart(POSEIDON_G_SM_EXECUTE);
        poseidonGExecutor.execute(required.PoseidonG, commitPols.PoseidonG);
        TimerStopAndLog(POSEIDON_G_SM_EXECUTE);
    }
    else
    {
        // This instance will store all data required to execute the rest of State Machines
        PROVER_FORK_NAMESPACE::MainExecRequired required;
        ExecutorContext executorContext;
        executorContext.pExecutor = this;
        executorContext.pCommitPols = &commitPols;
        executorContext.pRequired = &required;

        // Execute the Main State Machine
        TimerStart(MAIN_EXECUTOR_EXECUTE);
        if (config.useMainExecGenerated)
        {
            PROVER_FORK_NAMESPACE::main_exec_generated(mainExecutor_fork_6, proverRequest, commitPols.Main, required);
        }
        else
        {
            mainExecutor_fork_6.execute(proverRequest, commitPols.Main, required);
        }
            
        // Save input to <timestamp>.input.json after execution including dbReadLog
        if (config.saveDbReadsToFile)
        {
            json inputJsonEx;
            proverRequest.input.save(inputJsonEx, *proverRequest.dbReadLog);
            json2file(inputJsonEx, proverRequest.inputDbFile());
        }

        TimerStopAndLog(MAIN_EXECUTOR_EXECUTE);

        if (proverRequest.result != ZKR_SUCCESS)
        {
            zklog.error("Executor::execute() got from main execution proverRequest.result=" + to_string(proverRequest.result) + "=" + zkresult2string(proverRequest.result));
            return;
        }

        // Execute the Padding PG, Storage and Poseidon G State Machines
        pthread_t poseidonThread;
        pthread_create(&poseidonThread, NULL, PoseidonThread, &executorContext);

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

        // Execute the PaddingKK, PaddingKKBit, Bits2Field, Keccak F and NormGate9 State Machines
        pthread_t keccakThread;
        pthread_create(&keccakThread, NULL, KeccakThread, &executorContext);

        // Wait for the parallel SM threads
        pthread_join(binaryThread, NULL);
        pthread_join(memAlignThread, NULL);
        pthread_join(memoryThread, NULL);
        pthread_join(arithThread, NULL);
        pthread_join(poseidonThread, NULL);
        pthread_join(keccakThread, NULL);
    }
}