#include "definitions.hpp"
#include "executor.hpp"
#include "utils.hpp"
#include "main_sm/fork_4/main_exec_generated/main_exec_generated_fast.hpp"
#include "main_sm/fork_5/main_exec_generated/main_exec_generated_fast.hpp"
#include "main_sm/fork_6/main_exec_generated/main_exec_generated_fast.hpp"
#include "main_sm/fork_7/main_exec_generated/main_exec_generated_fast.hpp"
#include "main_sm/fork_8/main_exec_generated/main_exec_generated_fast.hpp"
#include "main_sm/fork_9/main_exec_generated/main_exec_generated_fast.hpp"
#include "main_sm/fork_10/main_exec_generated/main_exec_generated_10_fast.hpp"
#include "main_sm/fork_10/main_exec_generated/main_exec_generated_11_fast.hpp"
#include "main_sm/fork_12/main_exec_generated/main_exec_generated_12_fast.hpp"
#include "main_sm/fork_13/main_exec_generated/main_exec_generated_13_fast.hpp"
#include "main_sm/fork_13/main_exec_generated/main_exec_generated_13.hpp"

#include "timer.hpp"
#include "zklog.hpp"

// Reduced version: only 1 evaluation is allocated, and some asserts are disabled
void Executor::processBatch (ProverRequest &proverRequest)
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
                //zklog.info("Executor::processBatch() fork 1 native");

                // Allocate committed polynomials for only 1 evaluation
                void * pAddress = calloc(fork_1::CommitPols::numPols()*sizeof(Goldilocks::Element), 1);
                if (pAddress == NULL)
                {
                    zklog.error("Executor::processBatch() failed calling calloc(" + to_string(fork_1::CommitPols::numPols()*sizeof(Goldilocks::Element)) + ")");
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
                //zklog.info("Executor::processBatch() fork 2 native");

                // Allocate committed polynomials for only 1 evaluation
                void * pAddress = calloc(fork_2::CommitPols::numPols()*sizeof(Goldilocks::Element), 1);
                if (pAddress == NULL)
                {
                    zklog.error("Executor::processBatch() failed calling calloc(" + to_string(fork_2::CommitPols::numPols()*sizeof(Goldilocks::Element)) + ")");
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
                //zklog.info("Executor::processBatch() fork 3 native");

                // Allocate committed polynomials for only 1 evaluation
                void * pAddress = calloc(fork_3::CommitPols::numPols()*sizeof(Goldilocks::Element), 1);
                if (pAddress == NULL)
                {
                    zklog.error("Executor::processBatch() failed calling calloc(" + to_string(fork_3::CommitPols::numPols()*sizeof(Goldilocks::Element)) + ")");
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
#ifdef MAIN_SM_EXECUTOR_GENERATED_CODE
            if (config.useMainExecGenerated)
            {
                //zklog.info("Executor::processBatch() fork 4 generated");
                fork_4::main_exec_generated_fast(mainExecutor_fork_4, proverRequest);
            }
            else
#endif
            {
                //zklog.info("Executor::processBatch() fork 4 native");

                // Allocate committed polynomials for only 1 evaluation
                void * pAddress = calloc(fork_4::CommitPols::numPols()*sizeof(Goldilocks::Element), 1);
                if (pAddress == NULL)
                {
                    zklog.error("Executor::processBatch() failed calling calloc(" + to_string(fork_4::CommitPols::numPols()*sizeof(Goldilocks::Element)) + ")");
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
#ifdef MAIN_SM_EXECUTOR_GENERATED_CODE
            if (config.useMainExecGenerated)
            {
                //zklog.info("Executor::processBatch() fork 5 generated");
                fork_5::main_exec_generated_fast(mainExecutor_fork_5, proverRequest);
            }
            else
#endif
            {
                //zklog.info("Executor::processBatch() fork 5 native");

                // Allocate committed polynomials for only 1 evaluation
                void * pAddress = calloc(fork_5::CommitPols::numPols()*sizeof(Goldilocks::Element), 1);
                if (pAddress == NULL)
                {
                    zklog.error("Executor::processBatch() failed calling calloc(" + to_string(fork_5::CommitPols::numPols()*sizeof(Goldilocks::Element)) + ")");
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
#ifdef MAIN_SM_EXECUTOR_GENERATED_CODE
            if (config.useMainExecGenerated)
            {
                //zklog.info("Executor::processBatch() fork 6 generated");
                fork_6::main_exec_generated_fast(mainExecutor_fork_6, proverRequest);
            }
            else
#endif
            {
                //zklog.info("Executor::processBatch() fork 6 native");

                // Allocate committed polynomials for only 1 evaluation
                void * pAddress = calloc(fork_6::CommitPols::numPols()*sizeof(Goldilocks::Element), 1);
                if (pAddress == NULL)
                {
                    zklog.error("Executor::processBatch() failed calling calloc(" + to_string(fork_6::CommitPols::numPols()*sizeof(Goldilocks::Element)) + ")");
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
        case 7: // fork_7
        {
#ifdef MAIN_SM_EXECUTOR_GENERATED_CODE
            if (config.useMainExecGenerated)
            {
                //zklog.info("Executor::processBatch() fork 7 generated");
                fork_7::main_exec_generated_fast(mainExecutor_fork_7, proverRequest);
            }
            else
#endif
            {
                //zklog.info("Executor::processBatch() fork 7 native");

                // Allocate committed polynomials for only 1 evaluation
                void * pAddress = calloc(fork_7::CommitPols::numPols()*sizeof(Goldilocks::Element), 1);
                if (pAddress == NULL)
                {
                    zklog.error("Executor::processBatch() failed calling calloc(" + to_string(fork_7::CommitPols::numPols()*sizeof(Goldilocks::Element)) + ")");
                    exitProcess();
                }
                fork_7::CommitPols commitPols(pAddress,1);

                // This instance will store all data required to execute the rest of State Machines
                fork_7::MainExecRequired required;

                mainExecutor_fork_7.execute(proverRequest, commitPols.Main, required);

                // Free committed polynomials address space
                free(pAddress);
            }
            return;
        }
        case 8: // fork_8
        {
#ifdef MAIN_SM_EXECUTOR_GENERATED_CODE
            if (config.useMainExecGenerated)
            {
                //zklog.info("Executor::processBatch() fork 8 generated");
                fork_8::main_exec_generated_fast(mainExecutor_fork_8, proverRequest);
            }
            else
#endif
            {
                //zklog.info("Executor::processBatch() fork 8 native");

                // Allocate committed polynomials for only 1 evaluation
                void * pAddress = calloc(fork_8::CommitPols::numPols()*sizeof(Goldilocks::Element), 1);
                if (pAddress == NULL)
                {
                    zklog.error("Executor::processBatch() failed calling calloc(" + to_string(fork_8::CommitPols::numPols()*sizeof(Goldilocks::Element)) + ")");
                    exitProcess();
                }
                fork_8::CommitPols commitPols(pAddress,1);

                // This instance will store all data required to execute the rest of State Machines
                fork_8::MainExecRequired required;

                mainExecutor_fork_8.execute(proverRequest, commitPols.Main, required);

                // Free committed polynomials address space
                free(pAddress);
            }
            return;
        }
        case 9: // fork_9
        {
#ifdef MAIN_SM_EXECUTOR_GENERATED_CODE
            if (config.useMainExecGenerated)
            {
                //TimerStart(MAIN_EXEC_GENERATED_FAST);
                //zklog.info("Executor::processBatch() fork 9 generated");
                fork_9::main_exec_generated_fast(mainExecutor_fork_9, proverRequest);
                //TimerStopAndLog(MAIN_EXEC_GENERATED_FAST);
            }
            else
#endif
            {
                //zklog.info("Executor::processBatch() fork 9 native");

                // Allocate committed polynomials for only 1 evaluation
                void * pAddress = calloc(fork_9::CommitPols::numPols()*sizeof(Goldilocks::Element), 1);
                if (pAddress == NULL)
                {
                    zklog.error("Executor::processBatch() failed calling calloc(" + to_string(fork_9::CommitPols::numPols()*sizeof(Goldilocks::Element)) + ")");
                    exitProcess();
                }
                fork_9::CommitPols commitPols(pAddress,1);

                // This instance will store all data required to execute the rest of State Machines
                fork_9::MainExecRequired required;

                mainExecutor_fork_9.execute(proverRequest, commitPols.Main, required);

                // Free committed polynomials address space
                free(pAddress);
            }
            return;
        }
        case 10: // fork_10
        {
#ifdef MAIN_SM_EXECUTOR_GENERATED_CODE
            if (config.useMainExecGenerated)
            {
                TimerStart(MAIN_EXEC_GENERATED_FAST);
                //zklog.info("Executor::processBatch() fork 10 generated");
                fork_10::main_exec_generated_10_fast(mainExecutor_fork_10, proverRequest);
                TimerStopAndLog(MAIN_EXEC_GENERATED_FAST);
            }
            else
#endif
            {
                //zklog.info("Executor::processBatch() fork 10 native");

                // Allocate committed polynomials for only 1 evaluation
                void * pAddress = calloc(fork_10::CommitPols::numPols()*sizeof(Goldilocks::Element), 1);
                if (pAddress == NULL)
                {
                    zklog.error("Executor::processBatch() failed calling calloc(" + to_string(fork_10::CommitPols::numPols()*sizeof(Goldilocks::Element)) + ")");
                    exitProcess();
                }
                fork_10::CommitPols commitPols(pAddress,1);

                // This instance will store all data required to execute the rest of State Machines
                fork_10::MainExecRequired required;

                mainExecutor_fork_10.execute(proverRequest, commitPols.Main, required);

                // Free committed polynomials address space
                free(pAddress);
            }
            return;
        }
        case 11: // fork_11
        {
#ifdef MAIN_SM_EXECUTOR_GENERATED_CODE
            if (config.useMainExecGenerated)
            {
                TimerStart(MAIN_EXEC_GENERATED_FAST);
                //zklog.info("Executor::processBatch() fork 10 generated");
                fork_10::main_exec_generated_11_fast(mainExecutor_fork_10, proverRequest);
                TimerStopAndLog(MAIN_EXEC_GENERATED_FAST);
            }
            else
#endif
            {
                //zklog.info("Executor::processBatch() fork 10 native");

                // Allocate committed polynomials for only 1 evaluation
                void * pAddress = calloc(fork_10::CommitPols::numPols()*sizeof(Goldilocks::Element), 1);
                if (pAddress == NULL)
                {
                    zklog.error("Executor::processBatch() failed calling calloc(" + to_string(fork_10::CommitPols::numPols()*sizeof(Goldilocks::Element)) + ")");
                    exitProcess();
                }
                fork_10::CommitPols commitPols(pAddress,1);

                // This instance will store all data required to execute the rest of State Machines
                fork_10::MainExecRequired required;

                mainExecutor_fork_10.execute(proverRequest, commitPols.Main, required);

                // Free committed polynomials address space
                free(pAddress);
            }
            return;
        }
        case 12: // fork_12
        {
#ifdef MAIN_SM_EXECUTOR_GENERATED_CODE
            if (config.useMainExecGenerated)
            {
                TimerStart(MAIN_EXEC_GENERATED_FAST);
                //zklog.info("Executor::processBatch() fork 12 generated");
                fork_12::main_exec_generated_12_fast(mainExecutor_fork_12, proverRequest);
                TimerStopAndLog(MAIN_EXEC_GENERATED_FAST);
            }
            else
#endif
            {
                //zklog.info("Executor::processBatch() fork 12 native");

                // Allocate committed polynomials for only 1 evaluation
                void * pAddress = calloc(fork_12::CommitPols::numPols()*sizeof(Goldilocks::Element), 1);
                if (pAddress == NULL)
                {
                    zklog.error("Executor::processBatch() failed calling calloc(" + to_string(fork_12::CommitPols::numPols()*sizeof(Goldilocks::Element)) + ")");
                    exitProcess();
                }
                fork_12::CommitPols commitPols(pAddress,1);

                // This instance will store all data required to execute the rest of State Machines
                fork_12::MainExecRequired required;

                mainExecutor_fork_12.execute(proverRequest, commitPols.Main, required);

                // Free committed polynomials address space
                free(pAddress);
            }
            return;
        }
        case 13: // fork_13
        {
#ifdef MAIN_SM_EXECUTOR_GENERATED_CODE
            if (config.useMainExecGenerated)
            {
                TimerStart(MAIN_EXEC_GENERATED_FAST);
                //zklog.info("Executor::processBatch() fork 13 generated");
                fork_13::main_exec_generated_13_fast(mainExecutor_fork_13, proverRequest);
                TimerStopAndLog(MAIN_EXEC_GENERATED_FAST);
            }
            else
#endif
            {
                //zklog.info("Executor::processBatch() fork 13 native");

                // Allocate committed polynomials for only 1 evaluation
                void * pAddress = calloc(fork_13::CommitPols::numPols()*sizeof(Goldilocks::Element), 1);
                if (pAddress == NULL)
                {
                    zklog.error("Executor::processBatch() failed calling calloc(" + to_string(fork_13::CommitPols::numPols()*sizeof(Goldilocks::Element)) + ")");
                    exitProcess();
                }
                fork_13::CommitPols commitPols(pAddress,1);

                // This instance will store all data required to execute the rest of State Machines
                fork_13::MainExecRequired required;

                mainExecutor_fork_13.execute(proverRequest, commitPols.Main, required);

                // Free committed polynomials address space
                free(pAddress);
            }
            return;
        }
        
        default:
        {
            zklog.error("Executor::processBatch() got invalid fork ID=" + to_string(proverRequest.input.publicInputsExtended.publicInputs.forkID));
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

void* PaddingPGThread (void* arg)
{
    // Get the context
    ExecutorContext * pExecutorContext = (ExecutorContext *)arg;

    // Execute the Padding PG State Machine
    TimerStart(PADDING_PG_SM_EXECUTE_THREAD);
    pExecutorContext->pExecutor->paddingPGExecutor.execute(pExecutorContext->pRequired->PaddingPG, pExecutorContext->pCommitPols->PaddingPG, pExecutorContext->pRequired->PoseidonGFromPG);
    TimerStopAndLog(PADDING_PG_SM_EXECUTE_THREAD);

    return NULL;
}

void* StorageThread (void* arg)
{
    // Get the context
    ExecutorContext * pExecutorContext = (ExecutorContext *)arg;

    // Execute the Storage State Machine
    TimerStart(STORAGE_SM_EXECUTE_THREAD);
    pExecutorContext->pExecutor->storageExecutor.execute(pExecutorContext->pRequired->Storage, pExecutorContext->pCommitPols->Storage, pExecutorContext->pRequired->PoseidonGFromST, pExecutorContext->pRequired->ClimbKey);
    TimerStopAndLog(STORAGE_SM_EXECUTE_THREAD);

    return NULL;
}

void* ClimbKeyThread (void* arg)
{
    // Get the context
    ExecutorContext * pExecutorContext = (ExecutorContext *)arg;

    // Execute the ClimbKey State Machine
    TimerStart(CLIMB_KEY_SM_EXECUTE_THREAD);
    pExecutorContext->pExecutor->climbKeyExecutor.execute(pExecutorContext->pRequired->ClimbKey, pExecutorContext->pCommitPols->ClimbKey);
    TimerStopAndLog(CLIMB_KEY_SM_EXECUTE_THREAD);

    return NULL;
}

void* PoseidonThread (void* arg)
{
    // Get the context
    ExecutorContext * pExecutorContext = (ExecutorContext *)arg;

    // Execute the Poseidon G State Machine
    TimerStart(POSEIDON_G_SM_EXECUTE_THREAD);
    pExecutorContext->pExecutor->poseidonGExecutor.execute(pExecutorContext->pRequired->PoseidonG, pExecutorContext->pRequired->PoseidonGFromPG, pExecutorContext->pRequired->PoseidonGFromST, pExecutorContext->pCommitPols->PoseidonG);
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

    // Execute the Bits2Field State Machine
    TimerStart(BITS2FIELD_SM_EXECUTE_THREAD);
    pExecutorContext->pExecutor->bits2FieldExecutor.execute(pExecutorContext->pRequired->Bits2Field, pExecutorContext->pCommitPols->Bits2Field, pExecutorContext->pRequired->KeccakF);
    TimerStopAndLog(BITS2FIELD_SM_EXECUTE_THREAD);

    // Execute the Keccak F State Machine
    TimerStart(KECCAK_F_SM_EXECUTE_THREAD);
    pExecutorContext->pExecutor->keccakFExecutor.execute(pExecutorContext->pRequired->KeccakF, pExecutorContext->pCommitPols->KeccakF);
    TimerStopAndLog(KECCAK_F_SM_EXECUTE_THREAD);

    return NULL;
}

void* Sha256Thread (void* arg)
{
    // Get the context
    ExecutorContext * pExecutorContext = (ExecutorContext *)arg;

    // Execute the Padding SHA256 State Machine
    TimerStart(PADDING_SHA256_SM_EXECUTE_THREAD);
    pExecutorContext->pExecutor->paddingSha256Executor.execute(pExecutorContext->pRequired->PaddingSha256, pExecutorContext->pCommitPols->PaddingSha256, pExecutorContext->pRequired->PaddingSha256Bit);
    TimerStopAndLog(PADDING_SHA256_SM_EXECUTE_THREAD);

    // Execute the PaddingSha256Bit State Machine
    TimerStart(PADDING_SHA256_BIT_SM_EXECUTE_THREAD);
    pExecutorContext->pExecutor->paddingSha256BitExecutor.execute(pExecutorContext->pRequired->PaddingSha256Bit, pExecutorContext->pCommitPols->PaddingSha256Bit, pExecutorContext->pRequired->Bits2FieldSha256);
    TimerStopAndLog(PADDING_SHA256_BIT_SM_EXECUTE_THREAD);

    // Execute the Bits2FieldSha256 State Machine
    TimerStart(BITS2FIELDSHA256_SM_EXECUTE_THREAD);
    pExecutorContext->pExecutor->bits2FieldSha256Executor.execute(pExecutorContext->pRequired->Bits2FieldSha256, pExecutorContext->pCommitPols->Bits2FieldSha256, pExecutorContext->pRequired->Sha256F);
    TimerStopAndLog(BITS2FIELDSHA256_SM_EXECUTE_THREAD);

    // Execute the Sha256 F State Machine
    TimerStart(SHA256_F_SM_EXECUTE_THREAD);
    pExecutorContext->pExecutor->sha256FExecutor.execute(pExecutorContext->pRequired->Sha256F, pExecutorContext->pCommitPols->Sha256F);
    TimerStopAndLog(SHA256_F_SM_EXECUTE_THREAD);

    return NULL;
}

// Full version: all polynomials are evaluated, in all evaluations
void Executor::executeBatch (ProverRequest &proverRequest, PROVER_FORK_NAMESPACE::CommitPols & commitPols)
{
    if (!config.executeInParallel)
    {
        // This instance will store all data required to execute the rest of State Machines
        PROVER_FORK_NAMESPACE::MainExecRequired required;

        // Execute the Main State Machine
        TimerStart(MAIN_EXECUTOR_EXECUTE);
        if (proverRequest.input.publicInputsExtended.publicInputs.forkID == PROVER_FORK_ID)
        {
#ifdef MAIN_SM_PROVER_GENERATED_CODE
            if (config.useMainExecGenerated)
            {
                PROVER_FORK_NAMESPACE::main_exec_generated_13(mainExecutor_fork_13, proverRequest, commitPols.Main, required);
            }
            else
#endif
            {
                mainExecutor_fork_13.execute(proverRequest, commitPols.Main, required);
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
            zklog.error("Executor::executeBatch() got invalid fork ID=" + to_string(proverRequest.input.publicInputsExtended.publicInputs.forkID));
            proverRequest.result = ZKR_SM_MAIN_INVALID_FORK_ID;
        }
        TimerStopAndLog(MAIN_EXECUTOR_EXECUTE);

        if (proverRequest.result != ZKR_SUCCESS)
        {
            return;
        }

        // Execute the Padding PG State Machine
        TimerStart(PADDING_PG_SM_EXECUTE);
        paddingPGExecutor.execute(required.PaddingPG, commitPols.PaddingPG, required.PoseidonGFromPG);
        TimerStopAndLog(PADDING_PG_SM_EXECUTE);

        // Execute the Storage State Machine
        TimerStart(STORAGE_SM_EXECUTE);
        storageExecutor.execute(required.Storage, commitPols.Storage, required.PoseidonGFromST, required.ClimbKey);
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

        // Execute the PaddingSha256 State Machine
        TimerStart(PADDING_SHA256_SM_EXECUTE);
        paddingSha256Executor.execute(required.PaddingSha256, commitPols.PaddingSha256, required.PaddingSha256Bit);
        TimerStopAndLog(PADDING_SHA256_SM_EXECUTE);

        // Execute the PaddingSha256Bit State Machine
        TimerStart(PADDING_SHA256_BIT_SM_EXECUTE);
        paddingSha256BitExecutor.execute(required.PaddingSha256Bit, commitPols.PaddingSha256Bit, required.Bits2FieldSha256);
        TimerStopAndLog(PADDING_SHA256_BIT_SM_EXECUTE);

        // Execute the Bits2FieldSha256 State Machine
        TimerStart(BITS2FIELDSHA256_SM_EXECUTE);
        bits2FieldSha256Executor.execute(required.Bits2FieldSha256, commitPols.Bits2FieldSha256, required.Sha256F);
        TimerStopAndLog(BITS2FIELDSHA256_SM_EXECUTE);

        // Excute the Sha256 F State Machine
        TimerStart(SHA256_F_SM_EXECUTE);
        sha256FExecutor.execute(required.Sha256F, commitPols.Sha256F);
        TimerStopAndLog(SHA256_F_SM_EXECUTE);

        // Execute the PoseidonG State Machine
        TimerStart(POSEIDON_G_SM_EXECUTE);
        poseidonGExecutor.execute(required.PoseidonG, required.PoseidonGFromPG, required.PoseidonGFromST, commitPols.PoseidonG);
        TimerStopAndLog(POSEIDON_G_SM_EXECUTE);

        // Execute the ClimbKey State Machine
        TimerStart(CLIMB_KEY_SM_EXECUTE);
        climbKeyExecutor.execute(required.ClimbKey, commitPols.ClimbKey);
        TimerStopAndLog(CLIMB_KEY_SM_EXECUTE);
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
#ifdef MAIN_SM_PROVER_GENERATED_CODE
        if (config.useMainExecGenerated)
        {
            PROVER_FORK_NAMESPACE::main_exec_generated_13(mainExecutor_fork_13, proverRequest, commitPols.Main, required);
        }
        else
#endif
        {
            mainExecutor_fork_13.execute(proverRequest, commitPols.Main, required);
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
            zklog.error("Executor::executeBatch() got from main execution proverRequest.result=" + to_string(proverRequest.result) + "=" + zkresult2string(proverRequest.result));
            return;
        }

        // Execute the Storage State Machines
        pthread_t storageThread;
        pthread_create(&storageThread, NULL, StorageThread, &executorContext);

        // Execute the Padding PG
        pthread_t paddingPGThread;
        pthread_create(&paddingPGThread, NULL, PaddingPGThread, &executorContext);

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

        // Execute the PaddingKK, PaddingKKBit, Bits2Field, Keccak F
        pthread_t keccakThread;
        pthread_create(&keccakThread, NULL, KeccakThread, &executorContext);

        // Execute the PaddingSha256, PaddingSha256Bit, Bits2FieldSha256, Sha256 F
        pthread_t sha256Thread;
        pthread_create(&sha256Thread, NULL, Sha256Thread, &executorContext);

        // Wait for the Storage SM threads
        pthread_join(storageThread, NULL);

        // Execute the ClimKey State Machines (now that Storage is done)
        pthread_t climbKeyThread;
        pthread_create(&climbKeyThread, NULL, ClimbKeyThread, &executorContext);

        // Wait for the PaddingPG SM threads
        pthread_join(paddingPGThread, NULL);

        // Execute the PoseidonG State Machine (now that Storage and PaddingPG are done)
        pthread_t poseidonThread;
        pthread_create(&poseidonThread, NULL, PoseidonThread, &executorContext);

        // Wait for the parallel SM threads
        pthread_join(binaryThread, NULL);
        pthread_join(memAlignThread, NULL);
        pthread_join(memoryThread, NULL);
        pthread_join(arithThread, NULL);
        pthread_join(poseidonThread, NULL);
        pthread_join(keccakThread, NULL);
        pthread_join(sha256Thread, NULL);
        pthread_join(climbKeyThread, NULL);

    }
}