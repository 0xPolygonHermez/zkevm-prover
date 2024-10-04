#include <fstream>
#include <iomanip>
#include <unistd.h>
#include "prover.hpp"
#include "definitions.hpp"
#include "utils.hpp"
#include "scalar.hpp"
#include "proof2zkin.hpp"
#include "main.hpp"
#if (PROVER_FORK_ID == 13) // fork_13
    #include "fork_13/main.hpp"
    #include "fork_13/main.recursive1.hpp"
    #include "fork_13/main.recursive2.hpp"
    #include "fork_13/main.recursiveF.hpp"
    #include "fork_13/main.final.hpp"
#else
    #error "Invalid PROVER_FORK_ID"
#endif

#include "binfile_utils.hpp"
#include "zkey_utils.hpp"
#include "wtns_utils.hpp"
#include "sm/storage/storage_executor.hpp"
#include "timer.hpp"
#include "execFile.hpp"
#include <math.h> /* log2 */
#include "proof2zkinStark.hpp"

#include "friProofC12.hpp"
#include <algorithm> // std::min
#include <openssl/sha.h>

#include "commit_pols_starks.hpp"
#include "chelpers_steps.hpp"
#include "chelpers_steps_pack.hpp"
#include "chelpers_steps_gpu.hpp"
#ifdef __AVX512__
#include "chelpers_steps_avx512.hpp"
#endif

#include "ZkevmSteps.hpp"
#include "C12aSteps.hpp"
#include "Recursive1Steps.hpp"
#include "Recursive2Steps.hpp"
#include "RecursiveFSteps.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"
#include "memory.cuh"

#if defined(__USE_CUDA__) && defined(ENABLE_EXPERIMENTAL_CODE)
#include "cuda_utils.hpp"
#include "ntt_goldilocks.hpp"
#include <pthread.h>

int asynctask(void* (*task)(void* args), void* arg)
{
	pthread_t th;
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
	return pthread_create(&th, &attr, task, arg);
}

void* warmup_task(void* arg)
{
    warmup_all_gpus();
    return NULL;
}

void warmup_gpu()
{
    asynctask(warmup_task, NULL);
}
#endif

Prover::Prover(Goldilocks &fr,
               PoseidonGoldilocks &poseidon,
               const Config &config) : fr(fr),
                                       poseidon(poseidon),
                                       executor(fr, config, poseidon),
                                       pCurrentRequest(NULL),
                                       N(getForkN(PROVER_FORK_ID)),
                                       config(config),
                                       lastComputedRequestEndTime(0)
{
    mpz_init(altBbn128r);
    mpz_set_str(altBbn128r, "21888242871839275222246405745257275088548364400416034343698204186575808495617", 10);

    try
    {
        if (config.generateProof())
        {
            TimerStart(PROVER_INIT);

            //checkSetupHash(config.zkevmVerifier);
            //checkSetupHash(config.recursive1Verifier);
            //checkSetupHash(config.recursive2Verifier);
            //checkSetupHash(config.recursivefVerifier);
            //checkSetupHash(config.finalVerifier);

            lastComputedRequestEndTime = 0;

            sem_init(&pendingRequestSem, 0, 0);
            pthread_mutex_init(&mutex, NULL);
            pCurrentRequest = NULL;
            pthread_create(&proverPthread, NULL, proverThread, this);
            pthread_create(&cleanerPthread, NULL, cleanerThread, this);

            bool reduceMemoryZkevm = REDUCE_ZKEVM_MEMORY ? true : false;
            
            StarkInfo _starkInfo(config.zkevmStarkInfo, reduceMemoryZkevm);

            // Allocate an area of memory, mapped to file, to store all the committed polynomials,
            // and create them using the allocated address

            polsSize = _starkInfo.mapTotalN * sizeof(Goldilocks::Element);
            zkassert(_starkInfo.mapSectionsN.section[eSection::cm1_2ns] * sizeof(Goldilocks::Element) <= polsSize - _starkInfo.mapSectionsN.section[eSection::cm2_2ns] * sizeof(Goldilocks::Element));

            zkassert(PROVER_FORK_NAMESPACE::CommitPols::numPols()*sizeof(Goldilocks::Element)*N <= polsSize);

            if (config.zkevmCmPols.size() > 0)
            {
                pAddress = mapFile(config.zkevmCmPols, polsSize, true);
                zklog.info("Prover::Prover() successfully mapped " + to_string(polsSize) + " bytes to file " + config.zkevmCmPols);
            }
            else
            {
                pAddress = calloc_zkevm(polsSize, 1);
                if (pAddress == NULL)
                {
                    zklog.error("Prover::Prover() failed calling malloc() of size " + to_string(polsSize));
                    exitProcess();
                }
                zklog.info("Prover::Prover() successfully allocated " + to_string(polsSize) + " bytes");
            }

#if defined(__USE_CUDA__) && defined(ENABLE_EXPERIMENTAL_CODE)
            alloc_pinned_mem(uint64_t(1<<24) * _starkInfo.mapSectionsN.section[eSection::cm1_n]);
            warmup_gpu();
#endif
            
            json finalVerkeyJson;
            file2json(config.finalVerkey, finalVerkeyJson);
            domainSizeFflonk = 1 << uint64_t(finalVerkeyJson["power"]);
            nPublicsFflonk = finalVerkeyJson["nPublic"];

            TimerStopAndLog(PROVER_INIT);
            TimerStart(PROVER_INIT_STARKINFO);

            string zkevmChelpers = USE_GENERIC_PARSER ? config.zkevmGenericCHelpers : config.zkevmCHelpers;
            string c12aChelpers = USE_GENERIC_PARSER ? config.c12aGenericCHelpers : config.c12aCHelpers;
            string recursive1Chelpers = USE_GENERIC_PARSER ? config.recursive1GenericCHelpers : config.recursive1CHelpers;
            string recursive2Chelpers = USE_GENERIC_PARSER ? config.recursive2GenericCHelpers : config.recursive2CHelpers;
            TimerStopAndLog(PROVER_INIT_STARKINFO);
            TimerStart(PROVER_INIT_STARK_ZKEVM);    
            starkZkevm = new Starks(config, {config.zkevmConstPols, config.mapConstPolsFile, config.zkevmConstantsTree, config.zkevmStarkInfo, zkevmChelpers}, reduceMemoryZkevm, pAddress);
            TimerStopAndLog(PROVER_INIT_STARK_ZKEVM);
            TimerStart(PROVER_INIT_STARK_C12A);
            starksC12a = new Starks(config, {config.c12aConstPols, config.mapConstPolsFile, config.c12aConstantsTree, config.c12aStarkInfo, c12aChelpers}, false, pAddress);
            TimerStopAndLog(PROVER_INIT_STARK_C12A);
            TimerStart(PROVER_INIT_STARK_RECURSIVE1);
            starksRecursive1 = new Starks(config, {config.recursive1ConstPols, config.mapConstPolsFile, config.recursive1ConstantsTree, config.recursive1StarkInfo, recursive1Chelpers}, false, pAddress);
            TimerStopAndLog(PROVER_INIT_STARK_RECURSIVE1);
            TimerStart(PROVER_INIT_STARK_RECURSIVE2);
            starksRecursive2 = new Starks(config, {config.recursive2ConstPols, config.mapConstPolsFile, config.recursive2ConstantsTree, config.recursive2StarkInfo, recursive2Chelpers}, false, pAddress);
            TimerStopAndLog(PROVER_INIT_STARK_RECURSIVE2);
            TimerStart(PROVER_INIT_STARK_RECURSIVEF);
            starksRecursiveF = new StarkRecursiveF(config, pAddress);
            TimerStopAndLog(PROVER_INIT_STARK_RECURSIVEF);
        }
    }
    catch (std::exception &e)
    {
        zklog.error("Prover::Prover() got an exception: " + string(e.what()));
        exitProcess();
    }
}

Prover::~Prover()
{
    mpz_clear(altBbn128r);

    if (config.generateProof())
    {
        // Unmap committed polynomials address
        if (config.zkevmCmPols.size() > 0)
        {
            unmapFile(pAddress, polsSize);
        }
        else
        {
            free_zkevm(pAddress);
        }
#if defined(__USE_CUDA__) && defined(ENABLE_EXPERIMENTAL_CODE)
        free_pinned_mem();
#endif

        delete starkZkevm;
        delete starksC12a;
        delete starksRecursive1;
        delete starksRecursive2;
        delete starksRecursiveF;
    }
}

void *proverThread(void *arg)
{
    Prover *pProver = (Prover *)arg;
    zklog.info("proverThread() started");

    zkassert(pProver->config.generateProof());

    while (true)
    {
        pProver->lock();

        // Wait for the pending request queue semaphore to be released, if there are no more pending requests
        if (pProver->pendingRequests.size() == 0)
        {
            pProver->unlock();
            sem_wait(&pProver->pendingRequestSem);
        }

        // Check that the pending requests queue is not empty
        if (pProver->pendingRequests.size() == 0)
        {
            pProver->unlock();
            zklog.info("proverThread() found pending requests queue empty, so ignoring");
            continue;
        }

        // Extract the first pending request (first in, first out)
        pProver->pCurrentRequest = pProver->pendingRequests[0];
        pProver->pCurrentRequest->startTime = time(NULL);
        pProver->pendingRequests.erase(pProver->pendingRequests.begin());

        zklog.info("proverThread() starting to process request with UUID: " + pProver->pCurrentRequest->uuid);

        pProver->unlock();

        // Process the request
        switch (pProver->pCurrentRequest->type)
        {
        case prt_genBatchProof:
            pProver->genBatchProof(pProver->pCurrentRequest);
            break;
        case prt_genAggregatedProof:
            pProver->genAggregatedProof(pProver->pCurrentRequest);
            break;
        case prt_genFinalProof:
            pProver->genFinalProof(pProver->pCurrentRequest);
            break;
        case prt_execute:
            pProver->execute(pProver->pCurrentRequest);
            break;
        default:
            zklog.error("proverThread() got an invalid prover request type=" + to_string(pProver->pCurrentRequest->type));
            exitProcess();
        }

        // Move to completed requests
        pProver->lock();
        ProverRequest *pProverRequest = pProver->pCurrentRequest;
        pProverRequest->endTime = time(NULL);
        pProver->lastComputedRequestId = pProverRequest->uuid;
        pProver->lastComputedRequestEndTime = pProverRequest->endTime;

        pProver->completedRequests.push_back(pProver->pCurrentRequest);
        pProver->pCurrentRequest = NULL;
        pProver->unlock();

        zklog.info("proverThread() done processing request with UUID: " + pProverRequest->uuid);

        // Release the prove request semaphore to notify any blocked waiting call
        pProverRequest->notifyCompleted();
    }
    zklog.info("proverThread() done");
    return NULL;
}

void *cleanerThread(void *arg)
{
    Prover *pProver = (Prover *)arg;
    zklog.info("cleanerThread() started");

    zkassert(pProver->config.generateProof());

    while (true)
    {
        // Sleep for 10 minutes
        sleep(pProver->config.cleanerPollingPeriod);

        // Lock the prover
        pProver->lock();

        // Delete all requests older than requests persistence configuration setting
        time_t now = time(NULL);
        bool bRequestDeleted = false;
        do
        {
            bRequestDeleted = false;
            for (uint64_t i = 0; i < pProver->completedRequests.size(); i++)
            {
                if (now - pProver->completedRequests[i]->endTime > (int64_t)pProver->config.requestsPersistence)
                {
                    zklog.info("cleanerThread() deleting request with uuid: " + pProver->completedRequests[i]->uuid);
                    ProverRequest *pProverRequest = pProver->completedRequests[i];
                    pProver->completedRequests.erase(pProver->completedRequests.begin() + i);
                    pProver->requestsMap.erase(pProverRequest->uuid);
                    delete (pProverRequest);
                    bRequestDeleted = true;
                    break;
                }
            }
        } while (bRequestDeleted);

        // Unlock the prover
        pProver->unlock();
    }
    zklog.info("cleanerThread() done");
    return NULL;
}

string Prover::submitRequest(ProverRequest *pProverRequest) // returns UUID for this request
{
    zkassert(config.generateProof());
    zkassert(pProverRequest != NULL);

    zklog.info("Prover::submitRequest() started type=" + to_string(pProverRequest->type));

    // Get the prover request UUID
    string uuid = pProverRequest->uuid;

    // Add the request to the pending requests queue, and release the semaphore to notify the prover thread
    lock();
    requestsMap[uuid] = pProverRequest;
    pendingRequests.push_back(pProverRequest);
    sem_post(&pendingRequestSem);
    unlock();

    zklog.info("Prover::submitRequest() returns UUID: " + uuid);
    return uuid;
}

ProverRequest *Prover::waitForRequestToComplete(const string &uuid, const uint64_t timeoutInSeconds) // wait for the request with this UUID to complete; returns NULL if UUID is invalid
{
    zkassert(config.generateProof());
    zkassert(uuid.size() > 0);
    zklog.info("Prover::waitForRequestToComplete() waiting for request with UUID: " + uuid);

    // We will store here the address of the prove request corresponding to this UUID
    ProverRequest *pProverRequest = NULL;

    lock();

    // Map uuid to the corresponding prover request
    std::unordered_map<std::string, ProverRequest *>::iterator it = requestsMap.find(uuid);
    if (it == requestsMap.end())
    {
        zklog.error("Prover::waitForRequestToComplete() unknown uuid: " + uuid);
        unlock();
        return NULL;
    }

    // Wait for the request to complete
    pProverRequest = it->second;
    unlock();
    pProverRequest->waitForCompleted(timeoutInSeconds);
    zklog.info("Prover::waitForRequestToComplete() done waiting for request with UUID: " + uuid);

    // Return the request pointer
    return pProverRequest;
}

void Prover::processBatch(ProverRequest *pProverRequest)
{
    //TimerStart(PROVER_PROCESS_BATCH);
    zkassert(pProverRequest != NULL);
    zkassert(pProverRequest->type == prt_processBatch);

    if (config.runAggregatorClient)
    {
        zklog.info("Prover::processBatch() timestamp=" + pProverRequest->timestamp + " UUID=" + pProverRequest->uuid);
    }

    // Save input to <timestamp>.input.json, as provided by client
    if (config.saveInputToFile)
    {
        json inputJson;
        pProverRequest->input.save(inputJson);
        json2file(inputJson, pProverRequest->inputFile());
    }

    // Log input if requested
    if (config.logExecutorServerInput)
    {
        json inputJson;
        pProverRequest->input.save(inputJson);
        zklog.info("Input=" + inputJson.dump());
    }

    // Execute the program, in the process batch way
    executor.processBatch(*pProverRequest);

    // Save input to <timestamp>.input.json after execution including dbReadLog
    if (config.saveDbReadsToFile)
    {
        json inputJsonEx;
        pProverRequest->input.save(inputJsonEx, *pProverRequest->dbReadLog);
        json2file(inputJsonEx, pProverRequest->inputDbFile());
    }

    //TimerStopAndLog(PROVER_PROCESS_BATCH);
}

void Prover::genBatchProof(ProverRequest *pProverRequest)
{
    zkassert(config.generateProof());
    zkassert(pProverRequest != NULL);

    TimerStart(PROVER_BATCH_PROOF);

    printMemoryInfo(true);
    printProcessInfo(true);

    zkassert(pProverRequest != NULL);

    zklog.info("Prover::genBatchProof() timestamp: " + pProverRequest->timestamp);
    zklog.info("Prover::genBatchProof() UUID: " + pProverRequest->uuid);
    zklog.info("Prover::genBatchProof() input file: " + pProverRequest->inputFile());
    // zklog.info("Prover::genBatchProof() public file: " + pProverRequest->publicsOutputFile());
    // zklog.info("Prover::genBatchProof() proof file: " + pProverRequest->proofFile());

    // Save input to <timestamp>.input.json, as provided by client
    if (config.saveInputToFile)
    {
        json inputJson;
        pProverRequest->input.save(inputJson);
        json2file(inputJson, pProverRequest->inputFile());
    }

    /************/
    /* Executor */
    /************/
    TimerStart(EXECUTOR_EXECUTE_INITIALIZATION);

    PROVER_FORK_NAMESPACE::CommitPols cmPols((uint8_t *)pAddress + starkZkevm->starkInfo.mapOffsets.section[cm1_n] * sizeof(Goldilocks::Element), N);
    // Goldilocks::parSetZero((Goldilocks::Element*)pAddress, cmPols.size()/sizeof(Goldilocks::Element), omp_get_max_threads()/2);
    uint64_t num_threads = omp_get_max_threads();
    uint64_t bytes_per_thread = cmPols.size() / num_threads;
#pragma omp parallel for num_threads(num_threads)
    for (uint64_t i = 0; i < cmPols.size(); i += bytes_per_thread) // Each iteration processes 64 bytes at a time
    {
        memset((uint8_t *)pAddress + starkZkevm->starkInfo.mapOffsets.section[cm1_n]*sizeof(Goldilocks::Element) + i, 0, bytes_per_thread);
    }

    TimerStopAndLog(EXECUTOR_EXECUTE_INITIALIZATION);
    // Execute all the State Machines
    TimerStart(EXECUTOR_EXECUTE_BATCH_PROOF);
    executor.executeBatch(*pProverRequest, cmPols);
    TimerStopAndLog(EXECUTOR_EXECUTE_BATCH_PROOF);

    uint64_t lastN = N - 1;

    zklog.info("Prover::genBatchProof() called executor.execute() oldStateRoot=" + pProverRequest->input.publicInputsExtended.publicInputs.oldStateRoot.get_str(16) +
        " newStateRoot=" + pProverRequest->pFullTracer->get_new_state_root() +
        " pols.B[0]=" + fea2stringchain(fr, cmPols.Main.B0[0], cmPols.Main.B1[0], cmPols.Main.B2[0], cmPols.Main.B3[0], cmPols.Main.B4[0], cmPols.Main.B5[0], cmPols.Main.B6[0], cmPols.Main.B7[0]) +
        " pols.SR[lastN]=" + fea2stringchain(fr, cmPols.Main.SR0[lastN], cmPols.Main.SR1[lastN], cmPols.Main.SR2[lastN], cmPols.Main.SR3[lastN], cmPols.Main.SR4[lastN], cmPols.Main.SR5[lastN], cmPols.Main.SR6[lastN], cmPols.Main.SR7[lastN]) +
        " lastN=" + to_string(lastN));
    zklog.info("Prover::genBatchProof() called executor.execute() oldAccInputHash=" + pProverRequest->input.publicInputsExtended.publicInputs.oldAccInputHash.get_str(16) +
        " newAccInputHash=" + pProverRequest->pFullTracer->get_new_acc_input_hash() +
        " pols.C[0]=" + fea2stringchain(fr, cmPols.Main.C0[0], cmPols.Main.C1[0], cmPols.Main.C2[0], cmPols.Main.C3[0], cmPols.Main.C4[0], cmPols.Main.C5[0], cmPols.Main.C6[0], cmPols.Main.C7[0]) +
        " pols.D[lastN]=" + fea2stringchain(fr, cmPols.Main.D0[lastN], cmPols.Main.D1[lastN], cmPols.Main.D2[lastN], cmPols.Main.D3[lastN], cmPols.Main.D4[lastN], cmPols.Main.D5[lastN], cmPols.Main.D6[lastN], cmPols.Main.D7[lastN]) +
        " lastN=" + to_string(lastN));

    // Save commit pols to file zkevm.commit
    if (config.zkevmCmPolsAfterExecutor != "")
    {
        void *pointerCmPols = mapFile(config.zkevmCmPolsAfterExecutor, cmPols.size(), true);
        memcpy(pointerCmPols, cmPols.address(), cmPols.size());
        unmapFile(pointerCmPols, cmPols.size());
    }

    if (pProverRequest->result == ZKR_SUCCESS)
    {
        /*************************************/
        /*  Generate publics input           */
        /*************************************/
        TimerStart(SAVE_PUBLICS_JSON_BATCH_PROOF);
        json publicStarkJson;

        json zkevmVerkeyJson;
        file2json(config.zkevmVerkey, zkevmVerkeyJson);
        Goldilocks::Element zkevmVerkey[4];
        zkevmVerkey[0] = Goldilocks::fromU64(zkevmVerkeyJson["constRoot"][0]);
        zkevmVerkey[1] = Goldilocks::fromU64(zkevmVerkeyJson["constRoot"][1]);
        zkevmVerkey[2] = Goldilocks::fromU64(zkevmVerkeyJson["constRoot"][2]);
        zkevmVerkey[3] = Goldilocks::fromU64(zkevmVerkeyJson["constRoot"][3]);

        json c12aVerkeyJson;
        file2json(config.c12aVerkey, c12aVerkeyJson);
        Goldilocks::Element c12aVerkey[4];
        c12aVerkey[0] = Goldilocks::fromU64(c12aVerkeyJson["constRoot"][0]);
        c12aVerkey[1] = Goldilocks::fromU64(c12aVerkeyJson["constRoot"][1]);
        c12aVerkey[2] = Goldilocks::fromU64(c12aVerkeyJson["constRoot"][2]);
        c12aVerkey[3] = Goldilocks::fromU64(c12aVerkeyJson["constRoot"][3]);

        json recursive1VerkeyJson;
        file2json(config.recursive1Verkey, recursive1VerkeyJson);
        Goldilocks::Element recursive1Verkey[4];
        recursive1Verkey[0] = Goldilocks::fromU64(recursive1VerkeyJson["constRoot"][0]);
        recursive1Verkey[1] = Goldilocks::fromU64(recursive1VerkeyJson["constRoot"][1]);
        recursive1Verkey[2] = Goldilocks::fromU64(recursive1VerkeyJson["constRoot"][2]);
        recursive1Verkey[3] = Goldilocks::fromU64(recursive1VerkeyJson["constRoot"][3]);

        json recursive2Verkey;
        file2json(config.recursive2Verkey, recursive2Verkey);

        Goldilocks::Element publics[starksRecursive1->starkInfo.nPublics];

        // oldStateRoot
        publics[0] = cmPols.Main.B0[0];
        publics[1] = cmPols.Main.B1[0];
        publics[2] = cmPols.Main.B2[0];
        publics[3] = cmPols.Main.B3[0];
        publics[4] = cmPols.Main.B4[0];
        publics[5] = cmPols.Main.B5[0];
        publics[6] = cmPols.Main.B6[0];
        publics[7] = cmPols.Main.B7[0];

        // oldAccInputHash
        publics[8] = cmPols.Main.C0[0];
        publics[9] = cmPols.Main.C1[0];
        publics[10] = cmPols.Main.C2[0];
        publics[11] = cmPols.Main.C3[0];
        publics[12] = cmPols.Main.C4[0];
        publics[13] = cmPols.Main.C5[0];
        publics[14] = cmPols.Main.C6[0];
        publics[15] = cmPols.Main.C7[0];

        // oldBatchNum
        publics[16] = cmPols.Main.SP[0];
        // chainId
        publics[17] = cmPols.Main.GAS[0];
        // forkid
        publics[18] = cmPols.Main.CTX[0];

        // newStateRoot
        publics[19] = cmPols.Main.SR0[lastN];
        publics[20] = cmPols.Main.SR1[lastN];
        publics[21] = cmPols.Main.SR2[lastN];
        publics[22] = cmPols.Main.SR3[lastN];
        publics[23] = cmPols.Main.SR4[lastN];
        publics[24] = cmPols.Main.SR5[lastN];
        publics[25] = cmPols.Main.SR6[lastN];
        publics[26] = cmPols.Main.SR7[lastN];

        // newAccInputHash
        publics[27] = cmPols.Main.D0[lastN];
        publics[28] = cmPols.Main.D1[lastN];
        publics[29] = cmPols.Main.D2[lastN];
        publics[30] = cmPols.Main.D3[lastN];
        publics[31] = cmPols.Main.D4[lastN];
        publics[32] = cmPols.Main.D5[lastN];
        publics[33] = cmPols.Main.D6[lastN];
        publics[34] = cmPols.Main.D7[lastN];

        // localExitRoot
        publics[35] = cmPols.Main.E0[lastN];
        publics[36] = cmPols.Main.E1[lastN];
        publics[37] = cmPols.Main.E2[lastN];
        publics[38] = cmPols.Main.E3[lastN];
        publics[39] = cmPols.Main.E4[lastN];
        publics[40] = cmPols.Main.E5[lastN];
        publics[41] = cmPols.Main.E6[lastN];
        publics[42] = cmPols.Main.E7[lastN];

        // newBatchNum
        publics[43] = cmPols.Main.PC[lastN];

        publics[44] = Goldilocks::fromU64(recursive2Verkey["constRoot"][0]);
        publics[45] = Goldilocks::fromU64(recursive2Verkey["constRoot"][1]);
        publics[46] = Goldilocks::fromU64(recursive2Verkey["constRoot"][2]);
        publics[47] = Goldilocks::fromU64(recursive2Verkey["constRoot"][3]);

        for (uint64_t i = 0; i < starkZkevm->starkInfo.nPublics; i++)
        {
            publicStarkJson[i] = Goldilocks::toString(publics[i]);
        }

        TimerStopAndLog(SAVE_PUBLICS_JSON_BATCH_PROOF);

        /*************************************/
        /*  Generate stark proof            */
        /*************************************/

#if defined(__USE_CUDA__) && defined(ENABLE_EXPERIMENTAL_CODE)
        CHelpersStepsGPU cHelpersSteps;
#elif defined(__AVX512__)
        CHelpersStepsAvx512 cHelpersSteps;
#elif defined(__PACK__) 
        CHelpersStepsPack cHelpersSteps;
        cHelpersSteps.nrowsPack = NROWS_PACK;
#else
        CHelpersSteps cHelpersSteps;
#endif

        TimerStart(STARK_PROOF_BATCH_PROOF);

        ZkevmSteps zkevmChelpersSteps;
        uint64_t polBits = starkZkevm->starkInfo.starkStruct.steps[starkZkevm->starkInfo.starkStruct.steps.size() - 1].nBits;
        FRIProof fproof((1 << polBits), FIELD_EXTENSION, starkZkevm->starkInfo.starkStruct.steps.size(), starkZkevm->starkInfo.evMap.size(), starkZkevm->starkInfo.nPublics);
        
        if(USE_GENERIC_PARSER) {
            starkZkevm->genProof(fproof, &publics[0], zkevmVerkey, &cHelpersSteps);
        } else {
            starkZkevm->genProof(fproof, &publics[0], zkevmVerkey, &zkevmChelpersSteps);
        }

        TimerStopAndLog(STARK_PROOF_BATCH_PROOF);
        TimerStart(STARK_GEN_AND_CALC_WITNESS_C12A);
        TimerStart(STARK_JSON_GENERATION_BATCH_PROOF);

        nlohmann::ordered_json jProof = fproof.proofs.proof2json();
        nlohmann::json zkin = proof2zkinStark(jProof);
        // Generate publics
        jProof["publics"] = publicStarkJson;
        zkin["publics"] = publicStarkJson;

        TimerStopAndLog(STARK_JSON_GENERATION_BATCH_PROOF);


        CommitPolsStarks cmPols12a((uint8_t *)pAddress + starksC12a->starkInfo.mapOffsets.section[cm1_n] * sizeof(Goldilocks::Element), (1 << starksC12a->starkInfo.starkStruct.nBits), starksC12a->starkInfo.nCm1);
    #if (PROVER_FORK_ID == 13) // fork_13
        CircomFork13::getCommitedPols(&cmPols12a, config.zkevmVerifier, config.c12aExec, zkin, (1 << starksC12a->starkInfo.starkStruct.nBits), starksC12a->starkInfo.nCm1);
    #else
        #error "Invalid PROVER_FORK_ID"
    #endif

        // void *pointerCm12aPols = mapFile("config/c12a/c12a.commit", cmPols12a.size(), true);
        // memcpy(pointerCm12aPols, cmPols12a.address(), cmPols12a.size());
        // unmapFile(pointerCm12aPols, cmPols12a.size());

        //-------------------------------------------
        /* Generate C12a stark proof             */
        //-------------------------------------------
        TimerStopAndLog(STARK_GEN_AND_CALC_WITNESS_C12A);
        TimerStart(STARK_C12_A_PROOF_BATCH_PROOF);
        C12aSteps c12aChelpersSteps;
        uint64_t polBitsC12 = starksC12a->starkInfo.starkStruct.steps[starksC12a->starkInfo.starkStruct.steps.size() - 1].nBits;
        FRIProof fproofC12a((1 << polBitsC12), FIELD_EXTENSION, starksC12a->starkInfo.starkStruct.steps.size(), starksC12a->starkInfo.evMap.size(), starksC12a->starkInfo.nPublics);

        // Generate the proof
        if(USE_GENERIC_PARSER) {
            starksC12a->genProof(fproofC12a, publics, c12aVerkey, &cHelpersSteps);
        } else {
            starksC12a->genProof(fproofC12a, publics, c12aVerkey, &c12aChelpersSteps);
        }

        TimerStopAndLog(STARK_C12_A_PROOF_BATCH_PROOF);
        TimerStart(STARK_JSON_GENERATION_BATCH_PROOF_C12A);

        // Save the proof & zkinproof
        nlohmann::ordered_json jProofc12a = fproofC12a.proofs.proof2json();
        nlohmann::json zkinC12a = proof2zkinStark(jProofc12a);

        // Add the recursive2 verification key
        json rootC;
        rootC[0] = to_string(recursive2Verkey["constRoot"][0]);
        rootC[1] = to_string(recursive2Verkey["constRoot"][1]);
        rootC[2] = to_string(recursive2Verkey["constRoot"][2]);
        rootC[3] = to_string(recursive2Verkey["constRoot"][3]);
        zkinC12a["publics"] = publicStarkJson;
        zkinC12a["rootC"] = rootC;
        TimerStopAndLog(STARK_JSON_GENERATION_BATCH_PROOF_C12A);

        CommitPolsStarks cmPolsRecursive1((uint8_t *)pAddress + starksRecursive1->starkInfo.mapOffsets.section[cm1_n] * sizeof(Goldilocks::Element), (1 << starksRecursive1->starkInfo.starkStruct.nBits), starksRecursive1->starkInfo.nCm1);
    #if (PROVER_FORK_ID == 13) // fork_13
        CircomRecursive1Fork13::getCommitedPols(&cmPolsRecursive1, config.recursive1Verifier, config.recursive1Exec, zkinC12a, (1 << starksRecursive1->starkInfo.starkStruct.nBits), starksRecursive1->starkInfo.nCm1);
    #else
        #error "Invalid PROVER_FORK_ID"
    #endif

        // void *pointerCmRecursive1Pols = mapFile("config/recursive1/recursive1.commit", cmPolsRecursive1.size(), true);
        // memcpy(pointerCmRecursive1Pols, cmPolsRecursive1.address(), cmPolsRecursive1.size());
        // unmapFile(pointerCmRecursive1Pols, cmPolsRecursive1.size());

        //-------------------------------------------
        /* Generate Recursive 1 proof            */
        //-------------------------------------------

        TimerStart(STARK_RECURSIVE_1_PROOF_BATCH_PROOF);
        Recursive1Steps recursive1ChelpersSteps;
        uint64_t polBitsRecursive1 = starksRecursive1->starkInfo.starkStruct.steps[starksRecursive1->starkInfo.starkStruct.steps.size() - 1].nBits;
        FRIProof fproofRecursive1((1 << polBitsRecursive1), FIELD_EXTENSION, starksRecursive1->starkInfo.starkStruct.steps.size(), starksRecursive1->starkInfo.evMap.size(), starksRecursive1->starkInfo.nPublics);
        
        if(USE_GENERIC_PARSER) {
            starksRecursive1->genProof(fproofRecursive1, publics, recursive1Verkey, &cHelpersSteps);
        } else {
            starksRecursive1->genProof(fproofRecursive1, publics, recursive1Verkey, &recursive1ChelpersSteps);
        }
        TimerStopAndLog(STARK_RECURSIVE_1_PROOF_BATCH_PROOF);

        // Save the proof & zkinproof
        TimerStart(SAVE_PROOF);

        nlohmann::ordered_json jProofRecursive1 = fproofRecursive1.proofs.proof2json();
        nlohmann::ordered_json zkinRecursive1 = proof2zkinStark(jProofRecursive1);
        zkinRecursive1["publics"] = publicStarkJson;

        pProverRequest->batchProofOutput = zkinRecursive1;

        // save publics to filestarks
        json2file(publicStarkJson, pProverRequest->publicsOutputFile());

        // Save output to file
        if (config.saveOutputToFile)
        {
            json2file(pProverRequest->batchProofOutput, pProverRequest->filePrefix + "batch_proof.output.json");
        }
        // Save proof to file
        if (config.saveProofToFile)
        {
            jProofRecursive1["publics"] = publicStarkJson;
            json2file(jProofRecursive1, pProverRequest->filePrefix + "batch_proof.proof.json");
        }
        TimerStopAndLog(SAVE_PROOF);
    }

    TimerStopAndLog(PROVER_BATCH_PROOF);
}

void Prover::genAggregatedProof(ProverRequest *pProverRequest)
{

    zkassert(config.generateProof());
    zkassert(pProverRequest != NULL);
    zkassert(pProverRequest->type == prt_genAggregatedProof);

    TimerStart(PROVER_AGGREGATED_PROOF);

    printMemoryInfo(true);
    printProcessInfo(true);

    // Save input to file
    if (config.saveInputToFile)
    {
        json2file(pProverRequest->aggregatedProofInput1, pProverRequest->filePrefix + "aggregated_proof.input_1.json");
        json2file(pProverRequest->aggregatedProofInput2, pProverRequest->filePrefix + "aggregated_proof.input_2.json");
    }

    // Input is pProverRequest->aggregatedProofInput1 and pProverRequest->aggregatedProofInput2 (of type json)

    ordered_json verKey;
    file2json(config.recursive2Verkey, verKey);

    // ----------------------------------------------
    // CHECKS
    // ----------------------------------------------
    // Check chainID

    if (pProverRequest->aggregatedProofInput1["publics"][17] != pProverRequest->aggregatedProofInput2["publics"][17])
    {
        zklog.error("Prover::genAggregatedProof() Inputs has different chainId " + pProverRequest->aggregatedProofInput1["publics"][17].dump() + "!=" + pProverRequest->aggregatedProofInput2["publics"][17].dump());
        pProverRequest->result = ZKR_AGGREGATED_PROOF_INVALID_INPUT;
        return;
    }
    if (pProverRequest->aggregatedProofInput1["publics"][18] != pProverRequest->aggregatedProofInput2["publics"][18])
    {
        zklog.error("Prover::genAggregatedProof() Inputs has different forkId " + pProverRequest->aggregatedProofInput1["publics"][18].dump() + "!=" + pProverRequest->aggregatedProofInput2["publics"][18].dump());
        pProverRequest->result = ZKR_AGGREGATED_PROOF_INVALID_INPUT;
        return;
    }
    // Check midStateRoot
    for (int i = 0; i < 8; i++)
    {
        if (pProverRequest->aggregatedProofInput1["publics"][19 + i] != pProverRequest->aggregatedProofInput2["publics"][0 + i])
        {
            zklog.error("Prover::genAggregatedProof() The newStateRoot and the oldStateRoot are not consistent " + pProverRequest->aggregatedProofInput1["publics"][19 + i].dump() + "!=" + pProverRequest->aggregatedProofInput2["publics"][0 + i].dump());
            pProverRequest->result = ZKR_AGGREGATED_PROOF_INVALID_INPUT;
            return;
        }
    }
    // Check midAccInputHash0
    for (int i = 0; i < 8; i++)
    {
        if (pProverRequest->aggregatedProofInput1["publics"][27 + i] != pProverRequest->aggregatedProofInput2["publics"][8 + i])
        {
            zklog.error("Prover::genAggregatedProof() newAccInputHash and oldAccInputHash are not consistent" + pProverRequest->aggregatedProofInput1["publics"][27 + i].dump() + "!=" + pProverRequest->aggregatedProofInput2["publics"][8 + i].dump());
            pProverRequest->result = ZKR_AGGREGATED_PROOF_INVALID_INPUT;
            return;
        }
    }
    // Check batchNum
    if (pProverRequest->aggregatedProofInput1["publics"][43] != pProverRequest->aggregatedProofInput2["publics"][16])
    {
        zklog.error("Prover::genAggregatedProof() newBatchNum and oldBatchNum are not consistent" + pProverRequest->aggregatedProofInput1["publics"][43].dump() + "!=" + pProverRequest->aggregatedProofInput2["publics"][16].dump());
        pProverRequest->result = ZKR_AGGREGATED_PROOF_INVALID_INPUT;
        return;
    }

    json zkinInputRecursive2 = joinzkin(pProverRequest->aggregatedProofInput1, pProverRequest->aggregatedProofInput2, verKey, starksRecursive2->starkInfo.starkStruct.steps.size());
    json recursive2Verkey;
    file2json(config.recursive2Verkey, recursive2Verkey);

    Goldilocks::Element recursive2VerkeyValues[4];
    recursive2VerkeyValues[0] = Goldilocks::fromU64(recursive2Verkey["constRoot"][0]);
    recursive2VerkeyValues[1] = Goldilocks::fromU64(recursive2Verkey["constRoot"][1]);
    recursive2VerkeyValues[2] = Goldilocks::fromU64(recursive2Verkey["constRoot"][2]);
    recursive2VerkeyValues[3] = Goldilocks::fromU64(recursive2Verkey["constRoot"][3]);

    Goldilocks::Element publics[starksRecursive2->starkInfo.nPublics];

    for (uint64_t i = 0; i < starkZkevm->starkInfo.nPublics; i++)
    {
        publics[i] = Goldilocks::fromString(zkinInputRecursive2["publics"][i]);
    }

    for (uint64_t i = 0; i < recursive2Verkey["constRoot"].size(); i++)
    {
        publics[starkZkevm->starkInfo.nPublics + i] = Goldilocks::fromU64(recursive2Verkey["constRoot"][i]);
    }

    CommitPolsStarks cmPolsRecursive2((uint8_t *)pAddress + starksRecursive2->starkInfo.mapOffsets.section[cm1_n] * sizeof(Goldilocks::Element), (1 << starksRecursive2->starkInfo.starkStruct.nBits), starksRecursive2->starkInfo.nCm1);
    #if (PROVER_FORK_ID == 13) // fork_13
        CircomRecursive2Fork13::getCommitedPols(&cmPolsRecursive2, config.recursive2Verifier, config.recursive2Exec, zkinInputRecursive2, (1 << starksRecursive2->starkInfo.starkStruct.nBits), starksRecursive2->starkInfo.nCm1);
    #else
        #error "Invalid PROVER_FORK_ID"
    #endif

    // void *pointerCmRecursive2Pols = mapFile("config/recursive2/recursive2.commit", cmPolsRecursive2.size(), true);
    // memcpy(pointerCmRecursive2Pols, cmPolsRecursive2.address(), cmPolsRecursive2.size());
    // unmapFile(pointerCmRecursive2Pols, cmPolsRecursive2.size());

    //-------------------------------------------
    // Generate Recursive 2 proof
    //-------------------------------------------

    TimerStart(STARK_RECURSIVE_2_PROOF_BATCH_PROOF);
    Recursive2Steps recursive2ChelpersSteps;
    uint64_t polBitsRecursive2 = starksRecursive2->starkInfo.starkStruct.steps[starksRecursive2->starkInfo.starkStruct.steps.size() - 1].nBits;
    FRIProof fproofRecursive2((1 << polBitsRecursive2), FIELD_EXTENSION, starksRecursive2->starkInfo.starkStruct.steps.size(), starksRecursive2->starkInfo.evMap.size(), starksRecursive2->starkInfo.nPublics);
    
    if(USE_GENERIC_PARSER) {
#if defined(__USE_CUDA__) && defined(ENABLE_EXPERIMENTAL_CODE)
        CHelpersStepsGPU cHelpersSteps;        
#elif defined(__AVX512__)
        CHelpersStepsAvx512 cHelpersSteps;
#elif defined(__PACK__) 
        CHelpersStepsPack cHelpersSteps;
        cHelpersSteps.nrowsPack = NROWS_PACK;
#else
        CHelpersSteps cHelpersSteps;
#endif        
        starksRecursive2->genProof(fproofRecursive2, publics, recursive2VerkeyValues, &cHelpersSteps);
    } else {
        starksRecursive2->genProof(fproofRecursive2, publics, recursive2VerkeyValues, &recursive2ChelpersSteps);
    }
   
    TimerStopAndLog(STARK_RECURSIVE_2_PROOF_BATCH_PROOF);

    // Save the proof & zkinproof
    nlohmann::ordered_json jProofRecursive2 = fproofRecursive2.proofs.proof2json();
    nlohmann::ordered_json zkinRecursive2 = proof2zkinStark(jProofRecursive2);
    zkinRecursive2["publics"] = zkinInputRecursive2["publics"];

    // Output is pProverRequest->aggregatedProofOutput (of type json)
    pProverRequest->aggregatedProofOutput = zkinRecursive2;

    // Save output to file
    if (config.saveOutputToFile)
    {
        json2file(pProverRequest->aggregatedProofOutput, pProverRequest->filePrefix + "aggregated_proof.output.json");
    }
    // Save proof to file
    if (config.saveProofToFile)
    {
        jProofRecursive2["publics"] = zkinInputRecursive2["publics"];
        json2file(jProofRecursive2, pProverRequest->filePrefix + "aggregated_proof.proof.json");
    }

    // Add the recursive2 verification key
    json publicsJson = json::array();

    file2json(config.recursive2Verkey, recursive2Verkey);

    for (uint64_t i = 0; i < starkZkevm->starkInfo.nPublics; i++)
    {
        publicsJson[i] = zkinInputRecursive2["publics"][i];
    }
    // Add the recursive2 verification key
    publicsJson[44] = to_string(recursive2Verkey["constRoot"][0]);
    publicsJson[45] = to_string(recursive2Verkey["constRoot"][1]);
    publicsJson[46] = to_string(recursive2Verkey["constRoot"][2]);
    publicsJson[47] = to_string(recursive2Verkey["constRoot"][3]);

    json2file(publicsJson, pProverRequest->publicsOutputFile());

    pProverRequest->result = ZKR_SUCCESS;

    TimerStopAndLog(PROVER_AGGREGATED_PROOF);
}

void Prover::genFinalProof(ProverRequest *pProverRequest)
{
    zkassert(config.generateProof());
    zkassert(pProverRequest != NULL);
    zkassert(pProverRequest->type == prt_genFinalProof);

    TimerStart(PROVER_FINAL_PROOF);

    printMemoryInfo(true);
    printProcessInfo(true);

    // Save input to file
    if (config.saveInputToFile)
    {
        json2file(pProverRequest->finalProofInput, pProverRequest->filePrefix + "final_proof.input.json");
    }

    // Input is pProverRequest->finalProofInput (of type json)
    std::string strAddress = mpz_get_str(0, 16, pProverRequest->input.publicInputsExtended.publicInputs.aggregatorAddress.get_mpz_t());
    std::string strAddress10 = mpz_get_str(0, 10, pProverRequest->input.publicInputsExtended.publicInputs.aggregatorAddress.get_mpz_t());

    json zkinFinal = pProverRequest->finalProofInput;

    Goldilocks::Element publics[starksRecursiveF->starkInfo.nPublics];

    for (uint64_t i = 0; i < starksRecursiveF->starkInfo.nPublics; i++)
    {
        publics[i] = Goldilocks::fromString(zkinFinal["publics"][i]);
    }

    CommitPolsStarks cmPolsRecursiveF((uint8_t *)pAddress + starksRecursiveF->starkInfo.mapOffsets.section[cm1_n] * sizeof(Goldilocks::Element), (1 << starksRecursiveF->starkInfo.starkStruct.nBits), starksRecursiveF->starkInfo.nCm1);
    #if (PROVER_FORK_ID == 13) // fork_13
        CircomRecursiveFFork13::getCommitedPols(&cmPolsRecursiveF, config.recursivefVerifier, config.recursivefExec, zkinFinal, (1 << starksRecursiveF->starkInfo.starkStruct.nBits), starksRecursiveF->starkInfo.nCm1);
    #else
        #error "Invalid PROVER_FORK_ID"
    #endif
    

    // void *pointercmPolsRecursiveF = mapFile("config/recursivef/recursivef.commit", cmPolsRecursiveF.size(), true);
    // memcpy(pointercmPolsRecursiveF, cmPolsRecursiveF.address(), cmPolsRecursiveF.size());
    // unmapFile(pointercmPolsRecursiveF, cmPolsRecursiveF.size());

    //  ----------------------------------------------
    //  Generate Recursive Final proof
    //  ----------------------------------------------

    TimerStart(STARK_RECURSIVE_F_PROOF_BATCH_PROOF);
    uint64_t polBitsRecursiveF = starksRecursiveF->starkInfo.starkStruct.steps[starksRecursiveF->starkInfo.starkStruct.steps.size() - 1].nBits;
    FRIProofC12 fproofRecursiveF((1 << polBitsRecursiveF), FIELD_EXTENSION, starksRecursiveF->starkInfo.starkStruct.steps.size(), starksRecursiveF->starkInfo.evMap.size(), starksRecursiveF->starkInfo.nPublics);
    if(USE_GENERIC_PARSER) {
        #if defined(__USE_CUDA__) && defined(ENABLE_EXPERIMENTAL_CODE)
            CHelpersStepsGPU cHelpersSteps; 
        #elif defined(__AVX512__)
            CHelpersStepsAvx512 cHelpersSteps;
        #elif defined(__PACK__) 
            CHelpersStepsPack cHelpersSteps;
            cHelpersSteps.nrowsPack = NROWS_PACK;
        #else
            CHelpersSteps cHelpersSteps;
        #endif
        starksRecursiveF->genProof(fproofRecursiveF, publics, &cHelpersSteps);
    } else {
        RecursiveFSteps recursiveFChelpersSteps;
        starksRecursiveF->genProof(fproofRecursiveF, publics, &recursiveFChelpersSteps);
    }
    
    TimerStopAndLog(STARK_RECURSIVE_F_PROOF_BATCH_PROOF);

    // Save the proof & zkinproof
    nlohmann::ordered_json jProofRecursiveF = fproofRecursiveF.proofs.proof2json();
    json zkinRecursiveF = proof2zkinStark(jProofRecursiveF);
    zkinRecursiveF["publics"] = zkinFinal["publics"];
    zkinRecursiveF["aggregatorAddr"] = strAddress10;

    // Save proof to file
    if (config.saveProofToFile)
    {
        json2file(zkinRecursiveF["publics"], pProverRequest->filePrefix + "publics.json");

        jProofRecursiveF["publics"] = zkinRecursiveF["publics"];
        json2file(jProofRecursiveF, pProverRequest->filePrefix + "recursivef.proof.json");
    }

    //  ----------------------------------------------
    //  Verifier final
    //  ----------------------------------------------

    #if (PROVER_FORK_ID == 13) // fork_13
        TimerStart(CIRCOM_LOAD_CIRCUIT_FINAL);
        CircomFinalFork13::Circom_Circuit *circuitFinal = CircomFinalFork13::loadCircuit(config.finalVerifier);
        TimerStopAndLog(CIRCOM_LOAD_CIRCUIT_FINAL);

        TimerStart(CIRCOM_FINAL_LOAD_JSON);
        CircomFinalFork13::Circom_CalcWit *ctxFinal = new CircomFinalFork13::Circom_CalcWit(circuitFinal);

        CircomFinalFork13::loadJsonImpl(ctxFinal, zkinRecursiveF);
        if (ctxFinal->getRemaingInputsToBeSet() != 0)
        {
            zklog.error("Prover::genFinalProof() Not all inputs have been set. Only " + to_string(CircomFinalFork13::get_main_input_signal_no() - ctxFinal->getRemaingInputsToBeSet()) + " out of " + to_string(CircomFinalFork13::get_main_input_signal_no()));
            exitProcess();
        }
        TimerStopAndLog(CIRCOM_FINAL_LOAD_JSON);

        TimerStart(CIRCOM_GET_BIN_WITNESS_FINAL);
        AltBn128::FrElement *pWitnessFinal = NULL;
        uint64_t witnessSizeFinal = 0;
        CircomFinalFork13::getBinWitness(ctxFinal, pWitnessFinal, witnessSizeFinal);
        CircomFinalFork13::freeCircuit(circuitFinal);
        delete ctxFinal;

        TimerStopAndLog(CIRCOM_GET_BIN_WITNESS_FINAL);
    #else
        #error "Invalid PROVER_FORK_ID"
    #endif

    TimerStart(SAVE_PUBLICS_JSON);
    // Save public file
    json publicJson;
    AltBn128::FrElement aux;
    AltBn128::Fr.toMontgomery(aux, pWitnessFinal[1]);
    publicJson[0] = AltBn128::Fr.toString(aux);
    json2file(publicJson, pProverRequest->publicsOutputFile());
    TimerStopAndLog(SAVE_PUBLICS_JSON);

    TimerStart(PROVER_INIT_FFLONK);

    prover = new Fflonk::FflonkProver<AltBn128::Engine>(AltBn128::Engine::engine, pAddress, polsSize, true);

    uint64_t lengthPrecomputedBuffer = prover->getLengthPrecomputedBuffer(domainSizeFflonk, nPublicsFflonk);
    FrElement* binPointer = (FrElement *)pAddress + lengthPrecomputedBuffer;

    zkey = BinFileUtils::openExisting(config.finalStarkZkey, "zkey", 1, binPointer, polsSize - lengthPrecomputedBuffer);
    protocolId = Zkey::getProtocolIdFromZkey(zkey.get());
    if(protocolId != Zkey::FFLONK_PROTOCOL_ID) {
        zklog.error("Prover::genFinalProof() zkey protocolId has to be Fflonk");
        exitProcess();
    }
    
    prover->setZkey(zkey.get());

    BinFileUtils::BinFile *pZkey = zkey.release();
    assert(zkey.get() == nullptr);
    assert(zkey == nullptr);
    delete pZkey;

    TimerStopAndLog(PROVER_INIT_FFLONK);

    TimerStart(RAPID_SNARK);
    try
    {
        auto [jsonProof, publicSignalsJson] = prover->prove(pWitnessFinal);
        // Save proof to file
        if (config.saveProofToFile)
        {
            json2file(jsonProof, pProverRequest->filePrefix + "final_proof.proof.json");
        }
        TimerStopAndLog(RAPID_SNARK);

        // Populate Proof with the correct data
        PublicInputsExtended publicInputsExtended;
        publicInputsExtended.publicInputs = pProverRequest->input.publicInputsExtended.publicInputs;
        pProverRequest->proof.load(jsonProof, publicSignalsJson);

        pProverRequest->result = ZKR_SUCCESS;
    }
    catch (std::exception &e)
    {
        zklog.error("Prover::genFinalProof() got exception in rapid SNARK:" + string(e.what()));
        exitProcess();
    }

    /***********/
    /* Cleanup */
    /***********/
    delete prover;

    free(pWitnessFinal);

    TimerStopAndLog(PROVER_FINAL_PROOF);
}

void Prover::execute(ProverRequest *pProverRequest)
{
    zkassert(!config.generateProof());
    zkassert(pProverRequest != NULL);

    TimerStart(PROVER_EXECUTE);

    printMemoryInfo(true);
    printProcessInfo(true);

    zkassert(pProverRequest != NULL);

    zklog.info("Prover::execute() timestamp: " + pProverRequest->timestamp);
    zklog.info("Prover::execute() UUID: " + pProverRequest->uuid);
    zklog.info("Prover::execute() input file: " + pProverRequest->inputFile());
    // zklog.info("Prover::execute() public file: " + pProverRequest->publicsOutputFile());
    // zklog.info("Prover::execute() proof file: " + pProverRequest->proofFile());

    // In proof-generation executions we can only process the exact number of steps
    if (pProverRequest->input.stepsN > 0)
    {
        zklog.error("Prover::execute() called with input.stepsN=" + to_string(pProverRequest->input.stepsN));
        exitProcess();
    }

    // Save input to <timestamp>.input.json, as provided by client
    if (config.saveInputToFile)
    {
        json inputJson;
        pProverRequest->input.save(inputJson);
        json2file(inputJson, pProverRequest->inputFile());
    }

    /*******************/
    /* Allocate memory */
    /*******************/

    // Allocate an area of memory, mapped to file, to store all the committed polynomials,
    // and create them using the allocated address
    uint64_t commitPolsSize = PROVER_FORK_NAMESPACE::CommitPols::numPols()*sizeof(Goldilocks::Element)*N;
    void *pExecuteAddress = NULL;

    if (config.zkevmCmPols.size() > 0)
    {
        pExecuteAddress = mapFile(config.zkevmCmPols, commitPolsSize, true);
        zklog.info("Prover::execute() successfully mapped " + to_string(commitPolsSize) + " bytes to file " + config.zkevmCmPols);
    }
    else
    {
        pExecuteAddress = calloc_zkevm(polsSize, 1);
        if (pExecuteAddress == NULL)
        {
            zklog.error("Prover::execute() failed calling calloc() of size " + to_string(commitPolsSize));
            exitProcess();
        }
        zklog.info("Prover::execute() successfully allocated " + to_string(commitPolsSize) + " bytes");
    }

    /************/
    /* Executor */
    /************/

    PROVER_FORK_NAMESPACE::CommitPols cmPols(pExecuteAddress, N);

    // Execute all the State Machines
    TimerStart(EXECUTOR_EXECUTE_EXECUTE);
    executor.executeBatch(*pProverRequest, cmPols);
    TimerStopAndLog(EXECUTOR_EXECUTE_EXECUTE);

    uint64_t lastN = N - 1;
    zklog.info("Prover::execute() called executor.execute() oldStateRoot=" + pProverRequest->input.publicInputsExtended.publicInputs.oldStateRoot.get_str(16) +
        " newStateRoot=" + pProverRequest->pFullTracer->get_new_state_root() +
        " pols.B[0]=" + fea2stringchain(fr, cmPols.Main.B0[0], cmPols.Main.B1[0], cmPols.Main.B2[0], cmPols.Main.B3[0], cmPols.Main.B4[0], cmPols.Main.B5[0], cmPols.Main.B6[0], cmPols.Main.B7[0]) +
        " pols.SR[lastN]=" + fea2stringchain(fr, cmPols.Main.SR0[lastN], cmPols.Main.SR1[lastN], cmPols.Main.SR2[lastN], cmPols.Main.SR3[lastN], cmPols.Main.SR4[lastN], cmPols.Main.SR5[lastN], cmPols.Main.SR6[lastN], cmPols.Main.SR7[lastN]) +
        " lastN=" + to_string(lastN));
    zklog.info("Prover::execute() called executor.execute() oldAccInputHash=" + pProverRequest->input.publicInputsExtended.publicInputs.oldAccInputHash.get_str(16) +
        " newAccInputHash=" + pProverRequest->pFullTracer->get_new_acc_input_hash() +
        " pols.C[0]=" + fea2stringchain(fr, cmPols.Main.C0[0], cmPols.Main.C1[0], cmPols.Main.C2[0], cmPols.Main.C3[0], cmPols.Main.C4[0], cmPols.Main.C5[0], cmPols.Main.C6[0], cmPols.Main.C7[0]) +
        " pols.D[lastN]=" + fea2stringchain(fr, cmPols.Main.D0[lastN], cmPols.Main.D1[lastN], cmPols.Main.D2[lastN], cmPols.Main.D3[lastN], cmPols.Main.D4[lastN], cmPols.Main.D5[lastN], cmPols.Main.D6[lastN], cmPols.Main.D7[lastN]) +
        " lastN=" + to_string(lastN));

    // Save input to <timestamp>.input.json after execution including dbReadLog
    if (config.saveDbReadsToFile)
    {
        json inputJsonEx;
        pProverRequest->input.save(inputJsonEx, *pProverRequest->dbReadLog);
        json2file(inputJsonEx, pProverRequest->inputDbFile());
    }

    // Save commit pols to file zkevm.commit
    if (config.zkevmCmPolsAfterExecutor != "")
    {
        TimerStart(PROVER_EXECUTE_SAVE_COMMIT_POLS_AFTER_EXECUTOR);
        void *pointerCmPols = mapFile(config.zkevmCmPolsAfterExecutor, cmPols.size(), true);
        memcpy(pointerCmPols, cmPols.address(), cmPols.size());
        unmapFile(pointerCmPols, cmPols.size());
        TimerStopAndLog(PROVER_EXECUTE_SAVE_COMMIT_POLS_AFTER_EXECUTOR);
    }

    /***************/
    /* Free memory */
    /***************/

    // Unmap committed polynomials address
    if (config.zkevmCmPols.size() > 0)
    {
        unmapFile(pExecuteAddress, commitPolsSize);
    }
    else
    {
        free_zkevm(pExecuteAddress);
    }

    TimerStopAndLog(PROVER_EXECUTE);
}
