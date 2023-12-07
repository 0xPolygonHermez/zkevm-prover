#include <fstream>
#include <iomanip>
#include <unistd.h>
#include "prover.hpp"
#include "utils.hpp"
#include "scalar.hpp"
#include "proof2zkin.hpp"
#include "main.hpp"
#include "main.recursive1.hpp"
#include "main.recursive2.hpp"
#include "main.recursiveF.hpp"
#include "main.final.hpp"
#include "binfile_utils.hpp"
#include "zkey_utils.hpp"
#include "wtns_utils.hpp"
#include "groth16.hpp"
#include "sm/storage/storage_executor.hpp"
#include "timer.hpp"
#include "execFile.hpp"
#include <math.h> /* log2 */
#include "proof2zkinStark.hpp"

#include "friProofC12.hpp"
#include <algorithm> // std::min
#include <openssl/sha.h>

#include "commit_pols_starks.hpp"
#include "zkevmSteps.hpp"
#include "c12aSteps.hpp"
#include "recursive1Steps.hpp"
#include "recursive2Steps.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"

#ifndef __AVX512__
#define NROWS_STEPS_ 4
#else
#define NROWS_STEPS_ 8
#endif

Prover::Prover(Goldilocks &fr,
               PoseidonGoldilocks &poseidon,
               const Config &config) : fr(fr),
                                       poseidon(poseidon),
                                       executor(fr, config, poseidon),
                                       pCurrentRequest(NULL),
                                       config(config),
                                       lastComputedRequestEndTime(0)
{
    mpz_init(altBbn128r);
    mpz_set_str(altBbn128r, "21888242871839275222246405745257275088548364400416034343698204186575808495617", 10);

    try
    {
        if (config.generateProof())
        {
            zkey = BinFileUtils::openExisting(config.finalStarkZkey, "zkey", 1);
            protocolId = Zkey::getProtocolIdFromZkey(zkey.get());
            if (Zkey::GROTH16_PROTOCOL_ID == protocolId)
            {
                zkeyHeader = ZKeyUtils::loadHeader(zkey.get());

                if (mpz_cmp(zkeyHeader->rPrime, altBbn128r) != 0)
                {
                    throw std::invalid_argument("zkey curve not supported");
                }

                groth16Prover = Groth16::makeProver<AltBn128::Engine>(
                    zkeyHeader->nVars,
                    zkeyHeader->nPublic,
                    zkeyHeader->domainSize,
                    zkeyHeader->nCoefs,
                    zkeyHeader->vk_alpha1,
                    zkeyHeader->vk_beta1,
                    zkeyHeader->vk_beta2,
                    zkeyHeader->vk_delta1,
                    zkeyHeader->vk_delta2,
                    zkey->getSectionData(4), // Coefs
                    zkey->getSectionData(5), // pointsA
                    zkey->getSectionData(6), // pointsB1
                    zkey->getSectionData(7), // pointsB2
                    zkey->getSectionData(8), // pointsC
                    zkey->getSectionData(9)  // pointsH1
                );
            }

            lastComputedRequestEndTime = 0;

            sem_init(&pendingRequestSem, 0, 0);
            pthread_mutex_init(&mutex, NULL);
            pCurrentRequest = NULL;
            pthread_create(&proverPthread, NULL, proverThread, this);
            pthread_create(&cleanerPthread, NULL, cleanerThread, this);

            StarkInfo _starkInfo(config, config.zkevmStarkInfo);

            // Allocate an area of memory, mapped to file, to store all the committed polynomials,
            // and create them using the allocated address
            uint64_t polsSize = _starkInfo.mapTotalN * sizeof(Goldilocks::Element) + _starkInfo.mapSectionsN.section[eSection::cm3_2ns] * (1 << _starkInfo.starkStruct.nBitsExt) * sizeof(Goldilocks::Element);

            zkassert(_starkInfo.mapSectionsN.section[eSection::cm1_2ns] * sizeof(Goldilocks::Element) <= polsSize - _starkInfo.mapSectionsN.section[eSection::cm2_2ns] * sizeof(Goldilocks::Element));

            zkassert(PROVER_FORK_NAMESPACE::CommitPols::pilSize() <= polsSize);
            zkassert(PROVER_FORK_NAMESPACE::CommitPols::pilSize() == _starkInfo.mapOffsets.section[cm2_n] * sizeof(Goldilocks::Element));

            if (config.zkevmCmPols.size() > 0)
            {
                pAddress = mapFile(config.zkevmCmPols, polsSize, true);
                zklog.info("Prover::genBatchProof() successfully mapped " + to_string(polsSize) + " bytes to file " + config.zkevmCmPols);
            }
            else
            {
                pAddress = calloc(polsSize, 1);
                if (pAddress == NULL)
                {
                    zklog.error("Prover::genBatchProof() failed calling malloc() of size " + to_string(polsSize));
                    exitProcess();
                }
                zklog.info("Prover::genBatchProof() successfully allocated " + to_string(polsSize) + " bytes");
            }

            prover = new Fflonk::FflonkProver<AltBn128::Engine>(AltBn128::Engine::engine, pAddress, polsSize);
            prover->setZkey(zkey.get());

            StarkInfo _starkInfoRecursiveF(config, config.recursivefStarkInfo);
            pAddressStarksRecursiveF = (void *)malloc(_starkInfoRecursiveF.mapTotalN * sizeof(Goldilocks::Element));

            starkZkevm = new Starks(config, {config.zkevmConstPols, config.mapConstPolsFile, config.zkevmConstantsTree, config.zkevmStarkInfo}, pAddress);
            starkZkevm->nrowsStepBatch = NROWS_STEPS_;
            starksC12a = new Starks(config, {config.c12aConstPols, config.mapConstPolsFile, config.c12aConstantsTree, config.c12aStarkInfo}, pAddress);
            starksRecursive1 = new Starks(config, {config.recursive1ConstPols, config.mapConstPolsFile, config.recursive1ConstantsTree, config.recursive1StarkInfo}, pAddress);
            starksRecursive2 = new Starks(config, {config.recursive2ConstPols, config.mapConstPolsFile, config.recursive2ConstantsTree, config.recursive2StarkInfo}, pAddress);
            starksRecursiveF = new StarkRecursiveF(config, pAddressStarksRecursiveF);
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
        Groth16::Prover<AltBn128::Engine> *pGroth16 = groth16Prover.release();
        BinFileUtils::BinFile *pZkey = zkey.release();
        ZKeyUtils::Header *pZkeyHeader = zkeyHeader.release();

        assert(groth16Prover.get() == nullptr);
        assert(groth16Prover == nullptr);
        assert(zkey.get() == nullptr);
        assert(zkey == nullptr);
        assert(zkeyHeader.get() == nullptr);
        assert(zkeyHeader == nullptr);

        delete pGroth16;
        delete pZkey;
        delete pZkeyHeader;

        uint64_t polsSize = starkZkevm->starkInfo.mapTotalN * sizeof(Goldilocks::Element) + starkZkevm->starkInfo.mapSectionsN.section[eSection::cm1_n] * (1 << starkZkevm->starkInfo.starkStruct.nBits) * FIELD_EXTENSION * sizeof(Goldilocks::Element);

        // Unmap committed polynomials address
        if (config.zkevmCmPols.size() > 0)
        {
            unmapFile(pAddress, polsSize);
        }
        else
        {
            free(pAddress);
        }
        free(pAddressStarksRecursiveF);

        delete prover;

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
    executor.process_batch(*pProverRequest);

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

    PROVER_FORK_NAMESPACE::CommitPols cmPols(pAddress, PROVER_FORK_NAMESPACE::CommitPols::pilDegree());
    uint64_t num_threads = omp_get_max_threads();
    uint64_t bytes_per_thread = cmPols.size() / num_threads;
#pragma omp parallel for num_threads(num_threads)
    for (uint64_t i = 0; i < cmPols.size(); i += bytes_per_thread) // Each iteration processes 64 bytes at a time
    {
        memset((uint8_t *)pAddress + i, 0, bytes_per_thread);
    }

    TimerStopAndLog(EXECUTOR_EXECUTE_INITIALIZATION);
    // Execute all the State Machines
    TimerStart(EXECUTOR_EXECUTE_BATCH_PROOF);
    executor.execute(*pProverRequest, cmPols);
    TimerStopAndLog(EXECUTOR_EXECUTE_BATCH_PROOF);

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

        uint64_t lastN = cmPols.pilDegree() - 1;

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

        TimerStart(STARK_PROOF_BATCH_PROOF);

        ZkevmSteps zkevmSteps;
        uint64_t polBits = starkZkevm->starkInfo.starkStruct.steps[starkZkevm->starkInfo.starkStruct.steps.size() - 1].nBits;
        FRIProof fproof((1 << polBits), FIELD_EXTENSION, starkZkevm->starkInfo.starkStruct.steps.size(), starkZkevm->starkInfo.evMap.size(), starkZkevm->starkInfo.nPublics);
        starkZkevm->genProof(fproof, &publics[0], zkevmVerkey, &zkevmSteps);

        TimerStopAndLog(STARK_PROOF_BATCH_PROOF);
        TimerStart(STARK_GEN_AND_CALC_WITNESS_C12A);
        TimerStart(STARK_JSON_GENERATION_BATCH_PROOF);

        nlohmann::ordered_json jProof = fproof.proofs.proof2json();
        nlohmann::json zkin = proof2zkinStark(jProof);
        // Generate publics
        jProof["publics"] = publicStarkJson;
        zkin["publics"] = publicStarkJson;

        TimerStopAndLog(STARK_JSON_GENERATION_BATCH_PROOF);

        CommitPolsStarks cmPols12a(pAddress, (1 << starksC12a->starkInfo.starkStruct.nBits), starksC12a->starkInfo.nCm1);

        Circom::getCommitedPols(&cmPols12a, config.zkevmVerifier, config.c12aExec, zkin, (1 << starksC12a->starkInfo.starkStruct.nBits), starksC12a->starkInfo.nCm1);

        // void *pointerCm12aPols = mapFile("config/c12a/c12a.commit", cmPols12a.size(), true);
        // memcpy(pointerCm12aPols, cmPols12a.address(), cmPols12a.size());
        // unmapFile(pointerCm12aPols, cmPols12a.size());

        //-------------------------------------------
        /* Generate C12a stark proof             */
        //-------------------------------------------
        TimerStopAndLog(STARK_GEN_AND_CALC_WITNESS_C12A);
        TimerStart(STARK_C12_A_PROOF_BATCH_PROOF);
        uint64_t polBitsC12 = starksC12a->starkInfo.starkStruct.steps[starksC12a->starkInfo.starkStruct.steps.size() - 1].nBits;
        FRIProof fproofC12a((1 << polBitsC12), FIELD_EXTENSION, starksC12a->starkInfo.starkStruct.steps.size(), starksC12a->starkInfo.evMap.size(), starksC12a->starkInfo.nPublics);

        // Generate the proof
        C12aSteps c12aSteps;

        starksC12a->genProof(fproofC12a, publics, c12aVerkey, &c12aSteps);

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

        CommitPolsStarks cmPolsRecursive1(pAddress, (1 << starksRecursive1->starkInfo.starkStruct.nBits), starksRecursive1->starkInfo.nCm1);
        CircomRecursive1::getCommitedPols(&cmPolsRecursive1, config.recursive1Verifier, config.recursive1Exec, zkinC12a, (1 << starksRecursive1->starkInfo.starkStruct.nBits), starksRecursive1->starkInfo.nCm1);

        // void *pointerCmRecursive1Pols = mapFile("config/recursive1/recursive1.commit", cmPolsRecursive1.size(), true);
        // memcpy(pointerCmRecursive1Pols, cmPolsRecursive1.address(), cmPolsRecursive1.size());
        // unmapFile(pointerCmRecursive1Pols, cmPolsRecursive1.size());

        //-------------------------------------------
        /* Generate Recursive 1 proof            */
        //-------------------------------------------

        TimerStart(STARK_RECURSIVE_1_PROOF_BATCH_PROOF);
        uint64_t polBitsRecursive1 = starksRecursive1->starkInfo.starkStruct.steps[starksRecursive1->starkInfo.starkStruct.steps.size() - 1].nBits;
        FRIProof fproofRecursive1((1 << polBitsRecursive1), FIELD_EXTENSION, starksRecursive1->starkInfo.starkStruct.steps.size(), starksRecursive1->starkInfo.evMap.size(), starksRecursive1->starkInfo.nPublics);
        Recursive1Steps recursive1Steps;
        starksRecursive1->genProof(fproofRecursive1, publics, recursive1Verkey, &recursive1Steps);
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

    CommitPolsStarks cmPolsRecursive2(pAddress, (1 << starksRecursive2->starkInfo.starkStruct.nBits), starksRecursive2->starkInfo.nCm1);
    CircomRecursive2::getCommitedPols(&cmPolsRecursive2, config.recursive2Verifier, config.recursive2Exec, zkinInputRecursive2, (1 << starksRecursive2->starkInfo.starkStruct.nBits), starksRecursive2->starkInfo.nCm1);

    // void *pointerCmRecursive2Pols = mapFile("config/recursive2/recursive2.commit", cmPolsRecursive2.size(), true);
    // memcpy(pointerCmRecursive2Pols, cmPolsRecursive2.address(), cmPolsRecursive2.size());
    // unmapFile(pointerCmRecursive2Pols, cmPolsRecursive2.size());

    //-------------------------------------------
    // Generate Recursive 2 proof
    //-------------------------------------------

    TimerStart(STARK_RECURSIVE_2_PROOF_BATCH_PROOF);
    uint64_t polBitsRecursive2 = starksRecursive2->starkInfo.starkStruct.steps[starksRecursive2->starkInfo.starkStruct.steps.size() - 1].nBits;
    FRIProof fproofRecursive2((1 << polBitsRecursive2), FIELD_EXTENSION, starksRecursive2->starkInfo.starkStruct.steps.size(), starksRecursive2->starkInfo.evMap.size(), starksRecursive2->starkInfo.nPublics);
    Recursive2Steps recursive2Steps;
    starksRecursive2->genProof(fproofRecursive2, publics, recursive2VerkeyValues, &recursive2Steps);
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

    CommitPolsStarks cmPolsRecursiveF(pAddressStarksRecursiveF, (1 << starksRecursiveF->starkInfo.starkStruct.nBits), starksRecursiveF->starkInfo.nCm1);
    CircomRecursiveF::getCommitedPols(&cmPolsRecursiveF, config.recursivefVerifier, config.recursivefExec, zkinFinal, (1 << starksRecursiveF->starkInfo.starkStruct.nBits), starksRecursiveF->starkInfo.nCm1);

    // void *pointercmPolsRecursiveF = mapFile("config/recursivef/recursivef.commit", cmPolsRecursiveF.size(), true);
    // memcpy(pointercmPolsRecursiveF, cmPolsRecursiveF.address(), cmPolsRecursiveF.size());
    // unmapFile(pointercmPolsRecursiveF, cmPolsRecursiveF.size());

    //  ----------------------------------------------
    //  Generate Recursive Final proof
    //  ----------------------------------------------

    TimerStart(STARK_RECURSIVE_F_PROOF_BATCH_PROOF);
    uint64_t polBitsRecursiveF = starksRecursiveF->starkInfo.starkStruct.steps[starksRecursiveF->starkInfo.starkStruct.steps.size() - 1].nBits;
    FRIProofC12 fproofRecursiveF((1 << polBitsRecursiveF), FIELD_EXTENSION, starksRecursiveF->starkInfo.starkStruct.steps.size(), starksRecursiveF->starkInfo.evMap.size(), starksRecursiveF->starkInfo.nPublics);
    starksRecursiveF->genProof(fproofRecursiveF, publics);
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

    TimerStart(CIRCOM_LOAD_CIRCUIT_FINAL);
    CircomFinal::Circom_Circuit *circuitFinal = CircomFinal::loadCircuit(config.finalVerifier);
    TimerStopAndLog(CIRCOM_LOAD_CIRCUIT_FINAL);

    TimerStart(CIRCOM_FINAL_LOAD_JSON);
    CircomFinal::Circom_CalcWit *ctxFinal = new CircomFinal::Circom_CalcWit(circuitFinal);

    CircomFinal::loadJsonImpl(ctxFinal, zkinRecursiveF);
    if (ctxFinal->getRemaingInputsToBeSet() != 0)
    {
        zklog.error("Prover::genProof() Not all inputs have been set. Only " + to_string(CircomFinal::get_main_input_signal_no() - ctxFinal->getRemaingInputsToBeSet()) + " out of " + to_string(CircomFinal::get_main_input_signal_no()));
        exitProcess();
    }
    TimerStopAndLog(CIRCOM_FINAL_LOAD_JSON);

    TimerStart(CIRCOM_GET_BIN_WITNESS_FINAL);
    AltBn128::FrElement *pWitnessFinal = NULL;
    uint64_t witnessSizeFinal = 0;
    CircomFinal::getBinWitness(ctxFinal, pWitnessFinal, witnessSizeFinal);
    CircomFinal::freeCircuit(circuitFinal);
    delete ctxFinal;

    TimerStopAndLog(CIRCOM_GET_BIN_WITNESS_FINAL);

    TimerStart(SAVE_PUBLICS_JSON);
    // Save public file
    json publicJson;
    AltBn128::FrElement aux;
    AltBn128::Fr.toMontgomery(aux, pWitnessFinal[1]);
    publicJson[0] = AltBn128::Fr.toString(aux);
    json2file(publicJson, pProverRequest->publicsOutputFile());
    TimerStopAndLog(SAVE_PUBLICS_JSON);

    if (Zkey::GROTH16_PROTOCOL_ID != protocolId)
    {
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
            zklog.error("Prover::genProof() got exception in rapid SNARK:" + string(e.what()));
            exitProcess();
        }
    }
    else
    {
        // Generate Groth16 via rapid SNARK
        TimerStart(RAPID_SNARK);
        json jsonProof;
        try
        {
            auto proof = groth16Prover->prove(pWitnessFinal);
            jsonProof = proof->toJson();
        }
        catch (std::exception &e)
        {
            zklog.error("Prover::genProof() got exception in rapid SNARK:" + string(e.what()));
            exitProcess();
        }
        TimerStopAndLog(RAPID_SNARK);

        // Save proof to file
        if (config.saveProofToFile)
        {
            json2file(jsonProof, pProverRequest->filePrefix + "final_proof.proof.json");
        }
        // Populate Proof with the correct data
        PublicInputsExtended publicInputsExtended;
        publicInputsExtended.publicInputs = pProverRequest->input.publicInputsExtended.publicInputs;
        pProverRequest->proof.load(jsonProof, publicJson);

        pProverRequest->result = ZKR_SUCCESS;
    }

    /***********/
    /* Cleanup */
    /***********/
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
    uint64_t polsSize = PROVER_FORK_NAMESPACE::CommitPols::pilSize();
    void *pExecuteAddress = NULL;

    if (config.zkevmCmPols.size() > 0)
    {
        pExecuteAddress = mapFile(config.zkevmCmPols, polsSize, true);
        zklog.info("Prover::execute() successfully mapped " + to_string(polsSize) + " bytes to file " + config.zkevmCmPols);
    }
    else
    {
        pExecuteAddress = calloc(polsSize, 1);
        if (pExecuteAddress == NULL)
        {
            zklog.error("Prover::execute() failed calling malloc() of size " + to_string(polsSize));
            exitProcess();
        }
        zklog.info("Prover::execute() successfully allocated " + to_string(polsSize) + " bytes");
    }

    /************/
    /* Executor */
    /************/

    PROVER_FORK_NAMESPACE::CommitPols cmPols(pExecuteAddress, PROVER_FORK_NAMESPACE::CommitPols::pilDegree());

    // Execute all the State Machines
    TimerStart(EXECUTOR_EXECUTE_EXECUTE);
    executor.execute(*pProverRequest, cmPols);
    TimerStopAndLog(EXECUTOR_EXECUTE_EXECUTE);

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
        unmapFile(pExecuteAddress, polsSize);
    }
    else
    {
        free(pExecuteAddress);
    }

    TimerStopAndLog(PROVER_EXECUTE);
}
