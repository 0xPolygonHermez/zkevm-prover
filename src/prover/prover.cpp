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
#include "main.blob_inner.hpp"
#include "main.blob_inner_recursive1.hpp"
#include "main.blob_outer.hpp"
#include "main.blob_outer_recursive2.hpp"
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
#include "ZkevmSteps.hpp"
#include "C12aSteps.hpp"
#include "Recursive1Steps.hpp"
#include "Recursive2Steps.hpp"
#include "RecursiveFSteps.hpp"
#include "BlobInnerSteps.hpp"
#include "BlobInnerCompressorSteps.hpp"
#include "BlobInnerRecursive1Steps.hpp"
#include "BlobOuterSteps.hpp"
#include "BlobOuterRecursive2Steps.hpp"
#include "chelpers_steps.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"


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

            StarkInfo _starkInfo(config.zkevmStarkInfo);

            // Allocate an area of memory, mapped to file, to store all the committed polynomials,
            // and create them using the allocated address

            polsSize = _starkInfo.mapTotalN * sizeof(Goldilocks::Element);
            if( _starkInfo.mapOffsets.section[eSection::cm1_2ns] < _starkInfo.mapOffsets.section[eSection::tmpExp_n]) optimizeMemoryNTTCommitPols = true;
            for(uint64_t i = 1; i <= 3; ++i) {
                std::string currentSection = "cm" + to_string(i) + "_n";
                std::string nextSectionExtended = i == 1 && optimizeMemoryNTTCommitPols ? "tmpExp_n" : "cm" + to_string(i + 1) + "_2ns";
                uint64_t nttHelperSize = _starkInfo.mapSectionsN.section[string2section(currentSection)] * (1 << _starkInfo.starkStruct.nBitsExt) * sizeof(Goldilocks::Element);
                uint64_t currentSectionStart = _starkInfo.mapOffsets.section[string2section(currentSection)] * sizeof(Goldilocks::Element);
                if (i == 3 && currentSectionStart > nttHelperSize) optimizeMemoryNTT = true;
                uint64_t nttHelperBufferStart = optimizeMemoryNTT && i==3 ? _starkInfo.mapOffsets.section[eSection::cm1_n] * sizeof(Goldilocks::Element) : _starkInfo.mapOffsets.section[string2section(nextSectionExtended)] * sizeof(Goldilocks::Element);
                uint64_t totalMemSize = nttHelperBufferStart + nttHelperSize;
                if(totalMemSize > polsSize) {
                    polsSize = totalMemSize;
                }
            }

            // Check that we have enough memory for stage2 H1H2 helpers (if not add memory)
            uint64_t stage2Start = _starkInfo.mapOffsets.section[cm2_2ns] * sizeof(Goldilocks::Element);
            uint64_t buffTransposedH1H2Size = 4 * _starkInfo.puCtx.size() * ((1 << _starkInfo.starkStruct.nBits) * FIELD_EXTENSION + 8);
            uint64_t buffHelperH1H2Size = (1 << _starkInfo.starkStruct.nBits) * _starkInfo.puCtx.size();
            uint64_t buffStage2HelperSize = (buffTransposedH1H2Size + buffHelperH1H2Size)*sizeof(Goldilocks::Element);
            if(stage2Start + buffStage2HelperSize > polsSize) {
                polsSize = stage2Start + buffStage2HelperSize;
            }
            
            // Check that we have enough memory for stage3 (Z) helpers (if not add memory)
            uint64_t stage3Start = _starkInfo.mapOffsets.section[cm3_2ns] * sizeof(Goldilocks::Element);
            uint64_t tot_pols = 3 * (_starkInfo.puCtx.size() + _starkInfo.peCtx.size() + _starkInfo.ciCtx.size());
            uint64_t buffStage3HelperSize = (tot_pols*((1 << _starkInfo.starkStruct.nBits) * FIELD_EXTENSION + 8)) * sizeof(Goldilocks::Element);
            if(stage3Start + buffStage3HelperSize > polsSize) {
                polsSize = stage3Start + buffStage3HelperSize;
            }

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
                    zklog.error("Prover::genBatchProof() failed calling calloc() of size " + to_string(polsSize));
                    exitProcess();
                }
                zklog.info("Prover::genBatchProof() successfully allocated " + to_string(polsSize) + " bytes");
            }

            prover = new Fflonk::FflonkProver<AltBn128::Engine>(AltBn128::Engine::engine, pAddress, polsSize);
            prover->setZkey(zkey.get());

            StarkInfo _starkInfoRecursiveF(config.recursivefStarkInfo);
            pAddressStarksRecursiveF = (void *)malloc(_starkInfoRecursiveF.mapTotalN * sizeof(Goldilocks::Element));
            if (pAddressStarksRecursiveF == NULL){
                zklog.error("Prover::Prover() failed calling malloc() of size " + to_string(_starkInfoRecursiveF.mapTotalN * sizeof(Goldilocks::Element)));
                exitProcess();
            }

            string zkevmCHelpers = USE_GENERIC_PARSER ? config.zkevmGenericCHelpers : config.zkevmCHelpers;
            string c12aCHelpers = USE_GENERIC_PARSER ? config.c12aGenericCHelpers : config.c12aCHelpers;
            string recursive1CHelpers = USE_GENERIC_PARSER ? config.recursive1GenericCHelpers : config.recursive1CHelpers;
            string recursive2CHelpers = USE_GENERIC_PARSER ? config.recursive2GenericCHelpers : config.recursive2CHelpers;
            string blobInnerCHelpers = USE_GENERIC_PARSER ? config.blobInnerGenericCHelpers : config.blobInnerCHelpers;
            string blobInnerCompressorCHelpers = USE_GENERIC_PARSER ? config.blobInnerCompressorGenericCHelpers : config.blobInnerCompressorCHelpers;
            string blobInnerRecursive1CHelpers = USE_GENERIC_PARSER ? config.blobInnerRecursive1GenericCHelpers : config.blobInnerRecursive1CHelpers;
            string blobOuterCHelpers = USE_GENERIC_PARSER ? config.blobOuterGenericCHelpers : config.blobOuterCHelpers;
            string blobOuterRecursive2CHelpers = USE_GENERIC_PARSER ? config.blobOuterRecursive2GenericCHelpers : config.blobOuterRecursive2CHelpers;
            
            starkBatch = new Starks(config, {config.zkevmConstPols, config.mapConstPolsFile, config.zkevmConstantsTree, config.zkevmStarkInfo, zkevmCHelpers}, pAddress);
            if(optimizeMemoryNTT) starkBatch->optimizeMemoryNTT = true;
            if(optimizeMemoryNTTCommitPols) starkBatch->optimizeMemoryNTTCommitPols = true;
            starkBatchC12a = new Starks(config, {config.c12aConstPols, config.mapConstPolsFile, config.c12aConstantsTree, config.c12aStarkInfo, c12aCHelpers}, pAddress);
            starkBatchRecursive1 = new Starks(config, {config.recursive1ConstPols, config.mapConstPolsFile, config.recursive1ConstantsTree, config.recursive1StarkInfo, recursive1CHelpers}, pAddress);
            starkBatchRecursive2 = new Starks(config, {config.recursive2ConstPols, config.mapConstPolsFile, config.recursive2ConstantsTree, config.recursive2StarkInfo, recursive2CHelpers}, pAddress);
            starksRecursiveF = new StarkRecursiveF(config, pAddressStarksRecursiveF);
            starkBlobInner = new Starks(config, {config.blobInnerConstPols, config.mapConstPolsFile, config.blobInnerConstantsTree, config.blobInnerStarkInfo, blobInnerCHelpers}, pAddress);
            starkBlobInnerCompressor = new Starks(config, {config.blobInnerCompressorConstPols, config.mapConstPolsFile, config.blobInnerCompressorConstantsTree, config.blobInnerCompressorStarkInfo, blobInnerCompressorCHelpers}, pAddress);
            starkBlobInnerRecursive1 = new Starks(config, {config.blobInnerRecursive1ConstPols, config.mapConstPolsFile, config.blobInnerRecursive1ConstantsTree, config.blobInnerRecursive1StarkInfo, blobInnerRecursive1CHelpers}, pAddress);
            starkBlobOuter = new Starks(config, {config.blobOuterConstPols, config.mapConstPolsFile, config.blobOuterConstantsTree, config.blobOuterStarkInfo, blobOuterCHelpers}, pAddress);
            starkBlobOuterRecursive2 = new Starks(config, {config.blobOuterRecursive2ConstPols, config.mapConstPolsFile, config.blobOuterRecursive2ConstantsTree, config.blobOuterRecursive2StarkInfo, blobOuterRecursive2CHelpers}, pAddress);
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

        delete starkBatch;
        delete starkBatchC12a;
        delete starkBatchRecursive1;
        delete starkBatchRecursive2;
        delete starksRecursiveF;
        delete starkBlobInner;
        delete starkBlobInnerCompressor;
        delete starkBlobInnerRecursive1;
        delete starkBlobOuter;
        delete starkBlobOuterRecursive2;
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
        case prt_genAggregatedBatchProof:
            pProver->genAggregatedBatchProof(pProver->pCurrentRequest);
            break;
        case prt_genBlobInnerProof:
            pProver->genBlobInnerProof(pProver->pCurrentRequest);
            break;
        case prt_genBlobOuterProof:
            pProver->genBlobOuterProof(pProver->pCurrentRequest);
            break;
        case prt_genAggregatedBlobOuterProof:
            pProver->genAggregatedBlobOuterProof(pProver->pCurrentRequest);
            break;
        case prt_genFinalProof:
            pProver->genFinalProof(pProver->pCurrentRequest);
            break;
        case prt_processBatch:
            pProver->processBatch(pProver->pCurrentRequest);
            break;
        case prt_executeBatch:
            pProver->executeBatch(pProver->pCurrentRequest);
            break;
        case prt_processBlobInner:
            pProver->processBlobInner(pProver->pCurrentRequest);
            break;
        case prt_executeBlobInner:
            pProver->executeBlobInner(pProver->pCurrentRequest);
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

// returns UUID for this request
string Prover::submitRequest(ProverRequest *pProverRequest) 
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

// wait for the request with this UUID to complete; returns NULL if UUID is invalid
ProverRequest *Prover::waitForRequestToComplete(const string &uuid, const uint64_t timeoutInSeconds) 
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

void Prover::processBatch (ProverRequest *pProverRequest)
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

void Prover::processBlobInner (ProverRequest *pProverRequest)
{
    //TimerStart(PROVER_PROCESS_BLOB_INNER);
    zkassert(pProverRequest != NULL);
    zkassert(pProverRequest->type == prt_processBlobInner);

    if (config.runAggregatorClient)
    {
        zklog.info("Prover::processBlobInner() timestamp=" + pProverRequest->timestamp + " UUID=" + pProverRequest->uuid);
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
    executor.processBlobInner(*pProverRequest);

    //TimerStopAndLog(PROVER_PROCESS_BLOB_INNER);
}

void Prover::genBatchProof (ProverRequest *pProverRequest)
{
    zkassert(config.generateProof());
    zkassert(pProverRequest != NULL);
    zkassert(pProverRequest->type == prt_genBatchProof);


    TimerStart(PROVER_BATCH_PROOF);

    printMemoryInfo(true);
    printProcessInfo(true);

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
    Goldilocks::parSetZero((Goldilocks::Element*)pAddress, cmPols.size()/sizeof(Goldilocks::Element), omp_get_max_threads()/2);
    uint64_t lastN = cmPols.pilDegree() - 1;
    TimerStopAndLog(EXECUTOR_EXECUTE_INITIALIZATION);

    // Execute all the State Machines
    TimerStart(EXECUTOR_EXECUTE_BATCH_PROOF);
    executor.executeBatch(*pProverRequest, cmPols);
    logBatchExecutionInfo(cmPols, pProverRequest);
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

        Goldilocks::Element publics[starkBatchRecursive1->starkInfo.nPublics];
        zkassert(starkBatchRecursive1->starkInfo.nPublics == 65);

        // oldStateRoot
        publics[0] = cmPols.Main.SR0[0];
        publics[1] = cmPols.Main.SR1[0];
        publics[2] = cmPols.Main.SR2[0];
        publics[3] = cmPols.Main.SR3[0];
        publics[4] = cmPols.Main.SR4[0];
        publics[5] = cmPols.Main.SR5[0];
        publics[6] = cmPols.Main.SR6[0];
        publics[7] = cmPols.Main.SR7[0];

        // oldBatchAccInputHash
        publics[8] = cmPols.Main.C0[0];
        publics[9] = cmPols.Main.C1[0];
        publics[10] = cmPols.Main.C2[0];
        publics[11] = cmPols.Main.C3[0];
        publics[12] = cmPols.Main.C4[0];
        publics[13] = cmPols.Main.C5[0];
        publics[14] = cmPols.Main.C6[0];
        publics[15] = cmPols.Main.C7[0];

        // previousL1InfoTreeRoot
        publics[16] = cmPols.Main.D0[0];
        publics[17] = cmPols.Main.D1[0];
        publics[18] = cmPols.Main.D2[0];
        publics[19] = cmPols.Main.D3[0];
        publics[20] = cmPols.Main.D4[0];
        publics[21] = cmPols.Main.D5[0];
        publics[22] = cmPols.Main.D6[0];
        publics[23] = cmPols.Main.D7[0];

        // previousL1InfoTreeIndex
        publics[24] = cmPols.Main.RCX[0];

        // chainId
        publics[25] = cmPols.Main.GAS[0];

        // forkid
        publics[26] = cmPols.Main.CTX[0];

        //newStateRoot
        publics[27] = cmPols.Main.SR0[lastN];
        publics[28] = cmPols.Main.SR1[lastN];
        publics[29] = cmPols.Main.SR2[lastN];
        publics[30] = cmPols.Main.SR3[lastN];
        publics[31] = cmPols.Main.SR4[lastN];
        publics[32] = cmPols.Main.SR5[lastN];
        publics[33] = cmPols.Main.SR6[lastN];
        publics[34] = cmPols.Main.SR7[lastN];

        //newBatchAccInputHash
        publics[35] = cmPols.Main.C0[lastN];
        publics[36] = cmPols.Main.C1[lastN];
        publics[37] = cmPols.Main.C2[lastN];
        publics[38] = cmPols.Main.C3[lastN];
        publics[39] = cmPols.Main.C4[lastN];
        publics[40] = cmPols.Main.C5[lastN];
        publics[41] = cmPols.Main.C6[lastN];
        publics[42] = cmPols.Main.C7[lastN];

        //currentL1InfoTreeRoot
        publics[43] = cmPols.Main.D0[lastN];
        publics[44] = cmPols.Main.D1[lastN];
        publics[45] = cmPols.Main.D2[lastN];
        publics[46] = cmPols.Main.D3[lastN];
        publics[47] = cmPols.Main.D4[lastN];
        publics[48] = cmPols.Main.D5[lastN];
        publics[49] = cmPols.Main.D6[lastN];
        publics[50] = cmPols.Main.D7[lastN];

        // currentL1InfoTreeIndex
        publics[51] = cmPols.Main.RCX[lastN];

        // newLocalExitRoot
        publics[52] = cmPols.Main.E0[lastN];
        publics[53] = cmPols.Main.E1[lastN];
        publics[54] = cmPols.Main.E2[lastN];
        publics[55] = cmPols.Main.E3[lastN];
        publics[56] = cmPols.Main.E4[lastN];
        publics[57] = cmPols.Main.E5[lastN];
        publics[58] = cmPols.Main.E6[lastN];
        publics[59] = cmPols.Main.E7[lastN];

        // newLastTimeStamp
        publics[60] = cmPols.Main.RR[lastN];

        json recursive2VerkeyJson;
        file2json(config.recursive2Verkey, recursive2VerkeyJson);
        publics[61] = Goldilocks::fromU64(recursive2VerkeyJson["constRoot"][0]);
        publics[62] = Goldilocks::fromU64(recursive2VerkeyJson["constRoot"][1]);
        publics[63] = Goldilocks::fromU64(recursive2VerkeyJson["constRoot"][2]);
        publics[64] = Goldilocks::fromU64(recursive2VerkeyJson["constRoot"][3]);

        json publicStarkJson;
        for (uint64_t i = 0; i < starkBatch->starkInfo.nPublics; i++)
        {
            publicStarkJson[i] = Goldilocks::toString(publics[i]);
        }

        TimerStopAndLog(SAVE_PUBLICS_JSON_BATCH_PROOF);

        /*************************************/
        /*  Generate stark proof            */
        /*************************************/
        CHelpersSteps cHelpersSteps;
        
        TimerStart(STARK_PROOF_BATCH_PROOF);
        ZkevmSteps zkevmChelpersSteps;
        uint64_t polBits = starkBatch->getPolBits();
        FRIProof fproof((1 << polBits), FIELD_EXTENSION, starkBatch->starkInfo.starkStruct.steps.size(), starkBatch->starkInfo.evMap.size(), starkBatch->starkInfo.nPublics);
        if(USE_GENERIC_PARSER) {
            starkBatch->genProof(fproof, &publics[0], zkevmVerkey, &cHelpersSteps);
        } else {
            starkBatch->genProof(fproof, &publics[0], zkevmVerkey, &zkevmChelpersSteps);
        }
        
        TimerStopAndLog(STARK_PROOF_BATCH_PROOF);

        TimerStart(STARK_GEN_AND_CALC_WITNESS_C12A);
        TimerStart(STARK_JSON_GENERATION_BATCH_PROOF);

        nlohmann::ordered_json jProof = fproof.proofs.proof2json();
        nlohmann::json zkin = proof2zkinStark(jProof);
        // Generate publics
        zkin["publics"] = publicStarkJson;

        TimerStopAndLog(STARK_JSON_GENERATION_BATCH_PROOF);

        CommitPolsStarks cmPolsC12a(pAddress, (1 << starkBatchC12a->starkInfo.starkStruct.nBits), starkBatchC12a->starkInfo.nCm1);

        Circom::getCommitedPols(&cmPolsC12a, config.zkevmVerifier, config.c12aExec, zkin, (1 << starkBatchC12a->starkInfo.starkStruct.nBits), starkBatchC12a->starkInfo.nCm1);

        // void *pointerCm12aPols = mapFile("config/c12a/c12a.commit", cmPolsC12a.size(), true);
        // memcpy(pointerCm12aPols, cmPolsC12a.address(), cmPolsC12a.size());
        // unmapFile(pointerCm12aPols, cmPolsC12a.size());

        //-------------------------------------------
        /* Generate C12a stark proof             */
        //-------------------------------------------
        TimerStopAndLog(STARK_GEN_AND_CALC_WITNESS_C12A);
        TimerStart(STARK_C12_A_PROOF_BATCH_PROOF);
        C12aSteps c12aChelpersSteps;
        uint64_t polBitsC12 = starkBatchC12a->getPolBits();
        FRIProof fproofC12a((1 << polBitsC12), FIELD_EXTENSION, starkBatchC12a->starkInfo.starkStruct.steps.size(), starkBatchC12a->starkInfo.evMap.size(), starkBatchC12a->starkInfo.nPublics);

        // Generate the proof
        if(USE_GENERIC_PARSER) {
            starkBatchC12a->genProof(fproofC12a, publics, c12aVerkey, &cHelpersSteps);
        } else {
            starkBatchC12a->genProof(fproofC12a, publics, c12aVerkey, &c12aChelpersSteps);
        }

        TimerStopAndLog(STARK_C12_A_PROOF_BATCH_PROOF);
        TimerStart(STARK_JSON_GENERATION_BATCH_PROOF_C12A);

        // Save the proof & zkinproof
        nlohmann::ordered_json jProofc12a = fproofC12a.proofs.proof2json();
        nlohmann::json zkinC12a = proof2zkinStark(jProofc12a);

        // Add the recursive2 verification key
        json rootC;
        rootC[0] = to_string(recursive2VerkeyJson["constRoot"][0]);
        rootC[1] = to_string(recursive2VerkeyJson["constRoot"][1]);
        rootC[2] = to_string(recursive2VerkeyJson["constRoot"][2]);
        rootC[3] = to_string(recursive2VerkeyJson["constRoot"][3]);
        zkinC12a["publics"] = publicStarkJson;
        zkinC12a["rootC"] = rootC;
        TimerStopAndLog(STARK_JSON_GENERATION_BATCH_PROOF_C12A);

        CommitPolsStarks cmPolsRecursive1(pAddress, (1 << starkBatchRecursive1->starkInfo.starkStruct.nBits), starkBatchRecursive1->starkInfo.nCm1);
        CircomRecursive1::getCommitedPols(&cmPolsRecursive1, config.recursive1Circuit, config.recursive1Exec, zkinC12a, (1 << starkBatchRecursive1->starkInfo.starkStruct.nBits), starkBatchRecursive1->starkInfo.nCm1);

        // void *pointerCmRecursive1Pols = mapFile("config/recursive1/recursive1.commit", cmPolsRecursive1.size(), true);
        // memcpy(pointerCmRecursive1Pols, cmPolsRecursive1.address(), cmPolsRecursive1.size());
        // unmapFile(pointerCmRecursive1Pols, cmPolsRecursive1.size());

        //-------------------------------------------
        /* Generate Recursive 1 proof            */
        //-------------------------------------------

        TimerStart(STARK_RECURSIVE_1_PROOF_BATCH_PROOF);
        Recursive1Steps recursive1ChelpersSteps;
        uint64_t polBitsRecursive1 = starkBatchRecursive1->getPolBits();
        FRIProof fproofRecursive1((1 << polBitsRecursive1), FIELD_EXTENSION, starkBatchRecursive1->starkInfo.starkStruct.steps.size(), starkBatchRecursive1->starkInfo.evMap.size(), starkBatchRecursive1->starkInfo.nPublics);
        if(USE_GENERIC_PARSER) {
            starkBatchRecursive1->genProof(fproofRecursive1, publics, recursive1Verkey, &cHelpersSteps);
        } else {
            starkBatchRecursive1->genProof(fproofRecursive1, publics, recursive1Verkey, &recursive1ChelpersSteps);
        }
        
        TimerStopAndLog(STARK_RECURSIVE_1_PROOF_BATCH_PROOF);

        // Save the proof & zkinproof
        TimerStart(SAVE_PROOF);

        nlohmann::ordered_json jProofRecursive1 = fproofRecursive1.proofs.proof2json();
        nlohmann::ordered_json zkinRecursive1 = proof2zkinStark(jProofRecursive1);
        zkinRecursive1["isAggregatedCircuit"] = "0";
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

void Prover::genBlobInnerProof (ProverRequest *pProverRequest){

    zkassert(config.generateProof());
    zkassert(pProverRequest != NULL);
    zkassert(pProverRequest->type == prt_genBlobInnerProof);


    TimerStart(PROVER_BLOB_INNER_PROOF);

    printMemoryInfo(true);
    printProcessInfo(true);

    zklog.info("Prover::genBlobInnerProof() timestamp: " + pProverRequest->timestamp);
    zklog.info("Prover::genBlobInnerProof() UUID: " + pProverRequest->uuid);
    zklog.info("Prover::genBlobInnerProof() input file: " + pProverRequest->inputFile());
    // zklog.info("Prover::genBlobInnerProof() public file: " + pProverRequest->publicsOutputFile());
    // zklog.info("Prover::genBlobInnerProof() proof file: " + pProverRequest->proofFile());

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
    TimerStart(EXECUTOR_EXECUTE_BLOB_INITIALIZATION);
    PROVER_FORK_NAMESPACE::CommitPols cmPols(pAddress, PROVER_FORK_NAMESPACE::CommitPols::pilDegree());
    Goldilocks::parSetZero((Goldilocks::Element*)pAddress, cmPols.size()/sizeof(Goldilocks::Element), omp_get_max_threads()/2);
    uint64_t lastN = cmPols.pilDegree() - 1;
    TimerStopAndLog(EXECUTOR_EXECUTE_BLOB_INITIALIZATION);

    // Execute all the State Machines
    TimerStart(EXECUTOR_EXECUTE_BLOB_INNER_PROOF);
    executor.executeBlobInner(*pProverRequest, cmPols);
    logBlobInnerExecutionInfo(cmPols,pProverRequest);
    TimerStopAndLog(EXECUTOR_EXECUTE_BLOB_INNER_PROOF);

    if (pProverRequest->result == ZKR_SUCCESS)
    {
        /*************************************/
        /*  Generate publics input           */
        /*************************************/
        TimerStart(SAVE_PUBLICS_JSON_BLOB_INNER_PROOF);

        json blobInnerVerkeyJson;
        file2json(config.blobInnerVerkey, blobInnerVerkeyJson);
        Goldilocks::Element blobInnerVerkey[4];
        blobInnerVerkey[0] = Goldilocks::fromU64(blobInnerVerkeyJson["constRoot"][0]);
        blobInnerVerkey[1] = Goldilocks::fromU64(blobInnerVerkeyJson["constRoot"][1]);
        blobInnerVerkey[2] = Goldilocks::fromU64(blobInnerVerkeyJson["constRoot"][2]);
        blobInnerVerkey[3] = Goldilocks::fromU64(blobInnerVerkeyJson["constRoot"][3]);

        json compressorVerkeyJson;
        file2json(config.blobInnerCompressorVerkey, compressorVerkeyJson);
        Goldilocks::Element compressorVerkey[4];
        compressorVerkey[0] = Goldilocks::fromU64(compressorVerkeyJson["constRoot"][0]);
        compressorVerkey[1] = Goldilocks::fromU64(compressorVerkeyJson["constRoot"][1]);
        compressorVerkey[2] = Goldilocks::fromU64(compressorVerkeyJson["constRoot"][2]);
        compressorVerkey[3] = Goldilocks::fromU64(compressorVerkeyJson["constRoot"][3]);

        json recursive1VerkeyJson;
        file2json(config.blobInnerRecursive1Verkey, recursive1VerkeyJson);
        Goldilocks::Element recursive1Verkey[4];
        recursive1Verkey[0] = Goldilocks::fromU64(recursive1VerkeyJson["constRoot"][0]);
        recursive1Verkey[1] = Goldilocks::fromU64(recursive1VerkeyJson["constRoot"][1]);
        recursive1Verkey[2] = Goldilocks::fromU64(recursive1VerkeyJson["constRoot"][2]);
        recursive1Verkey[3] = Goldilocks::fromU64(recursive1VerkeyJson["constRoot"][3]);

        Goldilocks::Element publics[starkBlobInner->starkInfo.nPublics];
        zkassert(starkBlobInner->starkInfo.nPublics == 70);

        // oldBlobStateRoot
        publics[0] = cmPols.Main.B0[0];
        publics[1] = cmPols.Main.B1[0];
        publics[2] = cmPols.Main.B2[0];
        publics[3] = cmPols.Main.B3[0];
        publics[4] = cmPols.Main.B4[0];
        publics[5] = cmPols.Main.B5[0];
        publics[6] = cmPols.Main.B6[0];
        publics[7] = cmPols.Main.B7[0];

        // oldBlobAccInputHash
        publics[8] = cmPols.Main.C0[0];
        publics[9] = cmPols.Main.C1[0];
        publics[10] = cmPols.Main.C2[0];
        publics[11] = cmPols.Main.C3[0];
        publics[12] = cmPols.Main.C4[0];
        publics[13] = cmPols.Main.C5[0];
        publics[14] = cmPols.Main.C6[0];
        publics[15] = cmPols.Main.C7[0];

        // oldNumBlob
        publics[16] = cmPols.Main.RR[0];

        // oldStateRoot
        publics[17] = cmPols.Main.D0[0];
        publics[18] = cmPols.Main.D1[0];
        publics[19] = cmPols.Main.D2[0];
        publics[20] = cmPols.Main.D3[0];
        publics[21] = cmPols.Main.D4[0];
        publics[22] = cmPols.Main.D5[0];
        publics[23] = cmPols.Main.D6[0];
        publics[24] = cmPols.Main.D7[0];

        // forkID
        publics[25] = cmPols.Main.RCX[0];

        // newBlobStateRoot
        publics[26] = cmPols.Main.B0[lastN];
        publics[27] = cmPols.Main.B1[lastN];
        publics[28] = cmPols.Main.B2[lastN];
        publics[29] = cmPols.Main.B3[lastN];
        publics[30] = cmPols.Main.B4[lastN];
        publics[31] = cmPols.Main.B5[lastN];
        publics[32] = cmPols.Main.B6[lastN];
        publics[33] = cmPols.Main.B7[lastN];

        // newBlobAccInputHash
        publics[34] = cmPols.Main.C0[lastN];
        publics[35] = cmPols.Main.C1[lastN];
        publics[36] = cmPols.Main.C2[lastN];
        publics[37] = cmPols.Main.C3[lastN];
        publics[38] = cmPols.Main.C4[lastN];
        publics[39] = cmPols.Main.C5[lastN];
        publics[40] = cmPols.Main.C6[lastN];
        publics[41] = cmPols.Main.C7[lastN];
        
        // newNumBlob
        publics[42] = cmPols.Main.GAS[lastN];

        // finalAccBatchHashData
        publics[43] = cmPols.Main.A0[lastN];
        publics[44] = cmPols.Main.A1[lastN];
        publics[45] = cmPols.Main.A2[lastN];
        publics[46] = cmPols.Main.A3[lastN];
        publics[47] = cmPols.Main.A4[lastN];
        publics[48] = cmPols.Main.A5[lastN];
        publics[49] = cmPols.Main.A6[lastN];
        publics[50] = cmPols.Main.A7[lastN];

        // localExitRootFromBlob
        publics[51] = cmPols.Main.E0[lastN];
        publics[52] = cmPols.Main.E1[lastN];
        publics[53] = cmPols.Main.E2[lastN];
        publics[54] = cmPols.Main.E3[lastN];
        publics[55] = cmPols.Main.E4[lastN];
        publics[56] = cmPols.Main.E5[lastN];
        publics[57] = cmPols.Main.E6[lastN];
        publics[58] = cmPols.Main.E7[lastN];

        // isInvalid
        publics[59] = cmPols.Main.CTX[lastN];

        // timestampLimit
        publics[60] = cmPols.Main.RR[lastN];

        // lastL1InfoTreeRoot
        publics[61] = cmPols.Main.D0[lastN];
        publics[62] = cmPols.Main.D1[lastN];
        publics[63] = cmPols.Main.D2[lastN];
        publics[64] = cmPols.Main.D3[lastN];
        publics[65] = cmPols.Main.D4[lastN];
        publics[66] = cmPols.Main.D5[lastN];
        publics[67] = cmPols.Main.D6[lastN];
        publics[68] = cmPols.Main.D7[lastN];

        // lastL1InfoTreeIndex
        publics[69] = cmPols.Main.RCX[lastN];

        json publicStarkJson;
        for (uint64_t i = 0; i < starkBlobInner->starkInfo.nPublics; i++)
        {
            publicStarkJson[i] = Goldilocks::toString(publics[i]);
        }

        TimerStopAndLog(SAVE_PUBLICS_JSON_BLOB_INNER_PROOF);

        /*************************************/
        /*  Generate stark proof            */
        /*************************************/

        CHelpersSteps cHelpersSteps;

        TimerStart(STARK_PROOF_BLOB_INNER_PROOF);
        BlobInnerSteps blobInnerChelpersSteps;
        uint64_t polBits = starkBlobInner->getPolBits();
        FRIProof fproof((1 << polBits), FIELD_EXTENSION, starkBlobInner->starkInfo.starkStruct.steps.size(), starkBlobInner->starkInfo.evMap.size(), starkBlobInner->starkInfo.nPublics);
        if(USE_GENERIC_PARSER) {
            starkBlobInner->genProof(fproof, &publics[0], blobInnerVerkey, &cHelpersSteps);
        } else {
            starkBlobInner->genProof(fproof, &publics[0], blobInnerVerkey, &blobInnerChelpersSteps);
        }
        
        TimerStopAndLog(STARK_PROOF_BLOB_INNER_PROOF);

        TimerStart(STARK_GEN_AND_CALC_WITNESS_COMPRESSOR);

        TimerStart(STARK_JSON_GENERATION_BLOB_INNER_PROOF);
        nlohmann::ordered_json jProof = fproof.proofs.proof2json();
        nlohmann::json zkin = proof2zkinStark(jProof);
        // Generate publics
        zkin["publics"] = publicStarkJson;
        TimerStopAndLog(STARK_JSON_GENERATION_BLOB_INNER_PROOF);

        CommitPolsStarks cmPolsCompressor(pAddress, (1 << starkBlobInnerCompressor->starkInfo.starkStruct.nBits), starkBlobInnerCompressor->starkInfo.nCm1);
        CircomBlobInner::getCommitedPols(&cmPolsCompressor, config.blobInnerVerifier, config.blobInnerCompressorExec, zkin, (1 << starkBlobInnerCompressor->starkInfo.starkStruct.nBits), starkBlobInnerCompressor->starkInfo.nCm1);

        // void *pointerCmCompressorPols = mapFile("config/blob_inner_compressor/blob_inner_compressor.commit", cmPolsCompressor.size(), true);
        // memcpy(pointerCmCompressorPols, cmPolsCompressor.address(), cmPolsCompressor.size());
        // unmapFile(pointerCmCompressorPols, cmPolsCompressor.size());

        //-----------------------------------------------
        /* Generate Compressor stark proof             */
        //-----------------------------------------------
        TimerStopAndLog(STARK_GEN_AND_CALC_WITNESS_COMPRESSOR);
        
        TimerStart(STARKCOMPRESSOR_PROOF_BLOB_INNNER_PROOF);
        BlobInnerCompressorSteps compressorChelpersSteps;
        uint64_t polBitsCompressor = starkBlobInnerCompressor->getPolBits();
        FRIProof fproofCompressor((1 << polBitsCompressor), FIELD_EXTENSION, starkBlobInnerCompressor->starkInfo.starkStruct.steps.size(), starkBlobInnerCompressor->starkInfo.evMap.size(), starkBlobInnerCompressor->starkInfo.nPublics);

        // Generate the proof
        if(USE_GENERIC_PARSER) {
            starkBlobInnerCompressor->genProof(fproofCompressor, publics, compressorVerkey, &cHelpersSteps);
        } else {
            starkBlobInnerCompressor->genProof(fproofCompressor, publics, compressorVerkey, &compressorChelpersSteps);
        }
        TimerStopAndLog(STARKCOMPRESSOR_PROOF_BLOB_INNNER_PROOF);

        TimerStart(STARK_JSON_GENERATION_BLOB_INNER_PROOF_COMPRESSOR);
        // Save the proof & zkinproof
        nlohmann::ordered_json jProofCompressor = fproofCompressor.proofs.proof2json();
        nlohmann::json zkinCompressor = proof2zkinStark(jProofCompressor);
        zkinCompressor["publics"] = publicStarkJson;
        TimerStopAndLog(STARK_JSON_GENERATION_BLOB_INNER_PROOF_COMPRESSOR);

        CommitPolsStarks cmPolsRecursive1(pAddress, (1 << starkBlobInnerRecursive1->starkInfo.starkStruct.nBits), starkBlobInnerRecursive1->starkInfo.nCm1);
        CircomBlobInnerRecursive1::getCommitedPols(&cmPolsRecursive1, config.blobInnerRecursive1Circuit, config.blobInnerRecursive1Exec, zkinCompressor, (1 << starkBlobInnerRecursive1->starkInfo.starkStruct.nBits), starkBlobInnerRecursive1->starkInfo.nCm1);

        // void *pointerCmRecursive1Pols = mapFile("config/blob_inner_recursive1/blob_inner_recursive1.commit", cmPolsRecursive1.size(), true);
        // memcpy(pointerCmRecursive1Pols, cmPolsRecursive1.address(), cmPolsRecursive1.size());
        // unmapFile(pointerCmRecursive1Pols, cmPolsRecursive1.size());

        //-------------------------------------------
        /* Generate Recursive 1 proof            */
        //-------------------------------------------

        TimerStart(STARK_RECURSIVE_1_PROOF_BLOB_INNER_PROOF);
        BlobInnerRecursive1Steps recursive1ChelpersSteps;
        uint64_t polBitsRecursive1 = starkBlobInnerRecursive1->getPolBits();
        FRIProof fproofRecursive1((1 << polBitsRecursive1), FIELD_EXTENSION, starkBlobInnerRecursive1->starkInfo.starkStruct.steps.size(), starkBlobInnerRecursive1->starkInfo.evMap.size(), starkBlobInnerRecursive1->starkInfo.nPublics);
        
        if(USE_GENERIC_PARSER) {
            starkBlobInnerRecursive1->genProof(fproofRecursive1, publics, recursive1Verkey, &cHelpersSteps);
        } else {
            starkBlobInnerRecursive1->genProof(fproofRecursive1, publics, recursive1Verkey, &recursive1ChelpersSteps);
        }
        TimerStopAndLog(STARK_RECURSIVE_1_PROOF_BLOB_INNER_PROOF);

        // Save the proof & zkinproof
        TimerStart(SAVE_PROOF);

        nlohmann::ordered_json jProofRecursive1 = fproofRecursive1.proofs.proof2json();
        nlohmann::ordered_json zkinRecursive1 = proof2zkinStark(jProofRecursive1);
        zkinRecursive1["publics"] = publicStarkJson;

        pProverRequest->blobInnerProofOutput = zkinRecursive1;

        // save publics to filestarks
        json2file(publicStarkJson, pProverRequest->publicsOutputFile());

        // Save output to file
        if (config.saveOutputToFile)
        {
            json2file(pProverRequest->blobInnerProofOutput, pProverRequest->filePrefix + "blob_inner_proof.output.json");
        }
        // Save proof to file
        if (config.saveProofToFile)
        {
            jProofRecursive1["publics"] = publicStarkJson;
            json2file(jProofRecursive1, pProverRequest->filePrefix + "blob_inner_proof.proof.json");
        }
        TimerStopAndLog(SAVE_PROOF);
    }

    TimerStopAndLog(PROVER_BLOB_INNER_PROOF);
}

void Prover::genBlobOuterProof (ProverRequest *pProverRequest){

    zkassert(config.generateProof());
    zkassert(pProverRequest != NULL);
    zkassert(pProverRequest->type == prt_genBlobOuterProof);

    TimerStart(PROVER_BLOB_OUTER_PROOF);

    printMemoryInfo(true);
    printProcessInfo(true);

    // Save input to file
    if (config.saveInputToFile)
    {
        json2file(pProverRequest->blobOuterProofInputBatch, pProverRequest->filePrefix + "blob_outer_proof.input_batch.json");
        json2file(pProverRequest->blobOuterProofInputBlobInner, pProverRequest->filePrefix + "blob_outer_proof.input_blob_inner.json");
    }

    // ----------------------------------------------
    // CHECKS
    // ----------------------------------------------
    
    BatchPublics batchPublics;
    BlobInnerPublics blobInnerPublics;

    bool isInvalidFinalAccBatchHashData = true;
    for(int i = 0; i < 8; i++) 
    {
        if(pProverRequest->blobOuterProofInputBlobInner["publics"][blobInnerPublics.finalAccBatchHashDataPos + i] != to_string(0)) {
            isInvalidFinalAccBatchHashData = false;
            break;
        }
    }

    bool isInvalidBlob = pProverRequest->blobOuterProofInputBlobInner["publics"][blobInnerPublics.isInvalidPos] == to_string(1);

    if( !isInvalidBlob && !isInvalidFinalAccBatchHashData) {
        // Check that final acc batch data is the same as the new batch acc input hash
        for (int i = 0; i < 8; i++)
        {
            if (pProverRequest->blobOuterProofInputBatch["publics"][batchPublics.newBatchAccInputHashPos + i] != pProverRequest->blobOuterProofInputBlobInner["publics"][blobInnerPublics.finalAccBatchHashDataPos + i])
            {
                zklog.error("Prover::genBlobOuterProof() newBatchAccInputHashPos (batch) and finalAccBatchHashDataPos (blob inner) are not consistent" + pProverRequest->blobOuterProofInputBatch["publics"][batchPublics.newBatchAccInputHashPos + i].dump() + "!=" + pProverRequest->blobOuterProofInputBlobInner["publics"][blobInnerPublics.finalAccBatchHashDataPos + i].dump());
                pProverRequest->result = ZKR_BLOB_OUTER_PROOF_INVALID_INPUT;
                return;
            }
        }

        // If L1InfoTreeIndex is correct, check that the L1InfoTreeRoot matches
        bool isInvalidL1InfoTreeIndex = pProverRequest->blobOuterProofInputBatch["publics"][batchPublics.currentL1InfoTreeIndexPos] != pProverRequest->blobOuterProofInputBlobInner["publics"][blobInnerPublics.lastL1InfoTreeIndexPos];

        if (!isInvalidL1InfoTreeIndex) {
            for (int i = 0; i < 8; i++)
            {
                if (pProverRequest->blobOuterProofInputBatch["publics"][batchPublics.currentL1InfoTreeRootPos + i] != pProverRequest->blobOuterProofInputBlobInner["publics"][blobInnerPublics.lastL1InfoTreeRootPos + i])
                {
                    zklog.error("Prover::genBlobOuterProof() currentL1InfoTreeRootPos (batch) and lastL1InfoTreeRootPos (blob inner) are not consistent" + pProverRequest->blobOuterProofInputBatch["publics"][batchPublics.currentL1InfoTreeRootPos + i].dump() + "!=" + pProverRequest->blobOuterProofInputBlobInner["publics"][blobInnerPublics.lastL1InfoTreeRootPos + i].dump());
                    pProverRequest->result = ZKR_BLOB_OUTER_PROOF_INVALID_INPUT;
                    return;
                }
            }
        }
    }

    // ----------------------------------------------
    // JOIN PUBLICS AND GET COMMITED POLS
    // ----------------------------------------------
    ordered_json blobOuterRecursive2VerkeyJson;
    file2json(config.blobOuterRecursive2Verkey, blobOuterRecursive2VerkeyJson);

    json zkinInputBlobOuter = joinzkinBlobOuter(pProverRequest->blobOuterProofInputBatch, pProverRequest->blobOuterProofInputBlobInner, blobOuterRecursive2VerkeyJson, pProverRequest->blobOuterProofInputBatch["chain_id"].dump(), starkBlobOuter->starkInfo.starkStruct.steps.size());    

    ordered_json blobOuterVerkeyJson;
    file2json(config.blobOuterVerkey, blobOuterVerkeyJson);

    Goldilocks::Element blobOuterVerkey[4];
    blobOuterVerkey[0] = Goldilocks::fromU64(blobOuterVerkeyJson["constRoot"][0]);
    blobOuterVerkey[1] = Goldilocks::fromU64(blobOuterVerkeyJson["constRoot"][1]);
    blobOuterVerkey[2] = Goldilocks::fromU64(blobOuterVerkeyJson["constRoot"][2]);
    blobOuterVerkey[3] = Goldilocks::fromU64(blobOuterVerkeyJson["constRoot"][3]);

    Goldilocks::Element publics[starkBlobOuter->starkInfo.nPublics];
    for (uint64_t i = 0; i < starkBlobOuter->starkInfo.nPublics; i++)
    {
        publics[i] = Goldilocks::fromString(zkinInputBlobOuter["publics"][i]);
    }

    CommitPolsStarks cmPolsBlobOuter(pAddress, (1 << starkBlobOuter->starkInfo.starkStruct.nBits), starkBlobOuter->starkInfo.nCm1);
    CircomBlobOuter::getCommitedPols(&cmPolsBlobOuter, config.blobOuterCircuit, config.blobOuterExec, zkinInputBlobOuter, (1 << starkBlobOuter->starkInfo.starkStruct.nBits), starkBlobOuter->starkInfo.nCm1);

    // void *pointerCmBlobOuterPols = mapFile("config/blob_outer/blob_outer.commit", cmPolsBlobOuter.size(), true);
    // memcpy(pointerCmBlobOuterPols, cmPolsBlobOuter.address(), cmPolsBlobOuter.size());
    // unmapFile(pointerCmBlobOuterPols, cmPolsBlobOuter.size());

    //-------------------------------------------
    // Generate Blob Outer  proof
    //-------------------------------------------

    TimerStart(STARK_BLOB_OUTER_PROOF);
    Recursive2Steps blobOuterChelpersSteps;
    uint64_t polBitsBlobOuter = starkBlobOuter->getPolBits();
    FRIProof fproofBlobOuter((1 << polBitsBlobOuter), FIELD_EXTENSION, starkBlobOuter->starkInfo.starkStruct.steps.size(), starkBlobOuter->starkInfo.evMap.size(), starkBlobOuter->starkInfo.nPublics);
    if(USE_GENERIC_PARSER) {
        CHelpersSteps cHelpersSteps;
        starkBlobOuter->genProof(fproofBlobOuter, publics, blobOuterVerkey, &cHelpersSteps);
    } else {
        starkBlobOuter->genProof(fproofBlobOuter, publics, blobOuterVerkey, &blobOuterChelpersSteps);
    }
    
    TimerStopAndLog(STARK_BLOB_OUTER_PROOF);

    // Save the proof & zkinproof
    nlohmann::ordered_json jProofBlobOuter = fproofBlobOuter.proofs.proof2json();
    nlohmann::ordered_json zkinBlobOuter = proof2zkinStark(jProofBlobOuter);
    zkinBlobOuter["publics"] = zkinInputBlobOuter["publics"];

    // Output is pProverRequest->blobOuterProofOutput (of type json)
    pProverRequest->blobOuterProofOutput = zkinBlobOuter;

    // Save output to file
    if (config.saveOutputToFile)
    {
        json2file(pProverRequest->blobOuterProofOutput, pProverRequest->filePrefix + "blob_outer_proof.output.json");
    }
    // Save proof to file
    if (config.saveProofToFile)
    {
        jProofBlobOuter["publics"] = zkinInputBlobOuter["publics"];
        json2file(jProofBlobOuter, pProverRequest->filePrefix + "blob_outer_proof.proof.json");
    }


    json2file(zkinInputBlobOuter["publics"], pProverRequest->publicsOutputFile());
    
    pProverRequest->result = ZKR_SUCCESS;

    TimerStopAndLog(PROVER_BLOB_OUTER_PROOF);
};

void Prover::genAggregatedBatchProof (ProverRequest *pProverRequest)
{

    zkassert(config.generateProof());
    zkassert(pProverRequest != NULL);
    zkassert(pProverRequest->type == prt_genAggregatedBatchProof);

    TimerStart(PROVER_AGGREGATED_BATCH_PROOF);

    printMemoryInfo(true);
    printProcessInfo(true);

    // Save input to file
    if (config.saveInputToFile)
    {
        json2file(pProverRequest->aggregatedBatchProofInput1, pProverRequest->filePrefix + "aggregated_batch_proof.input_1.json");
        json2file(pProverRequest->aggregatedBatchProofInput2, pProverRequest->filePrefix + "aggregated_batch_proof.input_2.json");
    }

    // ----------------------------------------------
    // CHECKS
    // ----------------------------------------------
    
    BatchPublics batchPublics;

    // Check chainID
    if (pProverRequest->aggregatedBatchProofInput1["publics"][batchPublics.chainIdPos] != pProverRequest->aggregatedBatchProofInput2["publics"][batchPublics.chainIdPos])
    {
        zklog.error("Prover::genAggregatedBatchProof() Inputs has different chainId " + pProverRequest->aggregatedBatchProofInput1["publics"][batchPublics.chainIdPos].dump() + "!=" + pProverRequest->aggregatedBatchProofInput2["publics"][batchPublics.chainIdPos].dump());
        pProverRequest->result = ZKR_AGGREGATED_PROOF_INVALID_INPUT;
        return;
    }
    // Check ForkID
    if (pProverRequest->aggregatedBatchProofInput1["publics"][batchPublics.forkIdPos] != pProverRequest->aggregatedBatchProofInput2["publics"][batchPublics.forkIdPos])
    {
        zklog.error("Prover::genAggregatedBatchProof() Inputs has different forkId " + pProverRequest->aggregatedBatchProofInput1["publics"][batchPublics.forkIdPos].dump() + "!=" + pProverRequest->aggregatedBatchProofInput2["publics"][batchPublics.forkIdPos].dump());
        pProverRequest->result = ZKR_AGGREGATED_PROOF_INVALID_INPUT;
        return;
    }
    // Check midStateRoot
    for (int i = 0; i < 8; i++)
    {
        if (pProverRequest->aggregatedBatchProofInput1["publics"][batchPublics.newStateRootPos + i] != pProverRequest->aggregatedBatchProofInput2["publics"][batchPublics.oldStateRootPos + i])
        {
            zklog.error("Prover::genAggregatedBatchProof() The newStateRoot and the oldStateRoot are not consistent " + pProverRequest->aggregatedBatchProofInput1["publics"][batchPublics.newStateRootPos + i].dump() + "!=" + pProverRequest->aggregatedBatchProofInput2["publics"][batchPublics.oldStateRootPos + i].dump());
            pProverRequest->result = ZKR_AGGREGATED_PROOF_INVALID_INPUT;
            return;
        }
    }
    // Check midBatchAccInputHash0
    for (int i = 0; i < 8; i++)
    {
        if (pProverRequest->aggregatedBatchProofInput1["publics"][batchPublics.newBatchAccInputHashPos + i] != pProverRequest->aggregatedBatchProofInput2["publics"][batchPublics.oldBatchAccInputHashPos + i])
        {
            zklog.error("Prover::genAggregatedBatchProof() newAccInputHash and oldAccInputHash are not consistent" + pProverRequest->aggregatedBatchProofInput1["publics"][batchPublics.newBatchAccInputHashPos + i].dump() + "!=" + pProverRequest->aggregatedBatchProofInput2["publics"][batchPublics.oldBatchAccInputHashPos + i].dump());
            pProverRequest->result = ZKR_AGGREGATED_PROOF_INVALID_INPUT;
            return;
        }
    }
    // Check midL1InfoTreeRoot
    for (int i = 0; i < 8; i++)
    {
        if (pProverRequest->aggregatedBatchProofInput1["publics"][batchPublics.currentL1InfoTreeRootPos + i] != pProverRequest->aggregatedBatchProofInput2["publics"][batchPublics.previousL1InfoTreeRootPos + i])
        {
            zklog.error("Prover::genAggregatedBatchProof() previousL1InfoTreeRoot and currentL1InfoTreeRoot are not consistent" + pProverRequest->aggregatedBatchProofInput1["publics"][batchPublics.currentL1InfoTreeRootPos + i].dump() + "!=" + pProverRequest->aggregatedBatchProofInput2["publics"][batchPublics.previousL1InfoTreeRootPos + i].dump());
            pProverRequest->result = ZKR_AGGREGATED_PROOF_INVALID_INPUT;
            return;
        }
    }
    // Check midL1InfoTreeIndex
    if (pProverRequest->aggregatedBatchProofInput1["publics"][batchPublics.currentL1InfoTreeIndexPos] != pProverRequest->aggregatedBatchProofInput2["publics"][batchPublics.previousL1InfoTreeIndexPos])
    {
        zklog.error("Prover::genAggregatedBatchProof() previousL1InfoTreeIndex and currentL1InfoTreeIndex are not consistent" + pProverRequest->aggregatedBatchProofInput1["publics"][batchPublics.currentL1InfoTreeIndexPos].dump() + "!=" + pProverRequest->aggregatedBatchProofInput2["publics"][batchPublics.previousL1InfoTreeIndexPos].dump());
        pProverRequest->result = ZKR_AGGREGATED_PROOF_INVALID_INPUT;
        return;
    }

    // ----------------------------------------------
    // JOIN PUBLICS AND GET COMMITED POLS
    // ----------------------------------------------
    ordered_json recursive2VerkeyJson;
    file2json(config.recursive2Verkey, recursive2VerkeyJson);

    json zkinInputRecursive2 = joinzkinBatchRecursive2(pProverRequest->aggregatedBatchProofInput1, pProverRequest->aggregatedBatchProofInput2, recursive2VerkeyJson, starkBatchRecursive2->starkInfo.starkStruct.steps.size());

    Goldilocks::Element recursive2Verkey[4];
    recursive2Verkey[0] = Goldilocks::fromU64(recursive2VerkeyJson["constRoot"][0]);
    recursive2Verkey[1] = Goldilocks::fromU64(recursive2VerkeyJson["constRoot"][1]);
    recursive2Verkey[2] = Goldilocks::fromU64(recursive2VerkeyJson["constRoot"][2]);
    recursive2Verkey[3] = Goldilocks::fromU64(recursive2VerkeyJson["constRoot"][3]);

    Goldilocks::Element publics[starkBatchRecursive2->starkInfo.nPublics];

    for (uint64_t i = 0; i < starkBatch->starkInfo.nPublics; i++)
    {
        publics[i] = Goldilocks::fromString(zkinInputRecursive2["publics"][i]);
    }

    for (uint64_t i = 0; i < recursive2VerkeyJson["constRoot"].size(); i++)
    {
        publics[starkBatch->starkInfo.nPublics + i] = Goldilocks::fromU64(recursive2VerkeyJson["constRoot"][i]);
    }

    CommitPolsStarks cmPolsRecursive2(pAddress, (1 << starkBatchRecursive2->starkInfo.starkStruct.nBits), starkBatchRecursive2->starkInfo.nCm1);
    CircomRecursive2::getCommitedPols(&cmPolsRecursive2, config.recursive2Circuit, config.recursive2Exec, zkinInputRecursive2, (1 << starkBatchRecursive2->starkInfo.starkStruct.nBits), starkBatchRecursive2->starkInfo.nCm1);

    // void *pointerCmRecursive2Pols = mapFile("config/recursive2/recursive2.commit", cmPolsRecursive2.size(), true);
    // memcpy(pointerCmRecursive2Pols, cmPolsRecursive2.address(), cmPolsRecursive2.size());
    // unmapFile(pointerCmRecursive2Pols, cmPolsRecursive2.size());

    //-------------------------------------------
    // Generate Recursive 2 proof
    //-------------------------------------------

    TimerStart(STARK_RECURSIVE_2_PROOF_BATCH_PROOF);
    Recursive2Steps recursive2ChelpersSteps;
    uint64_t polBitsRecursive2 = starkBatchRecursive2->getPolBits();
    FRIProof fproofRecursive2((1 << polBitsRecursive2), FIELD_EXTENSION, starkBatchRecursive2->starkInfo.starkStruct.steps.size(), starkBatchRecursive2->starkInfo.evMap.size(), starkBatchRecursive2->starkInfo.nPublics);
    if(USE_GENERIC_PARSER) {
        CHelpersSteps cHelpersSteps;
        starkBatchRecursive2->genProof(fproofRecursive2, publics, recursive2Verkey, &cHelpersSteps);
    } else {
        starkBatchRecursive2->genProof(fproofRecursive2, publics, recursive2Verkey, &recursive2ChelpersSteps);
    }
    
    TimerStopAndLog(STARK_RECURSIVE_2_PROOF_BATCH_PROOF);

    // Save the proof & zkinproof
    nlohmann::ordered_json jProofRecursive2 = fproofRecursive2.proofs.proof2json();
    nlohmann::ordered_json zkinRecursive2 = proof2zkinStark(jProofRecursive2);
    zkinRecursive2["isAggregatedCircuit"] = "1";
    zkinRecursive2["publics"] = zkinInputRecursive2["publics"];

    // Output is pProverRequest->aggregatedBatchProofOutput (of type json)
    pProverRequest->aggregatedBatchProofOutput = zkinRecursive2;

    // Save output to file
    if (config.saveOutputToFile)
    {
        json2file(pProverRequest->aggregatedBatchProofOutput, pProverRequest->filePrefix + "aggregated_batch_proof.output.json");
    }
    // Save proof to file
    if (config.saveProofToFile)
    {
        jProofRecursive2["publics"] = zkinInputRecursive2["publics"];
        json2file(jProofRecursive2, pProverRequest->filePrefix + "aggregated_batch_proof.proof.json");
    }

    // Add the recursive2 verification key
    json publicsJson = json::array();
    for (uint64_t i = 0; i < starkBatch->starkInfo.nPublics; i++)
    {
        publicsJson[i] = zkinInputRecursive2["publics"][i];
    }
    // Add the recursive2 verification key
    publicsJson[starkBatch->starkInfo.nPublics] = to_string(recursive2VerkeyJson["constRoot"][0]);
    publicsJson[starkBatch->starkInfo.nPublics + 1] = to_string(recursive2VerkeyJson["constRoot"][1]);
    publicsJson[starkBatch->starkInfo.nPublics + 2] = to_string(recursive2VerkeyJson["constRoot"][2]);
    publicsJson[starkBatch->starkInfo.nPublics + 3] = to_string(recursive2VerkeyJson["constRoot"][3]);

    json2file(publicsJson, pProverRequest->publicsOutputFile());

    pProverRequest->result = ZKR_SUCCESS;

    TimerStopAndLog(PROVER_AGGREGATED_BATCH_PROOF);
}

void Prover::genAggregatedBlobOuterProof (ProverRequest *pProverRequest){
    
    zkassert(config.generateProof());
    zkassert(pProverRequest != NULL);
    zkassert(pProverRequest->type == prt_genAggregatedBlobOuterProof);

    TimerStart(PROVER_AGGREGATED_BLOB_OUTER_PROOF);

    printMemoryInfo(true);
    printProcessInfo(true);

    // Save input to file
    if (config.saveInputToFile)
    {
        json2file(pProverRequest->aggregatedBlobOuterProofInput1, pProverRequest->filePrefix + "aggregated_blob_outer_proof.input_1.json");
        json2file(pProverRequest->aggregatedBlobOuterProofInput2, pProverRequest->filePrefix + "aggregated_blob_outer_proof.input_2.json");
    }

    // ----------------------------------------------
    // CHECKS
    // ----------------------------------------------
    
    BlobOuterPublics blobOuterPublics;

    // Check chainID
    if (pProverRequest->aggregatedBlobOuterProofInput1["publics"][blobOuterPublics.chainIdPos] != pProverRequest->aggregatedBlobOuterProofInput2["publics"][blobOuterPublics.chainIdPos])
    {
        zklog.error("Prover::genAggregatedBlobOuterProof() Inputs has different chainId " + pProverRequest->aggregatedBlobOuterProofInput1["publics"][blobOuterPublics.chainIdPos].dump() + "!=" + pProverRequest->aggregatedBlobOuterProofInput2["publics"][blobOuterPublics.chainIdPos].dump());
        pProverRequest->result = ZKR_AGGREGATED_PROOF_INVALID_INPUT;
        return;
    }
    // Check ForkID
    if (pProverRequest->aggregatedBlobOuterProofInput1["publics"][blobOuterPublics.forkIdPos] != pProverRequest->aggregatedBlobOuterProofInput2["publics"][blobOuterPublics.forkIdPos])
    {
        zklog.error("Prover::genAggregatedBlobOuterProof() Inputs has different forkId " + pProverRequest->aggregatedBlobOuterProofInput1["publics"][blobOuterPublics.forkIdPos].dump() + "!=" + pProverRequest->aggregatedBlobOuterProofInput2["publics"][blobOuterPublics.forkIdPos].dump());
        pProverRequest->result = ZKR_AGGREGATED_PROOF_INVALID_INPUT;
        return;
    }
    // Check midStateRoot
    for (int i = 0; i < 8; i++)
    {
        if (pProverRequest->aggregatedBlobOuterProofInput1["publics"][blobOuterPublics.newStateRootPos + i] != pProverRequest->aggregatedBlobOuterProofInput2["publics"][blobOuterPublics.oldStateRootPos + i])
        {
            zklog.error("Prover::genAggregatedBlobOuterProof() The newStateRoot and the oldStateRoot are not consistent " + pProverRequest->aggregatedBlobOuterProofInput1["publics"][blobOuterPublics.newStateRootPos + i].dump() + "!=" + pProverRequest->aggregatedBlobOuterProofInput2["publics"][blobOuterPublics.oldStateRootPos + i].dump());
            pProverRequest->result = ZKR_AGGREGATED_PROOF_INVALID_INPUT;
            return;
        }
       
    }
    // Check midBlobStateRoot
    for (int i = 0; i < 8; i++)
    {
        if (pProverRequest->aggregatedBlobOuterProofInput1["publics"][blobOuterPublics.newBlobStateRootPos + i] != pProverRequest->aggregatedBlobOuterProofInput2["publics"][blobOuterPublics.oldBlobStateRootPos + i])
        {
            zklog.error("Prover::genAggregatedBlobOuterProof() newBlobStateRoot and oldBlobStateRoot are not consistent" + pProverRequest->aggregatedBlobOuterProofInput1["publics"][blobOuterPublics.newBlobStateRootPos + i].dump() + "!=" + pProverRequest->aggregatedBlobOuterProofInput2["publics"][blobOuterPublics.oldBlobStateRootPos + i].dump());
            pProverRequest->result = ZKR_AGGREGATED_PROOF_INVALID_INPUT;
            return;
        }
    }
    // Check midBlobAccInputHash
    for (int i = 0; i < 8; i++)
    {
        if (pProverRequest->aggregatedBlobOuterProofInput1["publics"][blobOuterPublics.newBlobAccInputHashPos + i] != pProverRequest->aggregatedBlobOuterProofInput2["publics"][blobOuterPublics.oldBlobAccInputHashPos + i])
        {
            zklog.error("Prover::genAggregatedBlobOuterProof() newBlobAccInputHash and oldBlobAccInputHash are not consistent" + pProverRequest->aggregatedBlobOuterProofInput1["publics"][blobOuterPublics.newBlobAccInputHashPos + i].dump() + "!=" + pProverRequest->aggregatedBlobOuterProofInput2["publics"][blobOuterPublics.oldBlobAccInputHashPos + i].dump());
            pProverRequest->result = ZKR_AGGREGATED_PROOF_INVALID_INPUT;
            return;
        }
    }
    // Check midNumBlob
    if (pProverRequest->aggregatedBlobOuterProofInput1["publics"][blobOuterPublics.newBlobNumPos] != pProverRequest->aggregatedBlobOuterProofInput2["publics"][blobOuterPublics.oldBlobNumPos])
    {
        zklog.error("Prover::genAggregatedBlobOuterProof() newNumBlob and oldNumBlob are not consistent" + pProverRequest->aggregatedBlobOuterProofInput1["publics"][blobOuterPublics.newBlobNumPos].dump() + "!=" + pProverRequest->aggregatedBlobOuterProofInput2["publics"][blobOuterPublics.oldBlobNumPos].dump());
        pProverRequest->result = ZKR_AGGREGATED_PROOF_INVALID_INPUT;
        return;
    }

    // ----------------------------------------------
    // JOIN PUBLICS AND GET COMMITED POLS
    // ----------------------------------------------
    ordered_json blobOuterRecursive2VerkeyJson;
    file2json(config.blobOuterRecursive2Verkey, blobOuterRecursive2VerkeyJson);

    json zkinInputRecursive2 = joinzkinBlobOuterRecursive2(pProverRequest->aggregatedBlobOuterProofInput1, pProverRequest->aggregatedBlobOuterProofInput2, blobOuterRecursive2VerkeyJson, starkBlobOuterRecursive2->starkInfo.starkStruct.steps.size());

    Goldilocks::Element recursive2Verkey[4];
    recursive2Verkey[0] = Goldilocks::fromU64(blobOuterRecursive2VerkeyJson["constRoot"][0]);
    recursive2Verkey[1] = Goldilocks::fromU64(blobOuterRecursive2VerkeyJson["constRoot"][1]);
    recursive2Verkey[2] = Goldilocks::fromU64(blobOuterRecursive2VerkeyJson["constRoot"][2]);
    recursive2Verkey[3] = Goldilocks::fromU64(blobOuterRecursive2VerkeyJson["constRoot"][3]);

    Goldilocks::Element publics[starkBlobOuterRecursive2->starkInfo.nPublics];

    for (uint64_t i = 0; i < starkBlobOuter->starkInfo.nPublics; i++)
    {
        publics[i] = Goldilocks::fromString(zkinInputRecursive2["publics"][i]);
    }

    for (uint64_t i = 0; i < blobOuterRecursive2VerkeyJson["constRoot"].size(); i++)
    {
        publics[starkBlobOuter->starkInfo.nPublics + i] = Goldilocks::fromU64(blobOuterRecursive2VerkeyJson["constRoot"][i]);
    }

    CommitPolsStarks cmPolsRecursive2(pAddress, (1 << starkBlobOuterRecursive2->starkInfo.starkStruct.nBits), starkBlobOuterRecursive2->starkInfo.nCm1);
    CircomBlobOuterRecursive2::getCommitedPols(&cmPolsRecursive2, config.blobOuterRecursive2Verkey, config.blobOuterRecursive2Exec, zkinInputRecursive2, (1 << starkBlobOuterRecursive2->starkInfo.starkStruct.nBits), starkBlobOuterRecursive2->starkInfo.nCm1);

    // void *pointerCmRecursive2Pols = mapFile("config/blob_outer_recursive2/blob_outer_recursive2.commit", cmPolsRecursive2.size(), true);
    // memcpy(pointerCmRecursive2Pols, cmPolsRecursive2.address(), cmPolsRecursive2.size());
    // unmapFile(pointerCmRecursive2Pols, cmPolsRecursive2.size());

    //-------------------------------------------
    // Generate Recursive 2 proof
    //-------------------------------------------

    TimerStart(STARK_RECURSIVE_2_PROOF_BLOB_OUTER_PROOF);
    Recursive2Steps blobOuterRecursive2ChelpersSteps;
    uint64_t polBitsRecursive2 = starkBlobOuterRecursive2->getPolBits();
    FRIProof fproofRecursive2((1 << polBitsRecursive2), FIELD_EXTENSION, starkBlobOuterRecursive2->starkInfo.starkStruct.steps.size(), starkBlobOuterRecursive2->starkInfo.evMap.size(), starkBlobOuterRecursive2->starkInfo.nPublics);
    if(USE_GENERIC_PARSER) {
        CHelpersSteps cHelpersSteps;
        starkBlobOuterRecursive2->genProof(fproofRecursive2, publics, recursive2Verkey, &cHelpersSteps);
    } else {
        starkBlobOuterRecursive2->genProof(fproofRecursive2, publics, recursive2Verkey, &blobOuterRecursive2ChelpersSteps);
    }
    
    TimerStopAndLog(STARK_RECURSIVE_2_PROOF_BLOB_OUTER_PROOF);

    // Save the proof & zkinproof
    nlohmann::ordered_json jProofRecursive2 = fproofRecursive2.proofs.proof2json();
    nlohmann::ordered_json zkinRecursive2 = proof2zkinStark(jProofRecursive2);
    zkinRecursive2["publics"] = zkinInputRecursive2["publics"];

    // Output is pProverRequest->aggregatedBlobOuterProofOutput (of type json)
    pProverRequest->aggregatedBlobOuterProofOutput = zkinRecursive2;

    // Save output to file
    if (config.saveOutputToFile)
    {
        json2file(pProverRequest->aggregatedBlobOuterProofOutput, pProverRequest->filePrefix + "aggregated_blob_outer_proof.output.json");
    }
    // Save proof to file
    if (config.saveProofToFile)
    {
        jProofRecursive2["publics"] = zkinInputRecursive2["publics"];
        json2file(jProofRecursive2, pProverRequest->filePrefix + "aggregated_blob_outer_proof.proof.json");
    }

    // Add the recursive2 verification key
    json publicsJson = json::array();
    for (uint64_t i = 0; i < starkBlobOuter->starkInfo.nPublics; i++)
    {
        publicsJson[i] = zkinInputRecursive2["publics"][i];
    }
    // Add the recursive2 verification key
    publicsJson[starkBlobOuter->starkInfo.nPublics ] = to_string(blobOuterRecursive2VerkeyJson["constRoot"][0]);
    publicsJson[starkBlobOuter->starkInfo.nPublics + 1] = to_string(blobOuterRecursive2VerkeyJson["constRoot"][1]);
    publicsJson[starkBlobOuter->starkInfo.nPublics + 2] = to_string(blobOuterRecursive2VerkeyJson["constRoot"][2]);
    publicsJson[starkBlobOuter->starkInfo.nPublics + 3] = to_string(blobOuterRecursive2VerkeyJson["constRoot"][3]);

    json2file(publicsJson, pProverRequest->publicsOutputFile());

    pProverRequest->result = ZKR_SUCCESS;

    TimerStopAndLog(PROVER_AGGREGATED_BLOB_OUTER_PROOF);
};

void Prover::genFinalProof (ProverRequest *pProverRequest)
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
    CircomRecursiveF::getCommitedPols(&cmPolsRecursiveF, config.recursivefCircuit, config.recursivefExec, zkinFinal, (1 << starksRecursiveF->starkInfo.starkStruct.nBits), starksRecursiveF->starkInfo.nCm1);

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
        CHelpersSteps cHelpersSteps;
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
    //  Final proof
    //  ----------------------------------------------

    TimerStart(CIRCOM_LOAD_CIRCUIT_FINAL);
    CircomFinal::Circom_Circuit *circuitFinal = CircomFinal::loadCircuit(config.finalCircuit);
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

void Prover::executeBatch (ProverRequest *pProverRequest)
{
    zkassert(!config.generateProof());
    zkassert(pProverRequest != NULL);

    TimerStart(PROVER_EXECUTE_BATCH);

    printMemoryInfo(true);
    printProcessInfo(true);

    zklog.info("Prover::executeBatch() timestamp: " + pProverRequest->timestamp);
    zklog.info("Prover::executeBatch() UUID: " + pProverRequest->uuid);
    zklog.info("Prover::executeBatch() input file: " + pProverRequest->inputFile());

    // In proof-generation executions we can only process the exact number of steps
    if (pProverRequest->input.stepsN > 0)
    {
        zklog.error("Prover::executeBatch() called with input.stepsN=" + to_string(pProverRequest->input.stepsN));
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
    uint64_t commitPolsSize = PROVER_FORK_NAMESPACE::CommitPols::pilSize();
    void *pExecuteAddress = NULL;

    if (config.zkevmCmPols.size() > 0)
    {
        pExecuteAddress = mapFile(config.zkevmCmPols, commitPolsSize, true);
        zklog.info("Prover::executeBatch() successfully mapped " + to_string(commitPolsSize) + " bytes to file " + config.zkevmCmPols);
    }
    else
    {
        pExecuteAddress = calloc(commitPolsSize, 1);
        if (pExecuteAddress == NULL)
        {
            zklog.error("Prover::executeBatch() failed calling calloc() of size " + to_string(commitPolsSize));
            exitProcess();
        }
        zklog.info("Prover::executeBatch() successfully allocated " + to_string(commitPolsSize) + " bytes");
    }

    /************/
    /* Executor */
    /************/

    PROVER_FORK_NAMESPACE::CommitPols cmPols(pExecuteAddress, PROVER_FORK_NAMESPACE::CommitPols::pilDegree());

    // Execute all the State Machines
    TimerStart(EXECUTOR_EXECUTE_EXECUTE_BATCH);
    executor.executeBatch(*pProverRequest, cmPols);
    TimerStopAndLog(EXECUTOR_EXECUTE_EXECUTE_BATCH);

    logBatchExecutionInfo(cmPols, pProverRequest);

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
        free(pExecuteAddress);
    }

    TimerStopAndLog(PROVER_EXECUTE_BATCH);
}

void Prover::executeBlobInner (ProverRequest *pProverRequest)
{
    zkassert(!config.generateProof());
    zkassert(pProverRequest != NULL);

    TimerStart(PROVER_EXECUTE_BLOB_INNER);

    printMemoryInfo(true);
    printProcessInfo(true);

    zkassert(pProverRequest != NULL);

    zklog.info("Prover::executeBlobInner() timestamp: " + pProverRequest->timestamp);
    zklog.info("Prover::executeBlobInner() UUID: " + pProverRequest->uuid);
    zklog.info("Prover::executeBlobInner() input file: " + pProverRequest->inputFile());

    // In proof-generation executions we can only process the exact number of steps
    if (pProverRequest->input.stepsN > 0)
    {
        zklog.error("Prover::executeBlobInner() called with input.stepsN=" + to_string(pProverRequest->input.stepsN));
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
    uint64_t blobInnerCmPolsSize = PROVER_FORK_NAMESPACE::CommitPols::pilSize();
    void *pExecuteAddress = NULL;

    if (config.blobInnerCmPols.size() > 0)
    {
        pExecuteAddress = mapFile(config.blobInnerCmPols, blobInnerCmPolsSize, true);
        zklog.info("Prover::executeBlobInner() successfully mapped " + to_string(blobInnerCmPolsSize) + " bytes to file " + config.blobInnerCmPols);
    }
    else
    {
        pExecuteAddress = calloc(blobInnerCmPolsSize, 1);
        if (pExecuteAddress == NULL)
        {
            zklog.error("Prover::executeBlobInner() failed calling calloc() of size " + to_string(blobInnerCmPolsSize));
            exitProcess();
        }
        zklog.info("Prover::executeBlobInner() successfully allocated " + to_string(blobInnerCmPolsSize) + " bytes");
    }

    /************/
    /* Executor */
    /************/

    PROVER_FORK_NAMESPACE::CommitPols cmPols(pExecuteAddress, PROVER_FORK_NAMESPACE::CommitPols::pilDegree());

    // Execute all the State Machines
    TimerStart(EXECUTOR_EXECUTE_EXECUTE_BLOB_INNER);
    executor.executeBlobInner(*pProverRequest, cmPols);
    TimerStopAndLog(EXECUTOR_EXECUTE_EXECUTE_BLOB_INNER);

    logBlobInnerExecutionInfo(cmPols, pProverRequest);

    /***************/
    /* Free memory */
    /***************/

    // Unmap committed polynomials address
    if (config.blobInnerCmPols.size() > 0)
    {
        unmapFile(pExecuteAddress, blobInnerCmPolsSize);
    }
    else
    {
        free(pExecuteAddress);
    }

    TimerStopAndLog(PROVER_EXECUTE_BLOB_INNER);
}

void Prover::logBatchExecutionInfo(PROVER_FORK_NAMESPACE::CommitPols &cmPols, ProverRequest *pProverRequest)
{
    zkassert(pProverRequest != NULL);
    uint64_t lastN = cmPols.pilDegree() - 1;

    // log old and new StateRoot
    zklog.info("Prover::genBatchProof() called executor.executeBatch() oldStateRoot=" + pProverRequest->input.publicInputsExtended.publicInputs.oldStateRoot.get_str(16) +
               " newStateRoot=" + pProverRequest->pFullTracer->get_new_state_root() +
               " pols.SR[0]=" + fea2string(fr, cmPols.Main.SR0[0], cmPols.Main.SR1[0], cmPols.Main.SR2[0], cmPols.Main.SR3[0], cmPols.Main.SR4[0], cmPols.Main.SR5[0], cmPols.Main.SR6[0], cmPols.Main.SR7[0]) +
               " pols.SR[lastN]=" + fea2string(fr, cmPols.Main.SR0[lastN], cmPols.Main.SR1[lastN], cmPols.Main.SR2[lastN], cmPols.Main.SR3[lastN], cmPols.Main.SR4[lastN], cmPols.Main.SR5[lastN], cmPols.Main.SR6[lastN], cmPols.Main.SR7[lastN]) +
               " lastN=" + to_string(lastN));

    // log old and new BatchAccInputHash
    zklog.info("Prover::genBatchProof() called executor.executeBatch() oldBatchAccInputHash=" + pProverRequest->input.publicInputsExtended.publicInputs.oldAccInputHash.get_str(16) +
               " newAccInputHash=" + pProverRequest->pFullTracer->get_new_acc_input_hash() +
               " pols.C[0]=" + fea2string(fr, cmPols.Main.C0[0], cmPols.Main.C1[0], cmPols.Main.C2[0], cmPols.Main.C3[0], cmPols.Main.C4[0], cmPols.Main.C5[0], cmPols.Main.C6[0], cmPols.Main.C7[0]) +
               " pols.C[lastN]=" + fea2string(fr, cmPols.Main.C0[lastN], cmPols.Main.C1[lastN], cmPols.Main.C2[lastN], cmPols.Main.C3[lastN], cmPols.Main.C4[lastN], cmPols.Main.C5[lastN], cmPols.Main.C6[lastN], cmPols.Main.C7[lastN]) +
               " lastN=" + to_string(lastN));

    // log previous and current L1InfoTreeRoot
    // note: missed function get_current_l1_info_tree_root()
    zklog.info("Prover::genBatchProof() called executor.executeBatch() previousL1InfoTreeRoot=" + pProverRequest->input.publicInputsExtended.publicInputs.previousL1InfoTreeRoot.get_str(16) +
               " currentL1InfoTreeRoot=" /*pProverRequest->pFullTracer->get_current_l1_info_tree_root()*/ +
               " pols.D[0]=" + fea2string(fr, cmPols.Main.D0[0], cmPols.Main.D1[0], cmPols.Main.D2[0], cmPols.Main.D3[0], cmPols.Main.D4[0], cmPols.Main.D5[0], cmPols.Main.D6[0], cmPols.Main.D7[0]) +
               " pols.D[lastN]=" + fea2string(fr, cmPols.Main.D0[lastN], cmPols.Main.D1[lastN], cmPols.Main.D2[lastN], cmPols.Main.D3[lastN], cmPols.Main.D4[lastN], cmPols.Main.D5[lastN], cmPols.Main.D6[lastN], cmPols.Main.D7[lastN]) +
               " lastN=" + to_string(lastN));
    // log previous and current L1InfoTreeIndex
    // note: missing function get_current_l1_info_tree_index()
    zklog.info("Prover::genBatchProof() called executor.executeBatch() previousL1InfoTreeIndex=" + to_string(pProverRequest->input.publicInputsExtended.publicInputs.previousL1InfoTreeIndex) +
               " currentL1InfoTreeIndex=" /*+ to_string(pProverRequest->pFullTracer->get_current_l1_info_tree_index()*/ +
               " pols.RCX[0]=" + Goldilocks::toString(cmPols.Main.RCX[0]) +
               " pols.RCX[lastN]=" + Goldilocks::toString(cmPols.Main.RCX[lastN]));
}

void Prover::logBlobInnerExecutionInfo(PROVER_BLOB_FORK_NAMESPACE::CommitPols &cmPols, ProverRequest *pProverRequest)
{
    uint64_t lastN = cmPols.pilDegree() - 1;
    // log old and new BlobStateRoot
    // note: we need /*pProverRequest->pFullTracer->get_blob_new_state_root()*/
    zklog.info("Prover::genBlobInnerProof() called executor.executeBlobInner() oldBlobStateRoot=" + pProverRequest->input.publicInputsExtended.publicInputs.oldBlobStateRoot.get_str(16) +
               " newBlobStateRoot=" /*+ pProverRequest->pFullTracer->get_blob_new_state_root()*/ +
               " pols.B[0]=" + fea2string(fr, cmPols.Main.B0[0], cmPols.Main.B1[0], cmPols.Main.B2[0], cmPols.Main.B3[0], cmPols.Main.B4[0], cmPols.Main.B5[0], cmPols.Main.B6[0], cmPols.Main.B7[0]) +
               " pols.B[lastN]=" + fea2string(fr, cmPols.Main.B0[lastN], cmPols.Main.B1[lastN], cmPols.Main.B2[lastN], cmPols.Main.B3[lastN], cmPols.Main.B4[lastN], cmPols.Main.B5[lastN], cmPols.Main.B6[lastN], cmPols.Main.B7[lastN]) +
               " lastN=" + to_string(lastN));

    // log old and new BlobAccInputHash
    // note: we need /*pProverRequest->pFullTracer->get_new_blob_acc_input_hash() */
    zklog.info("Prover::genBlobInnerProof() called executor.executeBlobInner() oldBlobAccInputHash=" + pProverRequest->input.publicInputsExtended.publicInputs.oldBlobAccInputHash.get_str(16) +
               " newBlobAccInputHash=" /* + pProverRequest->pFullTracer->get_new_blob_acc_input_hash()*/ +
               " pols.C[0]=" + fea2string(fr, cmPols.Main.C0[0], cmPols.Main.C1[0], cmPols.Main.C2[0], cmPols.Main.C3[0], cmPols.Main.C4[0], cmPols.Main.C5[0], cmPols.Main.C6[0], cmPols.Main.C7[0]) +
               " pols.C[lastN]=" + fea2string(fr, cmPols.Main.C0[lastN], cmPols.Main.C1[lastN], cmPols.Main.C2[lastN], cmPols.Main.C3[lastN], cmPols.Main.C4[lastN], cmPols.Main.C5[lastN], cmPols.Main.C6[lastN], cmPols.Main.C7[lastN]) +
               " lastN=" + to_string(lastN));
}