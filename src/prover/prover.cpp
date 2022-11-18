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

Prover::Prover(Goldilocks &fr,
               PoseidonGoldilocks &poseidon,
               const Config &config) : fr(fr),
                                       poseidon(poseidon),
                                       executor(fr, config, poseidon),
                                       stark(config),
                                       starkC12a(config),
                                       starkRecursive1(config),
                                       starkRecursive2(config),
                                       starkRecursiveF(config),
                                       // starksC12a(config, {config.c12aConstPols, config.mapConstPolsFile, config.c12aConstantsTree, config.c12aStarkInfo}),
                                       // starksRecursive1(config, {config.recursive1ConstPols, config.mapConstPolsFile, config.recursive1ConstantsTree, config.recursive1StarkInfo}),
                                       // starksRecursive2(config, {config.recursive2ConstPols, config.mapConstPolsFile, config.recursive2ConstantsTree, config.recursive2StarkInfo}),
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

            lastComputedRequestEndTime = 0;

            sem_init(&pendingRequestSem, 0, 0);
            pthread_mutex_init(&mutex, NULL);
            pCurrentRequest = NULL;
            pthread_create(&proverPthread, NULL, proverThread, this);
            pthread_create(&cleanerPthread, NULL, cleanerThread, this);
        }
    }
    catch (std::exception &e)
    {
        cerr << "Error: Prover::Prover() got an exception: " << e.what() << '\n';
        exitProcess();
    }
}

Prover::~Prover()
{
    //· delete zkey;
    //· delete groth16Prover;
    mpz_clear(altBbn128r);
}

void *proverThread(void *arg)
{
    Prover *pProver = (Prover *)arg;
    cout << "proverThread() started" << endl;

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
            cout << "proverThread() found pending requests queue empty, so ignoring" << endl;
            continue;
        }

        // Extract the first pending request (first in, first out)
        pProver->pCurrentRequest = pProver->pendingRequests[0];
        pProver->pCurrentRequest->startTime = time(NULL);
        pProver->pendingRequests.erase(pProver->pendingRequests.begin());

        cout << "proverThread() starting to process request with UUID: " << pProver->pCurrentRequest->uuid << endl;

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
        default:
            cerr << "Error: proverThread() got an invalid prover request type=" << pProver->pCurrentRequest->type << endl;
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

        cout << "proverThread() done processing request with UUID: " << pProverRequest->uuid << endl;

        // Release the prove request semaphore to notify any blocked waiting call
        pProverRequest->notifyCompleted();
    }
    cout << "proverThread() done" << endl;
    return NULL;
}

void *cleanerThread(void *arg)
{
    Prover *pProver = (Prover *)arg;
    cout << "cleanerThread() started" << endl;

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
                    cout << "cleanerThread() deleting request with uuid: " << pProver->completedRequests[i]->uuid << endl;
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
    cout << "cleanerThread() done" << endl;
    return NULL;
}

string Prover::submitRequest(ProverRequest *pProverRequest) // returns UUID for this request
{
    zkassert(config.generateProof());
    zkassert(pProverRequest != NULL);

    cout << "Prover::submitRequest() started type=" << pProverRequest->type << endl;

    // Get the prover request UUID
    string uuid = pProverRequest->uuid;

    // Add the request to the pending requests queue, and release the semaphore to notify the prover thread
    lock();
    requestsMap[uuid] = pProverRequest;
    pendingRequests.push_back(pProverRequest);
    sem_post(&pendingRequestSem);
    unlock();

    cout << "Prover::submitRequest() returns UUID: " << uuid << endl;
    return uuid;
}

ProverRequest *Prover::waitForRequestToComplete(const string &uuid, const uint64_t timeoutInSeconds) // wait for the request with this UUID to complete; returns NULL if UUID is invalid
{
    zkassert(config.generateProof());
    zkassert(uuid.size() > 0);
    cout << "Prover::waitForRequestToComplete() waiting for request with UUID: " << uuid << endl;

    // We will store here the address of the prove request corresponding to this UUID
    ProverRequest *pProverRequest = NULL;

    lock();

    // Map uuid to the corresponding prover request
    std::unordered_map<std::string, ProverRequest *>::iterator it = requestsMap.find(uuid);
    if (it == requestsMap.end())
    {
        cerr << "Prover::waitForRequestToComplete() unknown uuid: " << uuid << endl;
        unlock();
        return NULL;
    }

    // Wait for the request to complete
    pProverRequest = it->second;
    unlock();
    pProverRequest->waitForCompleted(timeoutInSeconds);
    cout << "Prover::waitForRequestToComplete() done waiting for request with UUID: " << uuid << endl;

    // Return the request pointer
    return pProverRequest;
}

void Prover::processBatch(ProverRequest *pProverRequest)
{
    TimerStart(PROVER_PROCESS_BATCH);
    zkassert(pProverRequest != NULL);
    zkassert(pProverRequest->type == prt_processBatch);

    cout << "Prover::processBatch() timestamp: " << pProverRequest->timestamp << endl;
    cout << "Prover::processBatch() UUID: " << pProverRequest->uuid << endl;

    // Save input to <timestamp>.input.json, as provided by client
    if (config.saveInputToFile)
    {
        json inputJson;
        pProverRequest->input.save(inputJson);
        json2file(inputJson, pProverRequest->inputFile);
    }

    // Execute the program, in the process batch way
    executor.process_batch(*pProverRequest);

    // Save input to <timestamp>.input.json after execution including dbReadLog
    if (config.saveDbReadsToFile)
    {
        json inputJsonEx;
        pProverRequest->input.save(inputJsonEx, *pProverRequest->dbReadLog);
        json2file(inputJsonEx, pProverRequest->inputFileEx);
    }

    TimerStopAndLog(PROVER_PROCESS_BATCH);
}

void Prover::genProof(ProverRequest *pProverRequest)
{
    // Deprecated - use genProofBatch instead
    return;
}

void Prover::genBatchProof(ProverRequest *pProverRequest)
{
    zkassert(config.generateProof());
    zkassert(pProverRequest != NULL);

    TimerStart(PROVER_BATCH_PROOF);

    printMemoryInfo();
    printProcessInfo();

    zkassert(pProverRequest != NULL);

    cout << "Prover::genBatchProof() timestamp: " << pProverRequest->timestamp << endl;
    cout << "Prover::genBatchProof() UUID: " << pProverRequest->uuid << endl;
    cout << "Prover::genBatchProof() input file: " << pProverRequest->inputFile << endl;
    cout << "Prover::genBatchProof() public file: " << pProverRequest->publicsOutput << endl;
    cout << "Prover::genBatchProof() proof file: " << pProverRequest->proofFile << endl;

    // Save input to <timestamp>.input.json, as provided by client
    if (config.saveInputToFile)
    {
        json inputJson;
        pProverRequest->input.save(inputJson);
        json2file(inputJson, pProverRequest->inputFile);
    }

    /************/
    /* Executor */
    /************/

    // Allocate an area of memory, mapped to file, to store all the committed polynomials,
    // and create them using the allocated address
    void *pAddress = NULL;
    uint64_t polsSize = stark.getTotalPolsSize();
    zkassert(CommitPols::pilSize() <= polsSize);
    zkassert(CommitPols::pilSize() == stark.getCommitPolsSize());

    if (config.zkevmCmPols.size() > 0)
    {
        pAddress = mapFile(config.zkevmCmPols, polsSize, true);
        cout << "Prover::genBatchProof() successfully mapped " << polsSize << " bytes to file " << config.zkevmCmPols << endl;
    }
    else
    {
        pAddress = calloc(polsSize, 1);
        if (pAddress == NULL)
        {
            cerr << "Error: Prover::genBatchProof() failed calling malloc() of size " << polsSize << endl;
            exitProcess();
        }
        cout << "Prover::genBatchProof() successfully allocated " << polsSize << " bytes" << endl;
    }

    CommitPols cmPols(pAddress, CommitPols::pilDegree());

    // Execute all the State Machines
    TimerStart(EXECUTOR_EXECUTE_BATCH_PROOF);
    executor.execute(*pProverRequest, cmPols);
    TimerStopAndLog(EXECUTOR_EXECUTE_BATCH_PROOF);

    // Save input to <timestamp>.input.json after execution including dbReadLog
    if (config.saveDbReadsToFile)
    {
        json inputJsonEx;
        pProverRequest->input.save(inputJsonEx, *pProverRequest->dbReadLog);
        json2file(inputJsonEx, pProverRequest->inputFileEx);
    }

    if (pProverRequest->result == ZKR_SUCCESS)
    {
        /*************************************/
        /*  Generate publics input           */
        /*************************************/
        TimerStart(SAVE_PUBLICS_JSON_BATCH_PROOF);
        json publicStarkJson;

        uint64_t lastN = cmPols.pilDegree() - 1;

        // oldStateRoot
        publicStarkJson[0] = fr.toString(cmPols.Main.B0[0]);
        publicStarkJson[1] = fr.toString(cmPols.Main.B1[0]);
        publicStarkJson[2] = fr.toString(cmPols.Main.B2[0]);
        publicStarkJson[3] = fr.toString(cmPols.Main.B3[0]);
        publicStarkJson[4] = fr.toString(cmPols.Main.B4[0]);
        publicStarkJson[5] = fr.toString(cmPols.Main.B5[0]);
        publicStarkJson[6] = fr.toString(cmPols.Main.B6[0]);
        publicStarkJson[7] = fr.toString(cmPols.Main.B7[0]);

        // oldAccInputHash
        publicStarkJson[8] = fr.toString(cmPols.Main.C0[0]);
        publicStarkJson[9] = fr.toString(cmPols.Main.C1[0]);
        publicStarkJson[10] = fr.toString(cmPols.Main.C2[0]);
        publicStarkJson[11] = fr.toString(cmPols.Main.C3[0]);
        publicStarkJson[12] = fr.toString(cmPols.Main.C4[0]);
        publicStarkJson[13] = fr.toString(cmPols.Main.C5[0]);
        publicStarkJson[14] = fr.toString(cmPols.Main.C6[0]);
        publicStarkJson[15] = fr.toString(cmPols.Main.C7[0]);

        // oldBatchNum
        publicStarkJson[16] = fr.toString(cmPols.Main.SP[0]);
        // chainId
        publicStarkJson[17] = fr.toString(cmPols.Main.GAS[0]);

        // newStateRoot
        publicStarkJson[18] = fr.toString(cmPols.Main.SR0[lastN]);
        publicStarkJson[19] = fr.toString(cmPols.Main.SR1[lastN]);
        publicStarkJson[20] = fr.toString(cmPols.Main.SR2[lastN]);
        publicStarkJson[21] = fr.toString(cmPols.Main.SR3[lastN]);
        publicStarkJson[22] = fr.toString(cmPols.Main.SR4[lastN]);
        publicStarkJson[23] = fr.toString(cmPols.Main.SR5[lastN]);
        publicStarkJson[24] = fr.toString(cmPols.Main.SR6[lastN]);
        publicStarkJson[25] = fr.toString(cmPols.Main.SR7[lastN]);

        // newAccInputHash
        publicStarkJson[26] = fr.toString(cmPols.Main.D0[lastN]);
        publicStarkJson[27] = fr.toString(cmPols.Main.D1[lastN]);
        publicStarkJson[28] = fr.toString(cmPols.Main.D2[lastN]);
        publicStarkJson[29] = fr.toString(cmPols.Main.D3[lastN]);
        publicStarkJson[30] = fr.toString(cmPols.Main.D4[lastN]);
        publicStarkJson[31] = fr.toString(cmPols.Main.D5[lastN]);
        publicStarkJson[32] = fr.toString(cmPols.Main.D6[lastN]);
        publicStarkJson[33] = fr.toString(cmPols.Main.D7[lastN]);

        // localExitRoot
        publicStarkJson[34] = fr.toString(cmPols.Main.E0[lastN]);
        publicStarkJson[35] = fr.toString(cmPols.Main.E1[lastN]);
        publicStarkJson[36] = fr.toString(cmPols.Main.E2[lastN]);
        publicStarkJson[37] = fr.toString(cmPols.Main.E3[lastN]);
        publicStarkJson[38] = fr.toString(cmPols.Main.E4[lastN]);
        publicStarkJson[39] = fr.toString(cmPols.Main.E5[lastN]);
        publicStarkJson[40] = fr.toString(cmPols.Main.E6[lastN]);
        publicStarkJson[41] = fr.toString(cmPols.Main.E7[lastN]);

        // newBatchNum
        publicStarkJson[42] = fr.toString(cmPols.Main.PC[lastN]);

        TimerStopAndLog(SAVE_PUBLICS_JSON_BATCH_PROOF);

        /*************************************/
        /*  Generate stark proof            */
        /*************************************/
        TimerStart(STARK_PROOF_BATCH_PROOF);
        uint64_t polBits = stark.starkInfo.starkStruct.steps[stark.starkInfo.starkStruct.steps.size() - 1].nBits;
        FRIProof fproof((1 << polBits), FIELD_EXTENSION, stark.starkInfo.starkStruct.steps.size(), stark.starkInfo.evMap.size(), stark.starkInfo.nPublics);
        stark.genProof(pAddress, fproof);
        TimerStopAndLog(STARK_PROOF_BATCH_PROOF);

        TimerStart(STARK_JSON_GENERATION_BATCH_PROOF);

        nlohmann::ordered_json jProof = fproof.proofs.proof2json();
        nlohmann::json zkin = proof2zkinStark(jProof);
        // Generate publics
        jProof["publics"] = publicStarkJson;
        zkin["publics"] = publicStarkJson;

        TimerStopAndLog(STARK_JSON_GENERATION_BATCH_PROOF);

        CommitPolsStarks cmPols12a(pAddress, (1 << starkC12a.starkInfo.starkStruct.nBits));

        Circom::getCommitedPols(&cmPols12a, config.zkevmVerifier, config.c12aExec, zkin);

        //-------------------------------------------
        /* Generate C12a stark proof             */
        //-------------------------------------------
        TimerStart(STARK_C12_A_PROOF_BATCH_PROOF);
        uint64_t polBitsC12 = starkC12a.starkInfo.starkStruct.steps[starkC12a.starkInfo.starkStruct.steps.size() - 1].nBits;
        FRIProof fproofC12a((1 << polBitsC12), FIELD_EXTENSION, starkC12a.starkInfo.starkStruct.steps.size(), starkC12a.starkInfo.evMap.size(), starkC12a.starkInfo.nPublics);

        Goldilocks::Element publics[43];

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

        // newStateRoot
        publics[18] = cmPols.Main.SR0[lastN];
        publics[19] = cmPols.Main.SR1[lastN];
        publics[20] = cmPols.Main.SR2[lastN];
        publics[21] = cmPols.Main.SR3[lastN];
        publics[22] = cmPols.Main.SR4[lastN];
        publics[23] = cmPols.Main.SR5[lastN];
        publics[24] = cmPols.Main.SR6[lastN];
        publics[25] = cmPols.Main.SR7[lastN];

        // newAccInputHash
        publics[26] = cmPols.Main.D0[lastN];
        publics[27] = cmPols.Main.D1[lastN];
        publics[28] = cmPols.Main.D2[lastN];
        publics[29] = cmPols.Main.D3[lastN];
        publics[30] = cmPols.Main.D4[lastN];
        publics[31] = cmPols.Main.D5[lastN];
        publics[32] = cmPols.Main.D6[lastN];
        publics[33] = cmPols.Main.D7[lastN];

        // localExitRoot
        publics[34] = cmPols.Main.E0[lastN];
        publics[35] = cmPols.Main.E1[lastN];
        publics[36] = cmPols.Main.E2[lastN];
        publics[37] = cmPols.Main.E3[lastN];
        publics[38] = cmPols.Main.E4[lastN];
        publics[39] = cmPols.Main.E5[lastN];
        publics[40] = cmPols.Main.E6[lastN];
        publics[41] = cmPols.Main.E7[lastN];

        // newBatchNum
        publics[42] = cmPols.Main.PC[lastN];
        
        // Generate the proof
        starkC12a.genProof(pAddress, fproofC12a, publics);
        TimerStopAndLog(STARK_C12_A_PROOF_BATCH_PROOF);

        // Save the proof & zkinproof
        nlohmann::ordered_json jProofc12a = fproofC12a.proofs.proof2json();
        nlohmann::json zkinC12a = proof2zkinStark(jProofc12a);

        // Add the recursive2 verification key
        json rootC, recursive2Verkey;
        file2json(config.recursive2Verkey, recursive2Verkey);
        rootC[0] = to_string(recursive2Verkey["constRoot"][0]);
        rootC[1] = to_string(recursive2Verkey["constRoot"][1]);
        rootC[2] = to_string(recursive2Verkey["constRoot"][2]);
        rootC[3] = to_string(recursive2Verkey["constRoot"][3]);
        zkinC12a["publics"] = publicStarkJson;
        zkinC12a["rootC"] = rootC;

        CommitPolsStarks cmPolsRecursive1(pAddress, starkRecursive1.starkInfo.starkStruct.nBits);

        CircomRecursive1::getCommitedPols(&cmPolsRecursive1, config.recursive1Verifier, config.recursive1Exec, zkinC12a);

        //-------------------------------------------
        /* Generate Recursive 1 proof            */
        //-------------------------------------------

        TimerStart(STARK_RECURSIVE_1_PROOF_BATCH_PROOF);
        uint64_t polBitsRecursive1 = starkRecursive1.starkInfo.starkStruct.steps[starkRecursive1.starkInfo.starkStruct.steps.size() - 1].nBits;
        FRIProof fproofRecursive1((1 << polBitsRecursive1), FIELD_EXTENSION, starkRecursive1.starkInfo.starkStruct.steps.size(), starkRecursive1.starkInfo.evMap.size(), starkRecursive1.starkInfo.nPublics);
        starkRecursive1.genProof(pAddress, fproofRecursive1, publics);
        TimerStopAndLog(STARK_RECURSIVE_1_PROOF_BATCH_PROOF);

        // Add the recursive2 verification key
        publicStarkJson[43] = to_string(recursive2Verkey["constRoot"][0]);
        publicStarkJson[44] = to_string(recursive2Verkey["constRoot"][1]);
        publicStarkJson[45] = to_string(recursive2Verkey["constRoot"][2]);
        publicStarkJson[46] = to_string(recursive2Verkey["constRoot"][3]);
        json2file(publicStarkJson, pProverRequest->publicsOutput);

        // Save the proof & zkinproof
        nlohmann::ordered_json jProofRecursive1 = fproofRecursive1.proofs.proof2json();
        nlohmann::ordered_json zkinRecursive1 = proof2zkinStark(jProofRecursive1);
        zkinRecursive1["publics"] = publicStarkJson;

        pProverRequest->batchProofOutput = zkinRecursive1;

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

        // Unmap committed polynomials address
        if (config.zkevmCmPols.size() > 0)
        {
            unmapFile(pAddress, polsSize);
        }
        else
        {
            free(pAddress);
        }
    }

    TimerStopAndLog(PROVER_BATCH_PROOF);
}

void Prover::genAggregatedProof(ProverRequest *pProverRequest)
{

    zkassert(config.generateProof());
    zkassert(pProverRequest != NULL);
    zkassert(pProverRequest->type == prt_genAggregatedProof);

    TimerStart(PROVER_AGGREGATED_PROOF);

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
        std::cerr << pProverRequest->aggregatedProofInput1["publics"][17] << "!=" << pProverRequest->aggregatedProofInput2["publics"][17] << std::endl;
        return;
    }
    // Check midStateRoot
    for (int i = 0; i < 8; i++)
    {
        if (pProverRequest->aggregatedProofInput1["publics"][18 + i] != pProverRequest->aggregatedProofInput2["publics"][0 + i])
        {
            std::cerr << pProverRequest->aggregatedProofInput1["publics"][18 + i] << "!=" << pProverRequest->aggregatedProofInput2["publics"][0 + i] << std::endl;
            return;
        }
    }
    // Check midAccInputHash0
    for (int i = 0; i < 8; i++)
    {
        if (pProverRequest->aggregatedProofInput1["publics"][26 + i] != pProverRequest->aggregatedProofInput2["publics"][8 + i])
        {
            std::cerr << pProverRequest->aggregatedProofInput1["publics"][26 + i] << "!=" << pProverRequest->aggregatedProofInput2["publics"][8 + i] << std::endl;
            return;
        }
    }
    // Check batchNum
    if (pProverRequest->aggregatedProofInput1["publics"][42] != pProverRequest->aggregatedProofInput2["publics"][16])
    {
        std::cerr << pProverRequest->aggregatedProofInput1["publics"][42] << "!=" << pProverRequest->aggregatedProofInput2["publics"][16] << std::endl;
        return;
    }

    json zkinInputRecursive2 = joinzkin(pProverRequest->aggregatedProofInput1, pProverRequest->aggregatedProofInput2, verKey);

    Goldilocks::Element publics[43];
    // oldStateRoot
    publics[0] = Goldilocks::fromString(zkinInputRecursive2["publics"][0]);
    publics[1] = Goldilocks::fromString(zkinInputRecursive2["publics"][1]);
    publics[2] = Goldilocks::fromString(zkinInputRecursive2["publics"][2]);
    publics[3] = Goldilocks::fromString(zkinInputRecursive2["publics"][3]);
    publics[4] = Goldilocks::fromString(zkinInputRecursive2["publics"][4]);
    publics[5] = Goldilocks::fromString(zkinInputRecursive2["publics"][5]);
    publics[6] = Goldilocks::fromString(zkinInputRecursive2["publics"][6]);
    publics[7] = Goldilocks::fromString(zkinInputRecursive2["publics"][7]);

    // oldAccInputHash
    publics[8] = Goldilocks::fromString(zkinInputRecursive2["publics"][8]);
    publics[9] = Goldilocks::fromString(zkinInputRecursive2["publics"][9]);
    publics[10] = Goldilocks::fromString(zkinInputRecursive2["publics"][10]);
    publics[11] = Goldilocks::fromString(zkinInputRecursive2["publics"][11]);
    publics[12] = Goldilocks::fromString(zkinInputRecursive2["publics"][12]);
    publics[13] = Goldilocks::fromString(zkinInputRecursive2["publics"][13]);
    publics[14] = Goldilocks::fromString(zkinInputRecursive2["publics"][14]);
    publics[15] = Goldilocks::fromString(zkinInputRecursive2["publics"][15]);

    // oldBatchNum
    publics[16] = Goldilocks::fromString(zkinInputRecursive2["publics"][16]);
    // chainId
    publics[17] = Goldilocks::fromString(zkinInputRecursive2["publics"][17]);

    // newStateRoot
    publics[18] = Goldilocks::fromString(zkinInputRecursive2["publics"][18]);
    publics[19] = Goldilocks::fromString(zkinInputRecursive2["publics"][19]);
    publics[20] = Goldilocks::fromString(zkinInputRecursive2["publics"][20]);
    publics[21] = Goldilocks::fromString(zkinInputRecursive2["publics"][21]);
    publics[22] = Goldilocks::fromString(zkinInputRecursive2["publics"][22]);
    publics[23] = Goldilocks::fromString(zkinInputRecursive2["publics"][23]);
    publics[24] = Goldilocks::fromString(zkinInputRecursive2["publics"][24]);
    publics[25] = Goldilocks::fromString(zkinInputRecursive2["publics"][25]);

    // newAccInputHash
    publics[26] = Goldilocks::fromString(zkinInputRecursive2["publics"][26]);
    publics[27] = Goldilocks::fromString(zkinInputRecursive2["publics"][27]);
    publics[28] = Goldilocks::fromString(zkinInputRecursive2["publics"][28]);
    publics[29] = Goldilocks::fromString(zkinInputRecursive2["publics"][29]);
    publics[30] = Goldilocks::fromString(zkinInputRecursive2["publics"][30]);
    publics[31] = Goldilocks::fromString(zkinInputRecursive2["publics"][31]);
    publics[32] = Goldilocks::fromString(zkinInputRecursive2["publics"][32]);
    publics[33] = Goldilocks::fromString(zkinInputRecursive2["publics"][33]);

    // localExitRoot
    publics[34] = Goldilocks::fromString(zkinInputRecursive2["publics"][34]);
    publics[35] = Goldilocks::fromString(zkinInputRecursive2["publics"][35]);
    publics[36] = Goldilocks::fromString(zkinInputRecursive2["publics"][36]);
    publics[37] = Goldilocks::fromString(zkinInputRecursive2["publics"][37]);
    publics[38] = Goldilocks::fromString(zkinInputRecursive2["publics"][38]);
    publics[39] = Goldilocks::fromString(zkinInputRecursive2["publics"][39]);
    publics[40] = Goldilocks::fromString(zkinInputRecursive2["publics"][40]);
    publics[41] = Goldilocks::fromString(zkinInputRecursive2["publics"][41]);

    // newBatchNum
    publics[42] = Goldilocks::fromString(zkinInputRecursive2["publics"][42]);

    uint64_t polsSizeRecursive2 = starkRecursive2.getTotalPolsSize();
    void *pAddressRecursive2 = (void *)malloc(polsSizeRecursive2);
    CommitPolsStarks cmPolsRecursive2(pAddressRecursive2, starkRecursive1.starkInfo.starkStruct.nBits);
    CircomRecursive2::getCommitedPols(&cmPolsRecursive2, config.recursive2Verifier, config.recursive2Exec, zkinInputRecursive2);

    //-------------------------------------------
    // Generate Recursive 2 proof
    //-------------------------------------------

    TimerStart(STARK_RECURSIVE_2_PROOF_BATCH_PROOF);
    uint64_t polBitsRecursive2 = starkRecursive2.starkInfo.starkStruct.steps[starkRecursive2.starkInfo.starkStruct.steps.size() - 1].nBits;
    FRIProof fproofRecursive2((1 << polBitsRecursive2), FIELD_EXTENSION, starkRecursive2.starkInfo.starkStruct.steps.size(), starkRecursive2.starkInfo.evMap.size(), starkRecursive2.starkInfo.nPublics);
    starkRecursive2.genProof(pAddressRecursive2, fproofRecursive2, publics);
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
    json recursive2Verkey;

    file2json(config.recursive2Verkey, recursive2Verkey);

    for (int i = 0; i < 43; i++)
    {
        publicsJson[i] = zkinInputRecursive2["publics"][i];
    }
    // Add the recursive2 verification key
    publicsJson[43] = to_string(recursive2Verkey["constRoot"][0]);
    publicsJson[44] = to_string(recursive2Verkey["constRoot"][1]);
    publicsJson[45] = to_string(recursive2Verkey["constRoot"][2]);
    publicsJson[46] = to_string(recursive2Verkey["constRoot"][3]);

    json2file(publicsJson, pProverRequest->publicsOutput);

    free(pAddressRecursive2);
    TimerStopAndLog(PROVER_AGGREGATED_PROOF);
}

void Prover::genFinalProof(ProverRequest *pProverRequest)
{
    zkassert(config.generateProof());
    zkassert(pProverRequest != NULL);
    zkassert(pProverRequest->type == prt_genFinalProof);

    TimerStart(PROVER_FINAL_PROOF);

    // Save input to file
    if (config.saveInputToFile)
    {
        json2file(pProverRequest->finalProofInput, pProverRequest->filePrefix + "final_proof.input.json");
    }

    // Input is pProverRequest->finalProofInput (of type json)
    mpz_t address;
    mpz_init_set_str(address, pProverRequest->input.publicInputsExtended.publicInputs.aggregatorAddress.c_str(), 0);
    std::string strAddress = mpz_get_str(0, 16, address);
    std::string strAddress10 = mpz_get_str(0, 10, address);
    mpz_clear(address);

    json zkinFinal = pProverRequest->finalProofInput;

    Goldilocks::Element publics[43];
    // oldStateRoot
    publics[0] = Goldilocks::fromString(zkinFinal["publics"][0]);
    publics[1] = Goldilocks::fromString(zkinFinal["publics"][1]);
    publics[2] = Goldilocks::fromString(zkinFinal["publics"][2]);
    publics[3] = Goldilocks::fromString(zkinFinal["publics"][3]);
    publics[4] = Goldilocks::fromString(zkinFinal["publics"][4]);
    publics[5] = Goldilocks::fromString(zkinFinal["publics"][5]);
    publics[6] = Goldilocks::fromString(zkinFinal["publics"][6]);
    publics[7] = Goldilocks::fromString(zkinFinal["publics"][7]);

    // oldAccInputHash
    publics[8] = Goldilocks::fromString(zkinFinal["publics"][8]);
    publics[9] = Goldilocks::fromString(zkinFinal["publics"][9]);
    publics[10] = Goldilocks::fromString(zkinFinal["publics"][10]);
    publics[11] = Goldilocks::fromString(zkinFinal["publics"][11]);
    publics[12] = Goldilocks::fromString(zkinFinal["publics"][12]);
    publics[13] = Goldilocks::fromString(zkinFinal["publics"][13]);
    publics[14] = Goldilocks::fromString(zkinFinal["publics"][14]);
    publics[15] = Goldilocks::fromString(zkinFinal["publics"][15]);

    // oldBatchNum
    publics[16] = Goldilocks::fromString(zkinFinal["publics"][16]);
    // chainId
    publics[17] = Goldilocks::fromString(zkinFinal["publics"][17]);

    // newStateRoot
    publics[18] = Goldilocks::fromString(zkinFinal["publics"][18]);
    publics[19] = Goldilocks::fromString(zkinFinal["publics"][19]);
    publics[20] = Goldilocks::fromString(zkinFinal["publics"][20]);
    publics[21] = Goldilocks::fromString(zkinFinal["publics"][21]);
    publics[22] = Goldilocks::fromString(zkinFinal["publics"][22]);
    publics[23] = Goldilocks::fromString(zkinFinal["publics"][23]);
    publics[24] = Goldilocks::fromString(zkinFinal["publics"][24]);
    publics[25] = Goldilocks::fromString(zkinFinal["publics"][25]);

    // newAccInputHash
    publics[26] = Goldilocks::fromString(zkinFinal["publics"][26]);
    publics[27] = Goldilocks::fromString(zkinFinal["publics"][27]);
    publics[28] = Goldilocks::fromString(zkinFinal["publics"][28]);
    publics[29] = Goldilocks::fromString(zkinFinal["publics"][29]);
    publics[30] = Goldilocks::fromString(zkinFinal["publics"][30]);
    publics[31] = Goldilocks::fromString(zkinFinal["publics"][31]);
    publics[32] = Goldilocks::fromString(zkinFinal["publics"][32]);
    publics[33] = Goldilocks::fromString(zkinFinal["publics"][33]);

    // localExitRoot
    publics[34] = Goldilocks::fromString(zkinFinal["publics"][34]);
    publics[35] = Goldilocks::fromString(zkinFinal["publics"][35]);
    publics[36] = Goldilocks::fromString(zkinFinal["publics"][36]);
    publics[37] = Goldilocks::fromString(zkinFinal["publics"][37]);
    publics[38] = Goldilocks::fromString(zkinFinal["publics"][38]);
    publics[39] = Goldilocks::fromString(zkinFinal["publics"][39]);
    publics[40] = Goldilocks::fromString(zkinFinal["publics"][40]);
    publics[41] = Goldilocks::fromString(zkinFinal["publics"][41]);

    // newBatchNum
    publics[42] = Goldilocks::fromString(zkinFinal["publics"][42]);

    uint64_t polsSizeRecursiveF = starkRecursive2.getTotalPolsSize();
    void *pAddressRecursiveF = (void *)malloc(polsSizeRecursiveF);
    CommitPolsStarks cmPolsRecursive2(pAddressRecursiveF, starkRecursiveF.starkInfo.starkStruct.nBits);
    CircomRecursiveF::getCommitedPols(&cmPolsRecursive2, config.recursivefVerifier, config.recursivefExec, zkinFinal);

    //  ----------------------------------------------
    //  Generate Recursive Final proof
    //  ----------------------------------------------

    TimerStart(STARK_RECURSIVE_F_PROOF_BATCH_PROOF);
    uint64_t polBitsRecursiveF = starkRecursiveF.starkInfo.starkStruct.steps[starkRecursiveF.starkInfo.starkStruct.steps.size() - 1].nBits;
    FRIProofC12 fproofRecursiveF((1 << polBitsRecursiveF), FIELD_EXTENSION, starkRecursiveF.starkInfo.starkStruct.steps.size(), starkRecursiveF.starkInfo.evMap.size(), starkRecursiveF.starkInfo.nPublics);
    starkRecursiveF.genProof(pAddressRecursiveF, fproofRecursiveF, publics);

    // Save the proof & zkinproof
    nlohmann::ordered_json jProofRecursiveF = fproofRecursiveF.proofs.proof2json();
    json zkinRecursiveF = proof2zkinStark(jProofRecursiveF);
    zkinRecursiveF["publics"] = zkinFinal["publics"];
    zkinRecursiveF["aggregatorAddr"] = strAddress10;

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
        cerr << "Error: Prover::genProof() Not all inputs have been set. Only " << CircomFinal::get_main_input_signal_no() - ctxFinal->getRemaingInputsToBeSet() << " out of " << CircomFinal::get_main_input_signal_no() << endl;
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
    json2file(publicJson, pProverRequest->publicsOutput);
    TimerStopAndLog(SAVE_PUBLICS_JSON);

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
        cerr << "Error: Prover::genProof() got exception in rapid SNARK:" << e.what() << '\n';
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
    // publicInputsExtended.inputHash = NormalizeTo0xNFormat(fr.toString(cmPols.Main.FREE0[0], 16), 64);
    pProverRequest->proof.load(jsonProof, publicInputsExtended);

    /***********/
    /* Cleanup */
    /***********/
    free(pWitnessFinal);
    free(pAddressRecursiveF);

    TimerStopAndLog(STARK_RECURSIVE_F_PROOF_BATCH_PROOF);
    TimerStopAndLog(PROVER_FINAL_PROOF);
}