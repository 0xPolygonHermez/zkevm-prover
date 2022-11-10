#include <fstream>
#include <iomanip>
#include <unistd.h>
#include "prover.hpp"
#include "utils.hpp"
#include "scalar.hpp"
#include "proof2zkin.hpp"
#include "main.hpp"
#include "main.recursive1.hpp"
#include "binfile_utils.hpp"
#include "zkey_utils.hpp"
#include "wtns_utils.hpp"
#include "groth16.hpp"
#include "sm/storage/storage_executor.hpp"
#include "timer.hpp"
#include "execFile.hpp"
#include <math.h> /* log2 */
#include "commit_pols_c12a.hpp"

#include "friProofC12.hpp"
#include <algorithm> // std::min
#include <openssl/sha.h>
#include "utils.hpp"
using namespace std;

Prover::Prover(Goldilocks &fr,
               PoseidonGoldilocks &poseidon,
               const Config &config) : fr(fr),
                                       poseidon(poseidon),
                                       executor(fr, config, poseidon),
                                       stark(config),
                                       starkC12a(config),
                                       starkRecursive1(config),
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
            zkey = BinFileUtils::openExisting(config.starkVerifierFile, "zkey", 1);
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
    cout << "Prover::genBatchProof() public file: " << pProverRequest->publicFile << endl;
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

    if (config.cmPolsFile.size() > 0)
    {
        pAddress = mapFile(config.cmPolsFile, polsSize, true);
        cout << "Prover::genBatchProof() successfully mapped " << polsSize << " bytes to file " << config.cmPolsFile << endl;
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
        json publicJson;
        mpz_t address;
        mpz_t publicshash;
        json publicStarkJson, oldStateRootPublic, oldAccInputHashPublic, oldBatchNumPublic, chainIdPublic, newStateRootPublic, newAccInputHashPublic, localExitRootPublic, newBatchNumPublic;
        RawFr::Element publicsHash;
        string freeInStrings16[8];
        string newAccInputHashPublicString[8];

        uint64_t last = cmPols.pilDegree() - 1;

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
        publicStarkJson[18] = fr.toString(cmPols.Main.SR0[last]);
        publicStarkJson[19] = fr.toString(cmPols.Main.SR1[last]);
        publicStarkJson[20] = fr.toString(cmPols.Main.SR2[last]);
        publicStarkJson[21] = fr.toString(cmPols.Main.SR3[last]);
        publicStarkJson[22] = fr.toString(cmPols.Main.SR4[last]);
        publicStarkJson[23] = fr.toString(cmPols.Main.SR5[last]);
        publicStarkJson[24] = fr.toString(cmPols.Main.SR6[last]);
        publicStarkJson[25] = fr.toString(cmPols.Main.SR7[last]);

        // newAccInputHash
        publicStarkJson[26] = fr.toString(cmPols.Main.D0[last]);
        publicStarkJson[27] = fr.toString(cmPols.Main.D1[last]);
        publicStarkJson[28] = fr.toString(cmPols.Main.D2[last]);
        publicStarkJson[29] = fr.toString(cmPols.Main.D3[last]);
        publicStarkJson[30] = fr.toString(cmPols.Main.D4[last]);
        publicStarkJson[31] = fr.toString(cmPols.Main.D5[last]);
        publicStarkJson[32] = fr.toString(cmPols.Main.D6[last]);
        publicStarkJson[33] = fr.toString(cmPols.Main.D7[last]);

        // localExitRoot
        publicStarkJson[34] = fr.toString(cmPols.Main.E0[last]);
        publicStarkJson[35] = fr.toString(cmPols.Main.E1[last]);
        publicStarkJson[36] = fr.toString(cmPols.Main.E2[last]);
        publicStarkJson[37] = fr.toString(cmPols.Main.E3[last]);
        publicStarkJson[38] = fr.toString(cmPols.Main.E4[last]);
        publicStarkJson[39] = fr.toString(cmPols.Main.E5[last]);
        publicStarkJson[40] = fr.toString(cmPols.Main.E6[last]);
        publicStarkJson[41] = fr.toString(cmPols.Main.E7[last]);

        // newBatchNum
        publicStarkJson[42] = fr.toString(cmPols.Main.PC[last]);

        /* DEPRECATED */
        newAccInputHashPublicString[0] = fr.toString(cmPols.Main.D0[last], 16);
        newAccInputHashPublicString[1] = fr.toString(cmPols.Main.D1[last], 16);
        newAccInputHashPublicString[2] = fr.toString(cmPols.Main.D2[last], 16);
        newAccInputHashPublicString[3] = fr.toString(cmPols.Main.D3[last], 16);
        newAccInputHashPublicString[4] = fr.toString(cmPols.Main.D4[last], 16);
        newAccInputHashPublicString[5] = fr.toString(cmPols.Main.D5[last], 16);
        newAccInputHashPublicString[6] = fr.toString(cmPols.Main.D6[last], 16);
        newAccInputHashPublicString[7] = fr.toString(cmPols.Main.D7[last], 16);
        std::string buffer = "";

        for (uint i = 0; i < 8; i++)
        {
            buffer = buffer + std::string(16 - std::min(16, (int)newAccInputHashPublicString[i].length()), '0') + newAccInputHashPublicString[i];
        }
        mpz_init_set_str(address, pProverRequest->input.publicInputsExtended.publicInputs.aggregatorAddress.c_str(), 0);
        std::string strAddress = mpz_get_str(0, 16, address);
        std::string strAddress10 = mpz_get_str(0, 10, address);
        mpz_clear(address);

        buffer = "";
        buffer = buffer + std::string(40 - std::min(40, (int)strAddress.length()), '0') + strAddress;

        std::string aux;
        for (uint i = 0; i < 8; i++)
        {
            buffer = buffer + std::string(16 - std::min(16, (int)freeInStrings16[i].length()), '0') + freeInStrings16[i];
        }

        mpz_init_set_str(publicshash, sha256(buffer).c_str(), 16);
        std::string publicsHashString = mpz_get_str(0, 10, publicshash);
        RawFr::field.fromString(publicsHash, publicsHashString);
        mpz_clear(publicshash);

        // Save public file
        publicJson[0] = RawFr::field.toString(publicsHash, 10);
        json2file(publicJson, pProverRequest->publicFile);
        json2file(publicStarkJson, config.publicStarkFile);
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

        // Generate publics
        jProof["publics"] = publicStarkJson;
        ofstream ofstark(config.starkFile);
        ofstark << setw(4) << jProof.dump() << endl;
        ofstark.close();

        nlohmann::json zkin = proof2zkinStark(jProof);
        zkin["publics"] = publicStarkJson;
        ofstream ofzkin(config.starkZkIn);
        ofzkin << setw(4) << zkin.dump() << endl;
        ofzkin.close();

        TimerStopAndLog(STARK_JSON_GENERATION_BATCH_PROOF);

        /************************/
        /* Verifier stark proof */
        /************************/

        TimerStart(CIRCOM_LOAD_CIRCUIT_BATCH_PROOF);
        Circom::Circom_Circuit *circuit = Circom::loadCircuit(config.verifierFile);
        TimerStopAndLog(CIRCOM_LOAD_CIRCUIT_BATCH_PROOF);

        TimerStart(CIRCOM_LOAD_JSON_BATCH_PROOF);
        Circom::Circom_CalcWit *ctx = new Circom::Circom_CalcWit(circuit);

        loadJsonImpl(ctx, zkin);
        if (ctx->getRemaingInputsToBeSet() != 0)
        {
            cerr << "Error: Prover::genBatchProof() Not all inputs have been set. Only " << Circom::get_main_input_signal_no() - ctx->getRemaingInputsToBeSet() << " out of " << Circom::get_main_input_signal_no() << endl;
            exitProcess();
        }
        TimerStopAndLog(CIRCOM_LOAD_JSON_BATCH_PROOF);

        // If present, save witness file
        if (config.witnessFile.size() > 0)
        {
            TimerStart(CIRCOM_WRITE_BIN_WITNESS_BATCH_PROOF);
            writeBinWitness(ctx, config.witnessFile); // No need to write the file to disk, 12-13M fe, in binary, in wtns format
            TimerStopAndLog(CIRCOM_WRITE_BIN_WITNESS_BATCH_PROOF);
        }

        /******************************************/
        /* Compute witness and c12a commited pols */
        /******************************************/
        TimerStart(STARK_WITNESS_AND_COMMITED_POLS_BATCH_PROOF);

        ExecFile execC12aFile(config.execC12aFile);
        uint64_t sizeWitness = Circom::get_size_of_witness();
        Goldilocks::Element *tmp = new Goldilocks::Element[execC12aFile.nAdds + sizeWitness];

        //#pragma omp parallel for
        for (uint64_t i = 0; i < sizeWitness; i++)
        {
            FrGElement aux;
            ctx->getWitness(i, &aux);
            FrG_toLongNormal(&aux, &aux);
            tmp[i] = Goldilocks::fromU64(aux.longVal[0]);
        }
        delete ctx;

        for (uint64_t i = 0; i < execC12aFile.nAdds; i++)
        {
            FrG_toLongNormal(&execC12aFile.p_adds[i * 4], &execC12aFile.p_adds[i * 4]);
            FrG_toLongNormal(&execC12aFile.p_adds[i * 4 + 1], &execC12aFile.p_adds[i * 4 + 1]);
            FrG_toLongNormal(&execC12aFile.p_adds[i * 4 + 2], &execC12aFile.p_adds[i * 4 + 2]);
            FrG_toLongNormal(&execC12aFile.p_adds[i * 4 + 3], &execC12aFile.p_adds[i * 4 + 3]);

            uint64_t idx_1 = execC12aFile.p_adds[i * 4].longVal[0];
            uint64_t idx_2 = execC12aFile.p_adds[i * 4 + 1].longVal[0];

            Goldilocks::Element c = tmp[idx_1] * Goldilocks::fromU64(execC12aFile.p_adds[i * 4 + 2].longVal[0]);
            Goldilocks::Element d = tmp[idx_2] * Goldilocks::fromU64(execC12aFile.p_adds[i * 4 + 3].longVal[0]);
            tmp[sizeWitness + i] = c + d;
        }

        uint64_t Nbits = log2(execC12aFile.nSMap - 1) + 1;
        uint64_t N = 1 << Nbits;

        uint64_t polsSizeC12 = starkC12a.getTotalPolsSize();
        cout << "Prover::genBatchProof() starkC12a.getTotalPolsSize()=" << polsSizeC12 << endl;

        // void *pAddressC12 = calloc(polsSizeC12, 1);
        void *pAddressC12 = pAddress;
        CommitPolsC12a cmPols12a(pAddressC12, CommitPolsC12a::pilDegree());

        //#pragma omp parallel for
        for (uint i = 0; i < execC12aFile.nSMap; i++)
        {
            for (uint j = 0; j < 12; j++)
            {
                FrGElement aux;
                FrG_toLongNormal(&aux, &execC12aFile.p_sMap[12 * i + j]);
                uint64_t idx_1 = aux.longVal[0];
                if (idx_1 != 0)
                {
                    uint64_t idx_2 = Goldilocks::toU64(tmp[idx_1]);
                    cmPols12a.Compressor.a[j][i] = Goldilocks::fromU64(idx_2);
                }
                else
                {
                    cmPols12a.Compressor.a[j][i] = Goldilocks::zero();
                }
            }
        }
        for (uint i = execC12aFile.nSMap; i < N; i++)
        {
            for (uint j = 0; j < 12; j++)
            {
                cmPols12a.Compressor.a[j][i] = Goldilocks::zero();
            }
        }
        delete[] tmp;
        Circom::freeCircuit(circuit);
        TimerStopAndLog(STARK_WITNESS_AND_COMMITED_POLS_BATCH_PROOF);

        if (config.cmPolsFileC12a.size() > 0)
        {
            void *pAddressC12tmp = mapFile(config.cmPolsFileC12a, CommitPolsC12a::pilSize(), true);
            cout << "Prover::genBatchProof() successfully mapped " << CommitPolsC12a::pilSize() << " bytes to file "
                 << config.cmPolsFileC12a << endl;
            std::memcpy(pAddressC12tmp, pAddressC12, CommitPolsC12a::pilSize());
            unmapFile(pAddressC12tmp, CommitPolsC12a::pilSize());
        }

        /*****************************************/
        /* Generate C12a stark proof             */
        /*****************************************/
        TimerStart(STARK_C12_A_PROOF_BATCH_PROOF);
        uint64_t polBitsC12 = starkC12a.starkInfo.starkStruct.steps[starkC12a.starkInfo.starkStruct.steps.size() - 1].nBits;
        cout << "polBitsC12=" << polBitsC12 << endl;
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
        publics[18] = cmPols.Main.SR0[last];
        publics[19] = cmPols.Main.SR1[last];
        publics[20] = cmPols.Main.SR2[last];
        publics[21] = cmPols.Main.SR3[last];
        publics[22] = cmPols.Main.SR4[last];
        publics[23] = cmPols.Main.SR5[last];
        publics[24] = cmPols.Main.SR6[last];
        publics[25] = cmPols.Main.SR7[last];

        // newAccInputHash
        publics[26] = cmPols.Main.D0[last];
        publics[27] = cmPols.Main.D1[last];
        publics[28] = cmPols.Main.D2[last];
        publics[29] = cmPols.Main.D3[last];
        publics[30] = cmPols.Main.D4[last];
        publics[31] = cmPols.Main.D5[last];
        publics[32] = cmPols.Main.D6[last];
        publics[33] = cmPols.Main.D7[last];

        // localExitRoot
        publics[34] = cmPols.Main.E0[last];
        publics[35] = cmPols.Main.E1[last];
        publics[36] = cmPols.Main.E2[last];
        publics[37] = cmPols.Main.E3[last];
        publics[38] = cmPols.Main.E4[last];
        publics[39] = cmPols.Main.E5[last];
        publics[40] = cmPols.Main.E6[last];
        publics[41] = cmPols.Main.E7[last];

        // newBatchNum
        publics[42] = cmPols.Main.PC[last];

        // Generate the proof
        starkC12a.genProof(pAddressC12, fproofC12a, publics);
        TimerStopAndLog(STARK_C12_A_PROOF_BATCH_PROOF);

        // Save the proof & zkinproof
        nlohmann::ordered_json jProofc12a = fproofC12a.proofs.proof2json();
        nlohmann::json zkinC12a = proof2zkinStark(jProofc12a);
        json rootC;
        // Hardcoded from recursive2.verkey.json
        rootC[0] = "9475605637534756122";
        rootC[1] = "16487946048269610282";
        rootC[2] = "8652149613517563773";
        rootC[3] = "4744476631234276952";
        zkinC12a["publics"] = publicStarkJson;
        zkinC12a["rootC"] = rootC;
        ofstream ofzkin2c12a(config.starkZkInC12a);
        ofzkin2c12a << setw(4) << zkinC12a.dump() << endl;
        ofzkin2c12a.close();

        jProofc12a["publics"] = publicStarkJson;
        ofstream ofstarkc12a(config.starkFilec12a);
        ofstarkc12a << setw(4) << jProofc12a.dump() << endl;
        ofstarkc12a.close();
        /************************/
        /* Verifier stark proof */
        /************************/

        TimerStart(CIRCOM_LOAD_CIRCUIT_BATCH_PROOF_RECURSIVE_1);
        CircomRecursive1::Circom_Circuit *circuitRecursive1 = CircomRecursive1::loadCircuit(config.verifierFileRecursive1);
        TimerStopAndLog(CIRCOM_LOAD_CIRCUIT_BATCH_PROOF_RECURSIVE_1);

        TimerStart(CIRCOM_LOAD_JSON_BATCH_PROOF_RECURSIVE_1);
        CircomRecursive1::Circom_CalcWit *ctxRecursive1 = new CircomRecursive1::Circom_CalcWit(circuitRecursive1);

        loadJsonImpl(ctxRecursive1, zkinC12a);
        if (ctxRecursive1->getRemaingInputsToBeSet() != 0)
        {
            cerr << "Error: Prover::genBatchProof() Not all inputs have been set. Only " << CircomRecursive1::get_main_input_signal_no() - ctxRecursive1->getRemaingInputsToBeSet() << " out of " << CircomRecursive1::get_main_input_signal_no() << endl;
            exitProcess();
        }
        TimerStopAndLog(CIRCOM_LOAD_JSON_BATCH_PROOF_RECURSIVE_1);

        // If present, save witness file
        if (config.witnessFileRecursive1.size() > 0)
        {
            TimerStart(CIRCOM_WRITE_BIN_WITNESS_BATCH_RECURSIVE_1);
            writeBinWitness(ctxRecursive1, config.witnessFileRecursive1); // No need to write the file to disk, 12-13M fe, in binary, in wtns format
            TimerStopAndLog(CIRCOM_WRITE_BIN_WITNESS_BATCH_RECURSIVE_1);
        }

        /*****************************************************/
        /* Compute witness and recursive1 commited pols      */
        /*****************************************************/

        TimerStart(WITNESS_AND_COMMITED_POLS_BATCH_PROOF_RECURSIVE_1);

        ExecFile execRecursive1File(config.execRecursive1File);
        uint64_t sizeWitnessRecursive1 = CircomRecursive1::get_size_of_witness();
        Goldilocks::Element *tmpRecursive1 = new Goldilocks::Element[execRecursive1File.nAdds + sizeWitnessRecursive1];

        for (uint64_t i = 0; i < sizeWitnessRecursive1; i++)
        {
            FrGElement aux;
            ctxRecursive1->getWitness(i, &aux);
            FrG_toLongNormal(&aux, &aux);
            tmpRecursive1[i] = Goldilocks::fromU64(aux.longVal[0]);
        }
        delete ctxRecursive1;

        for (uint64_t i = 0; i < execRecursive1File.nAdds; i++)
        {
            FrG_toLongNormal(&execRecursive1File.p_adds[i * 4], &execRecursive1File.p_adds[i * 4]);
            FrG_toLongNormal(&execRecursive1File.p_adds[i * 4 + 1], &execRecursive1File.p_adds[i * 4 + 1]);
            FrG_toLongNormal(&execRecursive1File.p_adds[i * 4 + 2], &execRecursive1File.p_adds[i * 4 + 2]);
            FrG_toLongNormal(&execRecursive1File.p_adds[i * 4 + 3], &execRecursive1File.p_adds[i * 4 + 3]);

            uint64_t idx_1 = execRecursive1File.p_adds[i * 4].longVal[0];
            uint64_t idx_2 = execRecursive1File.p_adds[i * 4 + 1].longVal[0];

            Goldilocks::Element c = tmpRecursive1[idx_1] * Goldilocks::fromU64(execRecursive1File.p_adds[i * 4 + 2].longVal[0]);
            Goldilocks::Element d = tmpRecursive1[idx_2] * Goldilocks::fromU64(execRecursive1File.p_adds[i * 4 + 3].longVal[0]);
            tmpRecursive1[sizeWitnessRecursive1 + i] = c + d;
        }

        uint64_t NbitsRecurive1 = log2(execRecursive1File.nSMap - 1) + 1;
        uint64_t NRecurive1 = 1 << NbitsRecurive1;

        uint64_t polsSizeRecursive1 = starkRecursive1.getTotalPolsSize();
        cout << "Prover::genBatchProof() starkRecursive1.getTotalPolsSize()=" << polsSizeRecursive1 << endl;

        void *pAddressRecursive1 = pAddress;
        CommitPolsRecursive1 cmPolsRecursive1(pAddressRecursive1, CommitPolsRecursive1::pilDegree());

        //#pragma omp parallel for
        for (uint i = 0; i < execRecursive1File.nSMap; i++)
        {
            for (uint j = 0; j < 12; j++)
            {
                FrGElement aux;
                FrG_toLongNormal(&aux, &execRecursive1File.p_sMap[12 * i + j]);
                uint64_t idx_1 = aux.longVal[0];
                if (idx_1 != 0)
                {
                    uint64_t idx_2 = Goldilocks::toU64(tmpRecursive1[idx_1]);
                    cmPolsRecursive1.Compressor.a[j][i] = Goldilocks::fromU64(idx_2);
                }
                else
                {
                    cmPolsRecursive1.Compressor.a[j][i] = Goldilocks::zero();
                }
            }
        }
        for (uint i = execRecursive1File.nSMap; i < NRecurive1; i++)
        {
            for (uint j = 0; j < 12; j++)
            {
                cmPolsRecursive1.Compressor.a[j][i] = Goldilocks::zero();
            }
        }
        delete[] tmpRecursive1;
        CircomRecursive1::freeCircuit(circuitRecursive1);

        TimerStopAndLog(WITNESS_AND_COMMITED_POLS_BATCH_PROOF_RECURSIVE_1);

        /*****************************************/
        /* Generate Recursive 1 proof            */
        /*****************************************/

        TimerStart(STARK_RECURSIVE_1_PROOF_BATCH_PROOF);
        uint64_t polBitsRecursive1 = starkRecursive1.starkInfo.starkStruct.steps[starkRecursive1.starkInfo.starkStruct.steps.size() - 1].nBits;
        cout << "polBitsRecursive1=" << polBitsRecursive1 << endl;
        FRIProof fproofRecursive1((1 << polBitsRecursive1), FIELD_EXTENSION, starkRecursive1.starkInfo.starkStruct.steps.size(), starkRecursive1.starkInfo.evMap.size(), starkRecursive1.starkInfo.nPublics);
        starkRecursive1.genProof(pAddressRecursive1, fproofRecursive1, publics);
        TimerStopAndLog(STARK_RECURSIVE_1_PROOF_BATCH_PROOF);

        // Save the proof & zkinproof
        nlohmann::ordered_json jProofRecursive1 = fproofRecursive1.proofs.proof2json();
        nlohmann::ordered_json zkinRecursive1 = proof2zkinStark(jProofRecursive1);
        zkinRecursive1["publics"] = publicStarkJson;
        ofstream ofzkinRecursive(config.starkZkInRecursive1);
        ofzkinRecursive << setw(4) << zkinRecursive1.dump() << endl;
        ofzkinRecursive.close();

        jProofRecursive1["publics"] = publicStarkJson;
        ofstream ofProofRecursive1(config.starkFileRecursive1);
        ofProofRecursive1 << setw(4) << jProofRecursive1.dump() << endl;
        ofProofRecursive1.close();

        pProverRequest->batchProofOutput = zkinRecursive1;

        // Save output to file
        if (config.saveOutputToFile)
        {
            json2file(pProverRequest->batchProofOutput, pProverRequest->filePrefix + "batch_proof_output.json");
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
        json2file(pProverRequest->aggregatedProofInput1, pProverRequest->filePrefix + "aggregated_proof_input_1.json");
        json2file(pProverRequest->aggregatedProofInput2, pProverRequest->filePrefix + "aggregated_proof_input_2.json");
    }

    // Input is pProverRequest->aggregatedProofInput1 and pProverRequest->aggregatedProofInput2 (of type json)

    // Output is pProverRequest->aggregatedProofOutput (of type json)

    pProverRequest->aggregatedProofOutput = getUUID();

    // Save output to file
    if (config.saveOutputToFile)
    {
        json2file(pProverRequest->aggregatedProofOutput, pProverRequest->filePrefix + "aggregated_proof_output.json");
    }

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
        json2file(pProverRequest->finalProofInput, pProverRequest->filePrefix + "final_proof_input.json");
    }

    // Input is pProverRequest->finalProofInput (of type json)

    // Output is pProverRequest->proof (of type Proof)

    // Save output to file
    if (config.saveOutputToFile)
    {
        // json2file(jsonProof, pProverRequest->filePrefix + "proof_gen_proof.json");
    }

    TimerStopAndLog(PROVER_FINAL_PROOF);
}