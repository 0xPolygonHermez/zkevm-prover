#include <fstream>
#include <iomanip>
#include <unistd.h>
#include "prover.hpp"
#include "utils.hpp"
#include "scalar.hpp"
#include "proof2zkin.hpp"
#include "zkevm_verifier_cpp/main.hpp"
#include "zkevm_c12_verifier_cpp/main.c12.hpp"
#include "binfile_utils.hpp"
#include "zkey_utils.hpp"
#include "wtns_utils.hpp"
#include "groth16.hpp"
#include "sm/storage/storage_executor.hpp"
#include "timer.hpp"
#include "execFile.hpp"
#include <math.h> /* log2 */
#include "commit_pols_c12.hpp"
#include "starkC12.hpp"
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
                                       starkC12(config),
                                       config(config)
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
        }
        lastComputedRequestEndTime = 0;

        sem_init(&pendingRequestSem, 0, 0);
        pthread_mutex_init(&mutex, NULL);
        pCurrentRequest = NULL;
        pthread_create(&proverPthread, NULL, proverThread, this);
        pthread_create(&cleanerPthread, NULL, cleanerThread, this);
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
        pProver->prove(pProver->pCurrentRequest);

        // Move to completed requests
        pProver->lock();
        ProverRequest *pProverRequest = pProver->pCurrentRequest;
        pProverRequest->endTime = time(NULL);
        pProver->lastComputedRequestId = pProverRequest->uuid;
        pProver->lastComputedRequestEndTime = pProverRequest->endTime;

        pProver->completedRequests.push_back(pProver->pCurrentRequest);
        pProver->pCurrentRequest = NULL;
        pProver->unlock();

        cout << "proverThread() dome processing request with UUID: " << pProverRequest->uuid << endl;

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
    cout << "Prover::submitRequest() started" << endl;

    // Initialize the prover request
    pProverRequest->init(config);

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
    zkassert(uuid.size() > 0);
    cout << "Prover::waitForRequestToComplete() waiting for request with UUID: " << uuid << endl;

    // We will store here the address of the prove request corresponding to this UUID
    ProverRequest *pProverRequest = NULL;

    lock();

    // Map uuid to the corresponding prover request
    std::map<std::string, ProverRequest *>::iterator it = requestsMap.find(uuid);
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

    cout << "Prover::processBatch() timestamp: " << pProverRequest->timestamp << endl;
    cout << "Prover::processBatch() UUID: " << pProverRequest->uuid << endl;

    // Save input to <timestamp>.input.json, as provided by client
    json inputJson;
    pProverRequest->input.save(inputJson);
    json2file(inputJson, pProverRequest->inputFile);

    // Execute the program, in the process batch way
    pProverRequest->bProcessBatch = true;
    executor.process_batch(*pProverRequest);

    // Save input to <timestamp>.input.json after execution including dbReadLog
    Database * pDatabase = executor.mainExecutor.pStateDB->getDatabase();
    if (pDatabase != NULL)
    {
        json inputJsonEx;
        pProverRequest->input.save(inputJsonEx, *pDatabase);
        json2file(inputJsonEx, pProverRequest->inputFileEx);
    }

    TimerStopAndLog(PROVER_PROCESS_BATCH);
}

void Prover::prove(ProverRequest *pProverRequest)
{
    TimerStart(PROVER_PROVE);

    printMemoryInfo();
    printProcessInfo();

    zkassert(pProverRequest != NULL);

    cout << "Prover::prove() timestamp: " << pProverRequest->timestamp << endl;
    cout << "Prover::prove() UUID: " << pProverRequest->uuid << endl;
    cout << "Prover::prove() input file: " << pProverRequest->inputFile << endl;
    cout << "Prover::prove() public file: " << pProverRequest->publicFile << endl;
    cout << "Prover::prove() proof file: " << pProverRequest->proofFile << endl;

    // Save input to <timestamp>.input.json, as provided by client
    json inputJson;
    pProverRequest->input.save(inputJson);
    json2file(inputJson, pProverRequest->inputFile);

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
        cout << "Prover::prove() successfully mapped " << polsSize << " bytes to file " << config.cmPolsFile << endl;
    }
    else
    {
        pAddress = calloc(polsSize, 1);
        if (pAddress == NULL)
        {
            cerr << "Error: Prover::prove() failed calling malloc() of size " << polsSize << endl;
            exitProcess();
        }
        cout << "Prover::prove() successfully allocated " << polsSize << " bytes" << endl;
    }
    CommitPols cmPols(pAddress, CommitPols::pilDegree());

    // Execute all the State Machines
    TimerStart(EXECUTOR_EXECUTE);
    executor.execute(*pProverRequest, cmPols);
    TimerStopAndLog(EXECUTOR_EXECUTE);

    // Save input to <timestamp>.input.json after execution including dbReadLog
    Database * pDatabase = executor.mainExecutor.pStateDB->getDatabase();
    if (pDatabase != NULL)
    {
        json inputJsonEx;
        pProverRequest->input.save(inputJsonEx, *pDatabase);
        json2file(inputJsonEx, pProverRequest->inputFileEx);
    }

    if (pProverRequest->result == ZKR_SUCCESS)
    {
        /*************************************/
        /*  Generate publics input           */
        /*************************************/
        TimerStart(SAVE_PUBLICS_JSON);
        json publicJson;
        mpz_t address;
        mpz_t publicshash;
        json publicStarkJson;
        RawFr::Element publicsHash;
        string freeInStrings16[8];

        publicStarkJson[0] = fr.toString(cmPols.Main.FREE0[0]);
        publicStarkJson[1] = fr.toString(cmPols.Main.FREE1[0]);
        publicStarkJson[2] = fr.toString(cmPols.Main.FREE2[0]);
        publicStarkJson[3] = fr.toString(cmPols.Main.FREE3[0]);
        publicStarkJson[4] = fr.toString(cmPols.Main.FREE4[0]);
        publicStarkJson[5] = fr.toString(cmPols.Main.FREE5[0]);
        publicStarkJson[6] = fr.toString(cmPols.Main.FREE6[0]);
        publicStarkJson[7] = fr.toString(cmPols.Main.FREE7[0]);

        freeInStrings16[0] = fr.toString(cmPols.Main.FREE0[0], 16);
        freeInStrings16[1] = fr.toString(cmPols.Main.FREE1[0], 16);
        freeInStrings16[2] = fr.toString(cmPols.Main.FREE2[0], 16);
        freeInStrings16[3] = fr.toString(cmPols.Main.FREE3[0], 16);
        freeInStrings16[4] = fr.toString(cmPols.Main.FREE4[0], 16);
        freeInStrings16[5] = fr.toString(cmPols.Main.FREE5[0], 16);
        freeInStrings16[6] = fr.toString(cmPols.Main.FREE6[0], 16);
        freeInStrings16[7] = fr.toString(cmPols.Main.FREE7[0], 16);

        mpz_init_set_str(address, pProverRequest->input.publicInputs.aggregatorAddress.c_str(), 0);
        std::string strAddress = mpz_get_str(0, 16, address);
        std::string strAddress10 = mpz_get_str(0, 10, address);

        std::string buffer = "";
        buffer = buffer + std::string(40 - std::min(40, (int)strAddress.length()), '0') + strAddress;

        std::string aux;
        for (uint i = 0; i < 8; i++)
        {
            buffer = buffer + std::string(16 - std::min(16, (int)freeInStrings16[i].length()), '0') + freeInStrings16[i];
        }

        mpz_init_set_str(publicshash, sha256(buffer).c_str(), 16);
        std::string publicsHashString = mpz_get_str(0, 10, publicshash);
        RawFr::field.fromString(publicsHash, publicsHashString);

        // Save public file
        publicJson[0] = RawFr::field.toString(publicsHash, 10);
        json2file(publicJson, pProverRequest->publicFile);
        json2file(publicStarkJson, config.publicStarkFile);
        TimerStopAndLog(SAVE_PUBLICS_JSON);

        /*************************************/
        /*  Generate  stark proof            */
        /*************************************/
        TimerStart(STARK_PROOF);
        uint64_t polBits = stark.starkInfo.starkStruct.steps[stark.starkInfo.starkStruct.steps.size() - 1].nBits;
        FRIProof fproof((1 << polBits), FIELD_EXTENSION, stark.starkInfo.starkStruct.steps.size(), stark.starkInfo.evMap.size(), stark.starkInfo.nPublics);
        stark.genProof(pAddress, fproof);
        TimerStopAndLog(STARK_PROOF);

        TimerStart(STARK_JSON_GENERATION);

        nlohmann::ordered_json jProof = fproof.proofs.proof2json();
        jProof["publics"] = publicStarkJson;
        ofstream ofstark(config.starkFile);
        ofstark << setw(4) << jProof.dump() << endl;
        ofstark.close();

        nlohmann::json zkin = proof2zkinStark(jProof);
        zkin["publics"] = publicStarkJson;
        ofstream ofzkin(config.starkZkIn);
        ofzkin << setw(4) << zkin.dump() << endl;
        ofzkin.close();

        TimerStopAndLog(STARK_JSON_GENERATION);

        /************/
        /* Verifier */
        /************/

        TimerStart(CIRCOM_LOAD_CIRCUIT);
        Circom::Circom_Circuit *circuit = Circom::loadCircuit(config.verifierFile);
        TimerStopAndLog(CIRCOM_LOAD_CIRCUIT);

        TimerStart(CIRCOM_LOAD_JSON);
        Circom::Circom_CalcWit *ctx = new Circom::Circom_CalcWit(circuit);
        loadJsonImpl(ctx, zkin);
        if (ctx->getRemaingInputsToBeSet() != 0)
        {
            cerr << "Error: Not all inputs have been set. Only " << Circom::get_main_input_signal_no() - ctx->getRemaingInputsToBeSet() << " out of " << Circom::get_main_input_signal_no() << endl;
            exitProcess();
        }
        TimerStopAndLog(CIRCOM_LOAD_JSON);

        // If present, save witness file
        if (config.witnessFile.size() > 0)
        {
            TimerStart(CIRCOM_WRITE_BIN_WITNESS);
            writeBinWitness(ctx, config.witnessFile); // No need to write the file to disk, 12-13M fe, in binary, in wtns format
            TimerStopAndLog(CIRCOM_WRITE_BIN_WITNESS);
        }

        /*****************************************/
        /* Compute witness and c12 commited pols */
        /*****************************************/
        TimerStart(C12_WITNESS_AND_COMMITED_POLS);

        ExecFile execFile(config.execFile);
        uint64_t sizeWitness = Circom::get_size_of_witness();
        Goldilocks::Element *tmp = new Goldilocks::Element[execFile.nAdds + sizeWitness];

#pragma omp parallel for
        for (uint64_t i = 0; i < sizeWitness; i++)
        {
            FrGElement aux;
            ctx->getWitness(i, &aux);
            FrG_toLongNormal(&aux, &aux);
            tmp[i] = Goldilocks::fromU64(aux.longVal[0]);
        }

        for (uint64_t i = 0; i < execFile.nAdds; i++)
        {
            FrG_toLongNormal(&execFile.p_adds[i * 4], &execFile.p_adds[i * 4]);
            FrG_toLongNormal(&execFile.p_adds[i * 4 + 1], &execFile.p_adds[i * 4 + 1]);
            FrG_toLongNormal(&execFile.p_adds[i * 4 + 2], &execFile.p_adds[i * 4 + 2]);
            FrG_toLongNormal(&execFile.p_adds[i * 4 + 3], &execFile.p_adds[i * 4 + 3]);

            uint64_t idx_1 = execFile.p_adds[i * 4].longVal[0];
            uint64_t idx_2 = execFile.p_adds[i * 4 + 1].longVal[0];

            Goldilocks::Element c = tmp[idx_1] * Goldilocks::fromU64(execFile.p_adds[i * 4 + 2].longVal[0]);
            Goldilocks::Element d = tmp[idx_2] * Goldilocks::fromU64(execFile.p_adds[i * 4 + 3].longVal[0]);
            tmp[sizeWitness + i] = c + d;
        }

        uint64_t Nbits = log2(execFile.nSMap - 1) + 1;
        uint64_t N = 1 << Nbits;

        uint64_t polsSizeC12 = starkC12.getTotalPolsSize();
        cout << "starkC12.getTotalPolsSize()=" << polsSizeC12 << endl;

        //void *pAddressC12 = calloc(polsSizeC12, 1);
        void *pAddressC12 = pAddress;
        CommitPolsC12 cmPols12(pAddressC12, CommitPolsC12::pilDegree());

#pragma omp parallel for
        for (uint i = 0; i < execFile.nSMap; i++)
        {
            for (uint j = 0; j < 12; j++)
            {
                FrGElement aux;
                FrG_toLongNormal(&aux, &execFile.p_sMap[12 * i + j]);
                uint64_t idx_1 = aux.longVal[0];
                if (idx_1 != 0)
                {
                    cmPols12.Compressor.a[j][i] = tmp[idx_1];
                }
                else
                {
                    cmPols12.Compressor.a[j][i] = Goldilocks::zero();
                }
            }
        }
        for (uint i = execFile.nSMap; i < N; i++)
        {
            for (uint j = 0; j < 12; j++)
            {
                cmPols12.Compressor.a[j][i] = Goldilocks::zero();
            }
        }
        delete[] tmp;
        TimerStopAndLog(C12_WITNESS_AND_COMMITED_POLS);

        /*****************************************/
        /* Generate C12 stark proof              */
        /*****************************************/
        TimerStart(STARK_C12_PROOF);
        uint64_t polBitsC12 = starkC12.starkInfo.starkStruct.steps[starkC12.starkInfo.starkStruct.steps.size() - 1].nBits;
        cout << "polBitsC12=" << polBitsC12 << endl;
        FRIProofC12 fproofC12((1 << polBitsC12), FIELD_EXTENSION, starkC12.starkInfo.starkStruct.steps.size(), starkC12.starkInfo.evMap.size(), starkC12.starkInfo.nPublics);

        Goldilocks::Element publics[8];
        publics[0] = cmPols.Main.FREE0[0];
        publics[1] = cmPols.Main.FREE1[0];
        publics[2] = cmPols.Main.FREE2[0];
        publics[3] = cmPols.Main.FREE3[0];
        publics[4] = cmPols.Main.FREE4[0];
        publics[5] = cmPols.Main.FREE5[0];
        publics[6] = cmPols.Main.FREE6[0];
        publics[7] = cmPols.Main.FREE7[0];

        // Generate the proof
        starkC12.genProof(pAddressC12, fproofC12, publics);
        TimerStopAndLog(STARK_C12_PROOF);

        nlohmann::ordered_json jProofC12 = fproofC12.proofs.proof2json();
        nlohmann::ordered_json zkinC12 = proof2zkinStark(jProofC12);
        zkinC12["publics"] = publicStarkJson;
        zkinC12["proverAddr"] = strAddress10;
        ofstream ofzkin2(config.starkZkInC12);
        ofzkin2 << setw(4) << zkinC12.dump() << endl;
        ofzkin2.close();

        /************/
        /* Verifier */
        /************/
        TimerStart(CIRCOM_LOAD_CIRCUIT_C12);
        CircomC12::Circom_Circuit *circuitC12 = CircomC12::loadCircuit(config.verifierFileC12);
        TimerStopAndLog(CIRCOM_LOAD_CIRCUIT_C12);

        TimerStart(CIRCOM_C12_LOAD_JSON);
        CircomC12::Circom_CalcWit *ctxC12 = new CircomC12::Circom_CalcWit(circuitC12);
        json zkinC12json = json::parse(zkinC12.dump().c_str());

        CircomC12::loadJsonImpl(ctxC12, zkinC12json);
        if (ctxC12->getRemaingInputsToBeSet() != 0)
        {
            cerr << "Error: Not all inputs have been set. Only " << Circom::get_main_input_signal_no() - ctxC12->getRemaingInputsToBeSet() << " out of " << Circom::get_main_input_signal_no() << endl;
            exitProcess();
        }
        TimerStopAndLog(CIRCOM_C12_LOAD_JSON);

        // If present, save witness file
        if (config.witnessFileC12.size() > 0)
        {
            TimerStart(CIRCOM_WRITE_BIN_WITNESS);
            CircomC12::writeBinWitness(ctxC12, config.witnessFileC12); // No need to write the file to disk, 12-13M fe, in binary, in wtns format
            TimerStopAndLog(CIRCOM_WRITE_BIN_WITNESS);
        }
        TimerStart(CIRCOM_GET_BIN_WITNESS);
        AltBn128::FrElement *pWitnessC12 = NULL;
        uint64_t witnessSize = 0;
        CircomC12::getBinWitness(ctxC12, pWitnessC12, witnessSize);
        TimerStopAndLog(CIRCOM_GET_BIN_WITNESS);

        // Generate Groth16 via rapid SNARK
        TimerStart(RAPID_SNARK);
        json jsonProof;
        try
        {
            auto proof = groth16Prover->prove(pWitnessC12);
            jsonProof = proof->toJson();
        }
        catch (std::exception &e)
        {
            cerr << "Error: Prover::Prove() got exception in rapid SNARK:" << e.what() << '\n';
            exitProcess();
        }
        TimerStopAndLog(RAPID_SNARK);

        // Save proof.json to disk 
        json2file(jsonProof, pProverRequest->proofFile);

        // Populate Proof with the correct data
        PublicInputsExtended publicInputsExtended;
        publicInputsExtended.publicInputs = pProverRequest->input.publicInputs;
        publicInputsExtended.inputHash = NormalizeTo0xNFormat(fr.toString(cmPols.Main.FREE0[0], 16), 64);
        pProverRequest->proof.load(jsonProof, publicInputsExtended);

        /***********/
        /* Cleanup */
        /***********/
        delete ctx;
        delete ctxC12;
        Circom::freeCircuit(circuit);
        CircomC12::freeCircuit(circuitC12);

        //free(pAddressC12);
        free(pWitnessC12);
    }

    // Unmap committed polynomials address
    if (config.cmPolsFile.size() > 0)
    {
        unmapFile(pAddress, polsSize);
    }
    else
    {
        free(pAddress);
    }

    // cout << "Prover::prove() done" << endl;

    // printMemoryInfo();
    // printProcessInfo();

    TimerStopAndLog(PROVER_PROVE);
}
