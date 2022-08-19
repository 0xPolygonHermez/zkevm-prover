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

using namespace std;

Prover::Prover(Goldilocks &fr,
               PoseidonGoldilocks &poseidon,
               const Config &config) : fr(fr),
                                       poseidon(poseidon),
                                       executor(fr, config, poseidon),
                                       stark(config),
                                       config(config)
{
    mpz_init(altBbn128r);
    mpz_set_str(altBbn128r, "21888242871839275222246405745257275088548364400416034343698204186575808495617", 10);

    try
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
    catch (std::exception &e)
    {
        cerr << "Error: Prover::Prover() got an exception: " << e.what() << '\n';
        exitProcess();
    }
}

Prover::~Prover()
{
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

    TimerStopAndLog(PROVER_PROCESS_BATCH);
}

void Prover::prove(ProverRequest *pProverRequest)
{
    TimerStart(PROVER_PROVE);
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

    // Save input to <timestamp>.input.json, after execution
    /*Database * pDatabase = executor.mainExecutor.pStateDB->getDatabase();
    if (pDatabase != NULL)
    {
        json inputJsonEx;
        pProverRequest->input.save(inputJsonEx, *pDatabase);
        json2file(inputJsonEx, pProverRequest->inputFileEx);
    }*/

    if (pProverRequest->result == ZKR_SUCCESS)
    {
        StarkInfo starkInfo(config);
        // Generate the first stark proof
        uint64_t polBits = starkInfo.starkStruct.steps[starkInfo.starkStruct.steps.size() - 1].nBits;
        FRIProof fproof((1 << polBits), FIELD_EXTENSION, starkInfo.starkStruct.steps.size(), starkInfo.evMap.size(), starkInfo.nPublics);
#if 0
        stark.genProof(pAddress, fproof);

        /****************/
        /* Proof 2 zkIn */
        /****************/
        TimerStart(SAVE_PUBLIC_JSON);
        json publicJson;
        publicJson[0] = fr.toString(cmPols.Main.FREE0[0]);
        publicJson[1] = fr.toString(cmPols.Main.FREE1[0]);
        publicJson[2] = fr.toString(cmPols.Main.FREE2[0]);
        publicJson[3] = fr.toString(cmPols.Main.FREE3[0]);
        publicJson[4] = fr.toString(cmPols.Main.FREE4[0]);
        publicJson[5] = fr.toString(cmPols.Main.FREE5[0]);
        publicJson[6] = fr.toString(cmPols.Main.FREE6[0]);
        publicJson[7] = fr.toString(cmPols.Main.FREE7[0]);

        json2file(publicJson, pProverRequest->publicFile);
        TimerStopAndLog(SAVE_PUBLIC_JSON);

        TimerStart(STARK_JSON_GENERATION);

#define zkinFile "zkevm.proof.zkin.json"
#define starkFile "zkevm.prove.json"
        nlohmann::ordered_json jProof = fproof.proofs.proof2json();
        jProof["publics"] = publicJson;
        ofstream ofstark(starkFile);
        ofstark << setw(4) << jProof.dump() << endl;
        ofstark.close();

        nlohmann::json zkin = proof2zkinStark(jProof);
        zkin["publics"] = publicJson;
        ofstream ofzkin(zkinFile);
        ofzkin << setw(4) << zkin.dump() << endl;
        ofzkin.close();
        TimerStopAndLog(STARK_JSON_GENERATION);
#else
        nlohmann::json zkin;

        TimerStart(PROVER_INJECT_ZKIN_JSON);
        zkin.clear();
        std::ifstream zkinStream("zkevm.proof.zkin.json");
        if (!zkinStream.good())
        {
            cerr << "Error: failed loading zkin.json file " << endl;
            exitProcess();
        }
        zkinStream >> zkin;
        zkinStream.close();
        TimerStopAndLog(PROVER_INJECT_ZKIN_JSON);
#endif

        /************/
        /* Verifier */
        /************/

        TimerStart(CIRCOM_LOAD_CIRCUIT);
        Circom_Circuit *circuit = loadCircuit(config.verifierFile);
        TimerStopAndLog(CIRCOM_LOAD_CIRCUIT);

        TimerStart(CIRCOM_LOAD_JSON);
        Circom_CalcWit *ctx = new Circom_CalcWit(circuit);
        loadJsonImpl(ctx, zkin);
        if (ctx->getRemaingInputsToBeSet() != 0)
        {
            cerr << "Error: Not all inputs have been set. Only " << get_main_input_signal_no() - ctx->getRemaingInputsToBeSet() << " out of " << get_main_input_signal_no() << endl;
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
        ExecFile execFile(config.execFile);
        uint64_t sizeWitness = get_size_of_witness();
        FrGElement *tmp = new FrGElement[execFile.nAdds + sizeWitness];

#pragma omp parallel for
        for (uint64_t i = 0; i < sizeWitness; i++)
        {
            ctx->getWitness(i, &tmp[i]);
        }

        for (uint64_t i = 0; i < execFile.nAdds; i++)
        {
            FrGElement tmp_1;
            FrGElement tmp_2;
            FrGElement tmp_3;
            FrGElement tmp_4;

            int idx_1 = FrG_toInt(&execFile.p_adds[i * 4]);
            int idx_2 = FrG_toInt(&execFile.p_adds[i * 4 + 1]);

            FrG_copy(&tmp_1, &tmp[idx_1]);
            FrG_copy(&tmp_2, &tmp[idx_2]);

            FrG_mul(&tmp_3, &tmp_1, &execFile.p_adds[i * 4 + 2]);
            FrG_mul(&tmp_4, &tmp_2, &execFile.p_adds[i * 4 + 3]);

            FrG_add(&tmp[sizeWitness + i], &tmp_3, &tmp_4);
        }

        uint64_t Nbits = log2(execFile.nSMap - 1) + 1;
        uint64_t N = 1 << Nbits;

        Config cfg;
        cfg.starkInfoFile = "zkevm.c12.starkinfo.json";
        cfg.constPolsFile = "zkevm.c12.const";
        cfg.mapConstPolsFile = false;
        cfg.constantsTreeFile = "zkevm.c12.consttree";
        StarkInfo starkInfoC12(cfg);
        StarkC12 starkC12(cfg);
        uint64_t polsSizeC12 = starkC12.getTotalPolsSize();

        void *pAddressC12 = calloc(polsSizeC12, 1);
        CommitPolsC12 cmPols12(pAddressC12, CommitPolsC12::pilDegree());

#pragma omp parallel for
        for (uint i = 0; i < execFile.nSMap; i++)
        {
            for (uint j = 0; j < 12; j++)
            {
                int idx_1 = FrG_toInt(&execFile.p_sMap[12 * i + j]);
                FrG_toLongNormal(&tmp[idx_1], &tmp[idx_1]);
                uint64_t val = tmp[idx_1].longVal[0];
                if (val != 0)
                {
                    cmPols12.Compressor.a[j][i] = Goldilocks::fromU64(val);
                }
            }
        }

        /*****************************************/
        /* Gen proof Compressor 12               */
        /*****************************************/

        void *pCommit = mapFile("zkevm.c12.commit", CommitPolsC12::pilSize(), false);
        std::memcpy(pAddressC12, pCommit, CommitPolsC12::pilSize());
        delete (tmp);

        uint64_t polBitsC12 = starkInfoC12.starkStruct.steps[starkInfoC12.starkStruct.steps.size() - 1].nBits;
        FRIProofC12 fproofC12((1 << polBitsC12), FIELD_EXTENSION, starkInfoC12.starkStruct.steps.size(), starkInfoC12.evMap.size(), starkInfoC12.nPublics);

        // Save public.json file
        TimerStart(SAVE_PUBLIC_JSON);
        json publicJson;

        mpz_t address;
        mpz_init_set_str(address, pProverRequest->input.aggregatorAddress.c_str(), 0);
        std::string strAddress = mpz_get_str(0, 16, address);

        std::string b = "";
        b = b + std::string(40 - std::min(40, (int)strAddress.length()), '0') + strAddress;

        for (uint i = 0; i < 8; i++)
        {
            std::string in = fr.toString(cmPols.Main.FREE0[i], 16);
            b = b + std::string(16 - std::min(16, (int)in.length()), '0') + in;
        }
        RawFr::Element publicsHash;

        mpz_t publicshash;
        mpz_init_set_str(publicshash, sha256(b).c_str(), 16);
        std::string publicsHashString = mpz_get_str(0, 10, publicshash);
        RawFr::field.fromString(publicsHash, publicsHashString);

        publicJson[0] = RawFr::field.toString(publicsHash, 10);
        json2file(publicJson, pProverRequest->publicFile);
        TimerStopAndLog(SAVE_PUBLIC_JSON);

        // Generate the proof
        starkC12.genProof(pAddressC12, fproofC12);
        nlohmann::ordered_json jProofC12 = fproofC12.proofs.proof2json();
        nlohmann::ordered_json zkinC12 = proof2zkinStark(jProofC12);
        // zkin["publics"] = publicJson;
        zkinC12["proverAddr"] = publicJson;
        ofstream ofzkin2("zkevm.c12.zkin.proof.json");
        ofzkin2 << setw(4) << zkinC12.dump() << endl;
        ofzkin2.close();

        /************/
        /* Verifier */
        /************/
        TimerStart(CIRCOM_LOAD_CIRCUIT_C12);
        Circom_CircuitC12 *circuitC12 = loadCircuitC12("zkevm.c12.verifier.dat");
        TimerStopAndLog(CIRCOM_LOAD_CIRCUIT_C12);

        TimerStart(CIRCOM_C12_LOAD_JSON);
        Circom_CalcWitC12 *ctxC12 = new Circom_CalcWitC12(circuitC12);
        json zkinC12json = json::parse(zkinC12.dump().c_str());

        loadJsonImplC12(ctxC12, zkinC12json);
        if (ctxC12->getRemaingInputsToBeSet() != 0)
        {
            cerr << "Error: Not all inputs have been set. Only " << get_main_input_signal_no() - ctxC12->getRemaingInputsToBeSet() << " out of " << get_main_input_signal_no() << endl;
            exitProcess();
        }
        TimerStopAndLog(CIRCOM_C12_LOAD_JSON);

        // If present, save witness file
        if (config.witnessFile.size() > 0)
        {
            TimerStart(CIRCOM_WRITE_BIN_WITNESS);
            writeBinWitnessC12(ctxC12, "zkevm.c12.witness.wtns"); // No need to write the file to disk, 12-13M fe, in binary, in wtns format
            TimerStopAndLog(CIRCOM_WRITE_BIN_WITNESS);
        }
        TimerStart(CIRCOM_GET_BIN_WITNESS);
        AltBn128::FrElement *pWitnessC12 = NULL;
        uint64_t witnessSize = 0;
        getBinWitnessC12(ctxC12, pWitnessC12, witnessSize);
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
        free(pAddressC12);
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

    TimerStopAndLog(PROVER_PROVE);
}
