#include <fstream>
#include <iomanip>
#include <unistd.h>
#include "prover.hpp"
#include "utils.hpp"
#include "scalar.hpp"
#include "proof2zkin.hpp"
#include "zkevm_verifier_cpp/main.hpp"
#include "zkevm_c12a_verifier_cpp/main.c12a.hpp"
#include "zkevm_c12b_verifier_cpp/main.c12b.hpp"
#include "binfile_utils.hpp"
#include "zkey_utils.hpp"
#include "wtns_utils.hpp"
#include "groth16.hpp"
#include "sm/storage/storage_executor.hpp"
#include "timer.hpp"
#include "execFile.hpp"
#include <math.h> /* log2 */
#include "commit_pols_c12a.hpp"
#include "commit_pols_c12b.hpp"
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
                                       starkC12b(config),
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
        case prt_genProof:
            pProver->genProof(pProver->pCurrentRequest);
            break;
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
    zkassert(config.generateProof());
    zkassert(pProverRequest != NULL);
    zkassert(pProverRequest->type == prt_genProof);

    TimerStart(PROVER_GEN_PROOF);
    
    printMemoryInfo(true);
    printProcessInfo(true);

    zkassert(pProverRequest != NULL);

    cout << "Prover::genProof() timestamp: " << pProverRequest->timestamp << endl;
    cout << "Prover::genProof() UUID: " << pProverRequest->uuid << endl;
    cout << "Prover::genProof() input file: " << pProverRequest->inputFile << endl;
    cout << "Prover::genProof() public file: " << pProverRequest->publicFile << endl;
    cout << "Prover::genProof() proof file: " << pProverRequest->proofFile << endl;

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
        cout << "Prover::genProof() successfully mapped " << polsSize << " bytes to file " << config.cmPolsFile << endl;
    }
    else
    {
        pAddress = calloc(polsSize, 1);
        if (pAddress == NULL)
        {
            cerr << "Error: Prover::genProof() failed calling malloc() of size " << polsSize << endl;
            exitProcess();
        }
        cout << "Prover::genProof() successfully allocated " << polsSize << " bytes" << endl;
    }

    CommitPols cmPols(pAddress, CommitPols::pilDegree());

    // Execute all the State Machines
    TimerStart(EXECUTOR_EXECUTE);
    executor.execute(*pProverRequest, cmPols);
    TimerStopAndLog(EXECUTOR_EXECUTE);

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

        mpz_init_set_str(address, pProverRequest->input.publicInputsExtended.publicInputs.aggregatorAddress.c_str(), 0);
        std::string strAddress = mpz_get_str(0, 16, address);
        std::string strAddress10 = mpz_get_str(0, 10, address);
        mpz_clear(address);

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
        mpz_clear(publicshash);

        // Save public file
        publicJson[0] = RawFr::field.toString(publicsHash, 10);
        json2file(publicJson, pProverRequest->publicFile);
        json2file(publicStarkJson, config.publicStarkFile);
        TimerStopAndLog(SAVE_PUBLICS_JSON);

        /*************************************/
        /*  Generate stark proof            */
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

        /************************/
        /* Verifier stark proof */
        /************************/

        TimerStart(CIRCOM_LOAD_CIRCUIT);
        Circom::Circom_Circuit *circuit = Circom::loadCircuit(config.verifierFile);
        TimerStopAndLog(CIRCOM_LOAD_CIRCUIT);

        TimerStart(CIRCOM_LOAD_JSON);
        Circom::Circom_CalcWit *ctx = new Circom::Circom_CalcWit(circuit);

        loadJsonImpl(ctx, zkin);
        if (ctx->getRemaingInputsToBeSet() != 0)
        {
            cerr << "Error: Prover::genProof() Not all inputs have been set. Only " << Circom::get_main_input_signal_no() - ctx->getRemaingInputsToBeSet() << " out of " << Circom::get_main_input_signal_no() << endl;
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

        /******************************************/
        /* Compute witness and c12a commited pols */
        /******************************************/
        TimerStart(C12_A_WITNESS_AND_COMMITED_POLS);

        ExecFile execC12aFile(config.execC12aFile);
        uint64_t sizeWitness = Circom::get_size_of_witness();
        Goldilocks::Element *tmp = new Goldilocks::Element[execC12aFile.nAdds + sizeWitness];

#pragma omp parallel for
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
        cout << "Prover::genProof() starkC12a.getTotalPolsSize()=" << polsSizeC12 << endl;

        // void *pAddressC12 = calloc(polsSizeC12, 1);
        void *pAddressC12 = pAddress;
        CommitPolsC12a cmPols12a(pAddressC12, CommitPolsC12a::pilDegree());

#pragma omp parallel for
        for (uint i = 0; i < execC12aFile.nSMap; i++)
        {
            for (uint j = 0; j < 12; j++)
            {
                FrGElement aux;
                FrG_toLongNormal(&aux, &execC12aFile.p_sMap[12 * i + j]);
                uint64_t idx_1 = aux.longVal[0];
                if (idx_1 != 0)
                {
                    cmPols12a.Compressor.a[j][i] = tmp[idx_1];
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
        TimerStopAndLog(C12_A_WITNESS_AND_COMMITED_POLS);

        if (config.cmPolsFileC12a.size() > 0)
        {
            void *pAddressC12tmp = mapFile(config.cmPolsFileC12a, CommitPolsC12a::pilSize(), true);
            cout << "Prover::genProof() successfully mapped " << CommitPolsC12a::pilSize() << " bytes to file "
                 << config.cmPolsFileC12a << endl;
            std::memcpy(pAddressC12tmp, pAddressC12, CommitPolsC12a::pilSize());
            unmapFile(pAddressC12tmp, CommitPolsC12a::pilSize());
        }

        /*****************************************/
        /* Generate C12a stark proof             */
        /*****************************************/
        TimerStart(STARK_C12_A_PROOF);
        uint64_t polBitsC12 = starkC12a.starkInfo.starkStruct.steps[starkC12a.starkInfo.starkStruct.steps.size() - 1].nBits;
        cout << "polBitsC12=" << polBitsC12 << endl;
        FRIProof fproofC12a((1 << polBitsC12), FIELD_EXTENSION, starkC12a.starkInfo.starkStruct.steps.size(), starkC12a.starkInfo.evMap.size(), starkC12a.starkInfo.nPublics);

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
        starkC12a.genProof(pAddressC12, fproofC12a, publics);
        TimerStopAndLog(STARK_C12_A_PROOF);

        // Save the proof & zkinproof
        nlohmann::ordered_json jProofc12a = fproofC12a.proofs.proof2json();
        nlohmann::ordered_json zkinC12a = proof2zkinStark(jProofc12a);
        zkinC12a["publics"] = publicStarkJson;
        ofstream ofzkin2c12a(config.starkZkInC12a);
        ofzkin2c12a << setw(4) << zkinC12a.dump() << endl;
        ofzkin2c12a.close();

        jProofc12a["publics"] = publicStarkJson;
        ofstream ofstarkc12a(config.starkFilec12a);
        ofstarkc12a << setw(4) << jProofc12a.dump() << endl;
        ofstarkc12a.close();

        /*****************/
        /* Verifier C12a */
        /*****************/
        TimerStart(CIRCOM_LOAD_CIRCUIT_C12_A);
        CircomC12a::Circom_Circuit *circuitC12a = CircomC12a::loadCircuit(config.verifierFileC12a);
        TimerStopAndLog(CIRCOM_LOAD_CIRCUIT_C12_A);

        TimerStart(CIRCOM_C12_A_LOAD_JSON);
        CircomC12a::Circom_CalcWit *ctxC12a = new CircomC12a::Circom_CalcWit(circuitC12a);
        json zkinC12ajson = json::parse(zkinC12a.dump().c_str());

        CircomC12a::loadJsonImpl(ctxC12a, zkinC12ajson);
        if (ctxC12a->getRemaingInputsToBeSet() != 0)
        {
            cerr << "Error: Prover::genProof() Not all inputs have been set. Only " << CircomC12a::get_main_input_signal_no() - ctxC12a->getRemaingInputsToBeSet() << " out of " << CircomC12a::get_main_input_signal_no() << endl;
            exitProcess();
        }
        TimerStopAndLog(CIRCOM_C12_A_LOAD_JSON);

        // If present, save witness file
        if (config.witnessFileC12a.size() > 0)
        {
            TimerStart(CIRCOM_WRITE_BIN_WITNESS_C12_A);
            CircomC12a::writeBinWitness(ctxC12a, config.witnessFileC12a); // No need to write the file to disk, 12-13M fe, in binary, in wtns format
            TimerStopAndLog(CIRCOM_WRITE_BIN_WITNESS_C12_A);
        }

        /******************************************/
        /* Compute witness and C12b commited pols */
        /******************************************/
        TimerStart(C12_B_WITNESS_AND_COMMITED_POLS);

        ExecFile execC12bFile(config.execC12bFile);
        uint64_t sizeWitnessc12a = CircomC12a::get_size_of_witness();
        Goldilocks::Element *tmpc12a = new Goldilocks::Element[execC12bFile.nAdds + sizeWitnessc12a];

#pragma omp parallel for
        for (uint64_t i = 0; i < sizeWitnessc12a; i++)
        {
            FrGElement aux;
            ctxC12a->getWitness(i, &aux);
            FrG_toLongNormal(&aux, &aux);
            tmpc12a[i] = Goldilocks::fromU64(aux.longVal[0]);
        }
        delete ctxC12a;

        for (uint64_t i = 0; i < execC12bFile.nAdds; i++)
        {
            FrG_toLongNormal(&execC12bFile.p_adds[i * 4], &execC12bFile.p_adds[i * 4]);
            FrG_toLongNormal(&execC12bFile.p_adds[i * 4 + 1], &execC12bFile.p_adds[i * 4 + 1]);
            FrG_toLongNormal(&execC12bFile.p_adds[i * 4 + 2], &execC12bFile.p_adds[i * 4 + 2]);
            FrG_toLongNormal(&execC12bFile.p_adds[i * 4 + 3], &execC12bFile.p_adds[i * 4 + 3]);

            uint64_t idx_1 = execC12bFile.p_adds[i * 4].longVal[0];
            uint64_t idx_2 = execC12bFile.p_adds[i * 4 + 1].longVal[0];

            Goldilocks::Element c = tmpc12a[idx_1] * Goldilocks::fromU64(execC12bFile.p_adds[i * 4 + 2].longVal[0]);
            Goldilocks::Element d = tmpc12a[idx_2] * Goldilocks::fromU64(execC12bFile.p_adds[i * 4 + 3].longVal[0]);
            tmpc12a[sizeWitnessc12a + i] = c + d;
        }

        uint64_t NbitsC12a = log2(execC12bFile.nSMap - 1) + 1;
        uint64_t NC12a = 1 << NbitsC12a;

        // void *pAddressC12b = calloc(polsSizeC12b, 1);
        void *pAddressC12b = pAddress;

        CommitPolsC12b cmPols12b(pAddressC12b, CommitPolsC12b::pilDegree());

#pragma omp parallel for
        for (uint i = 0; i < execC12bFile.nSMap; i++)
        {
            for (uint j = 0; j < 12; j++)
            {
                FrGElement aux;
                FrG_toLongNormal(&aux, &execC12bFile.p_sMap[12 * i + j]);
                uint64_t idx_1 = aux.longVal[0];
                if (idx_1 != 0)
                {
                    cmPols12b.Compressor.a[j][i] = tmpc12a[idx_1];
                }
                else
                {
                    cmPols12b.Compressor.a[j][i] = Goldilocks::zero();
                }
            }
        }
        for (uint i = execC12bFile.nSMap; i < NC12a; i++)
        {
            for (uint j = 0; j < 12; j++)
            {
                cmPols12b.Compressor.a[j][i] = Goldilocks::zero();
            }
        }
        CircomC12a::freeCircuit(circuitC12a);
        delete[] tmpc12a;
        TimerStopAndLog(C12_B_WITNESS_AND_COMMITED_POLS);

        if (config.cmPolsFileC12b.size() > 0)
        {
            void *pAddressC12btmp = mapFile(config.cmPolsFileC12b, CommitPolsC12b::pilSize(), true);
            cout << "Prover::genProof() successfully mapped " << CommitPolsC12b::pilSize() << " bytes to file "
                 << config.cmPolsFileC12b << endl;
            std::memcpy(pAddressC12btmp, pAddressC12b, CommitPolsC12b::pilSize());
            unmapFile(pAddressC12btmp, CommitPolsC12b::pilSize());
        }

        /*****************************************/
        /* Generate C12b stark proof              */
        /*****************************************/

        TimerStart(STARK_C12_B_PROOF);
        uint64_t polBitsC12b = starkC12b.starkInfo.starkStruct.steps[starkC12b.starkInfo.starkStruct.steps.size() - 1].nBits;
        FRIProofC12 fproof_c12b((1 << polBitsC12b), FIELD_EXTENSION, starkC12b.starkInfo.starkStruct.steps.size(), starkC12b.starkInfo.evMap.size(), starkC12b.starkInfo.nPublics);

        // Generate the proof
        starkC12b.genProof(pAddressC12b, fproof_c12b, publics);
        TimerStopAndLog(STARK_C12_B_PROOF);

        nlohmann::ordered_json jProofC12b = fproof_c12b.proofs.proof2json();
        nlohmann::ordered_json zkinC12b = proof2zkinStark(jProofC12b);
        zkinC12b["publics"] = publicStarkJson;
        zkinC12b["proverAddr"] = strAddress10;
        ofstream ofzkin2b(config.starkZkInC12b);
        ofzkin2b << setw(4) << zkinC12b.dump() << endl;
        ofzkin2b.close();

        /*****************/
        /* Verifier c12b */
        /*****************/
        TimerStart(CIRCOM_LOAD_CIRCUIT_C12_B);
        CircomC12b::Circom_Circuit *circuitC12b = CircomC12b::loadCircuit(config.verifierFileC12b);
        TimerStopAndLog(CIRCOM_LOAD_CIRCUIT_C12_B);

        TimerStart(CIRCOM_C12_B_LOAD_JSON);
        CircomC12b::Circom_CalcWit *ctxC12b = new CircomC12b::Circom_CalcWit(circuitC12b);

        json zkinC12bjson = json::parse(zkinC12b.dump().c_str());

        CircomC12b::loadJsonImpl(ctxC12b, zkinC12bjson);
        if (ctxC12b->getRemaingInputsToBeSet() != 0)
        {
            cerr << "Error: Prover::genProof() Not all inputs have been set. Only " << CircomC12b::get_main_input_signal_no() - ctxC12b->getRemaingInputsToBeSet() << " out of " << CircomC12b::get_main_input_signal_no() << endl;
            exitProcess();
        }
        TimerStopAndLog(CIRCOM_C12_B_LOAD_JSON);

        // If present, save witness file
        if (config.witnessFileC12b.size() > 0)
        {
            TimerStart(CIRCOM_WRITE_BIN_WITNESS_C12_B);
            CircomC12b::writeBinWitness(ctxC12b, config.witnessFileC12b); // No need to write the file to disk, 12-13M fe, in binary, in wtns format
            TimerStopAndLog(CIRCOM_WRITE_BIN_WITNESS_C12_B);
        }
        TimerStart(CIRCOM_GET_BIN_WITNESS_C12_B);
        AltBn128::FrElement *pWitnessC12b = NULL;
        uint64_t witnessSizeb = 0;
        CircomC12b::getBinWitness(ctxC12b, pWitnessC12b, witnessSizeb);
        CircomC12b::freeCircuit(circuitC12b);
        delete ctxC12b;

        TimerStopAndLog(CIRCOM_GET_BIN_WITNESS_C12_B);

        // Generate Groth16 via rapid SNARK
        TimerStart(RAPID_SNARK);
        json jsonProof;
        try
        {
            auto proof = groth16Prover->prove(pWitnessC12b);
            jsonProof = proof->toJson();
        }
        catch (std::exception &e)
        {
            cerr << "Error: Prover::genProof() got exception in rapid SNARK:" << e.what() << '\n';
            exitProcess();
        }
        TimerStopAndLog(RAPID_SNARK);

        // Save proof.json to disk
        json2file(jsonProof, pProverRequest->proofFile);

        // Populate Proof with the correct data
        PublicInputsExtended publicInputsExtended;
        publicInputsExtended.publicInputs = pProverRequest->input.publicInputsExtended.publicInputs;
        publicInputsExtended.inputHash = NormalizeTo0xNFormat(fr.toString(cmPols.Main.FREE0[0], 16), 64);
        pProverRequest->proof.load(jsonProof, publicInputsExtended);

        /***********/
        /* Cleanup */
        /***********/
        free(pWitnessC12b);

        // Save output to file
        if (config.saveOutputToFile)
        {
            json2file(jsonProof, pProverRequest->filePrefix + "proof_gen_proof.json");
        }
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

    // printMemoryInfo();
    // printProcessInfo();

    TimerStopAndLog(PROVER_GEN_PROOF);
}

void Prover::genBatchProof(ProverRequest *pProverRequest)
{
    zkassert(config.generateProof());
    zkassert(pProverRequest != NULL);
    zkassert(pProverRequest->type == prt_genBatchProof);

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
    TimerStart(EXECUTOR_EXECUTE);
    executor.execute(*pProverRequest, cmPols);
    TimerStopAndLog(EXECUTOR_EXECUTE);

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

        mpz_init_set_str(address, pProverRequest->input.publicInputsExtended.publicInputs.aggregatorAddress.c_str(), 0);
        std::string strAddress = mpz_get_str(0, 16, address);
        std::string strAddress10 = mpz_get_str(0, 10, address);
        mpz_clear(address);

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
        mpz_clear(publicshash);

        // Save public file
        publicJson[0] = RawFr::field.toString(publicsHash, 10);
        json2file(publicJson, pProverRequest->publicFile);
        json2file(publicStarkJson, config.publicStarkFile);
        TimerStopAndLog(SAVE_PUBLICS_JSON);

        /*************************************/
        /*  Generate stark proof            */
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

        /************************/
        /* Verifier stark proof */
        /************************/

        TimerStart(CIRCOM_LOAD_CIRCUIT);
        Circom::Circom_Circuit *circuit = Circom::loadCircuit(config.verifierFile);
        TimerStopAndLog(CIRCOM_LOAD_CIRCUIT);

        TimerStart(CIRCOM_LOAD_JSON);
        Circom::Circom_CalcWit *ctx = new Circom::Circom_CalcWit(circuit);

        loadJsonImpl(ctx, zkin);
        if (ctx->getRemaingInputsToBeSet() != 0)
        {
            cerr << "Error: Prover::genBatchProof() Not all inputs have been set. Only " << Circom::get_main_input_signal_no() - ctx->getRemaingInputsToBeSet() << " out of " << Circom::get_main_input_signal_no() << endl;
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

        /******************************************/
        /* Compute witness and c12a commited pols */
        /******************************************/
        TimerStart(C12_A_WITNESS_AND_COMMITED_POLS);

        ExecFile execC12aFile(config.execC12aFile);
        uint64_t sizeWitness = Circom::get_size_of_witness();
        Goldilocks::Element *tmp = new Goldilocks::Element[execC12aFile.nAdds + sizeWitness];

#pragma omp parallel for
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

#pragma omp parallel for
        for (uint i = 0; i < execC12aFile.nSMap; i++)
        {
            for (uint j = 0; j < 12; j++)
            {
                FrGElement aux;
                FrG_toLongNormal(&aux, &execC12aFile.p_sMap[12 * i + j]);
                uint64_t idx_1 = aux.longVal[0];
                if (idx_1 != 0)
                {
                    cmPols12a.Compressor.a[j][i] = tmp[idx_1];
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
        TimerStopAndLog(C12_A_WITNESS_AND_COMMITED_POLS);

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
        TimerStart(STARK_C12_A_PROOF);
        uint64_t polBitsC12 = starkC12a.starkInfo.starkStruct.steps[starkC12a.starkInfo.starkStruct.steps.size() - 1].nBits;
        cout << "polBitsC12=" << polBitsC12 << endl;
        FRIProof fproofC12a((1 << polBitsC12), FIELD_EXTENSION, starkC12a.starkInfo.starkStruct.steps.size(), starkC12a.starkInfo.evMap.size(), starkC12a.starkInfo.nPublics);

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
        starkC12a.genProof(pAddressC12, fproofC12a, publics);
        TimerStopAndLog(STARK_C12_A_PROOF);

        // Save the proof & zkinproof
        nlohmann::ordered_json jProofc12a = fproofC12a.proofs.proof2json();
        nlohmann::ordered_json zkinC12a = proof2zkinStark(jProofc12a);
        zkinC12a["publics"] = publicStarkJson;
        ofstream ofzkin2c12a(config.starkZkInC12a);
        ofzkin2c12a << setw(4) << zkinC12a.dump() << endl;
        ofzkin2c12a.close();

        jProofc12a["publics"] = publicStarkJson;
        ofstream ofstarkc12a(config.starkFilec12a);
        ofstarkc12a << setw(4) << jProofc12a.dump() << endl;
        ofstarkc12a.close();

        pProverRequest->batchProofOutput = zkinC12a;

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
        //json2file(jsonProof, pProverRequest->filePrefix + "proof_gen_proof.json");
    }

    TimerStopAndLog(PROVER_FINAL_PROOF);
}