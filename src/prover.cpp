#include <fstream>
#include <iomanip>
#include <unistd.h>
#include "prover.hpp"
#include "utils.hpp"
#include "scalar.hpp"
#include "mem.hpp"
#include "batchmachine_executor.hpp"
#include "proof2zkin.hpp"
#include "verifier_cpp/main.hpp"
#include "binfile_utils.hpp"
#include "zkey_utils.hpp"
#include "wtns_utils.hpp"
#include "groth16.hpp"
#include "sm/storage/storage_executor.hpp"

using namespace std;

Prover::Prover( FiniteField &fr,
            Poseidon_goldilocks &poseidon,
            const Rom &romData,
            const Script &script,
            const Pil &pil,
            const Pols &constPols,
            const Config &config ) :
        fr(fr),
        poseidon(poseidon),
        romData(romData),
        executor(fr, poseidon, romData, config),
        storageExecutor(fr, poseidon, config),
        memoryExecutor(fr, config),
        binaryExecutor(fr, config),
        arithExecutor(fr, config),
        paddingKKExecutor(fr),
        nine2OneExecutor(fr),
        keccakFExecutor(config),
        script(script),
        pil(pil),
        constPols(constPols),
        config(config)
{
    mpz_init(altBbn128r);
    mpz_set_str(altBbn128r, "21888242871839275222246405745257275088548364400416034343698204186575808495617", 10);

#if 0 // TODO: Activate prover constructor code when proof generation available
    try {
        zkey = BinFileUtils::openExisting(config.starkVerifierFile, "zkey", 1);
        zkeyHeader = ZKeyUtils::loadHeader(zkey.get());

        if (mpz_cmp(zkeyHeader->rPrime, altBbn128r) != 0) {
            throw std::invalid_argument( "zkey curve not supported" );
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
            zkey->getSectionData(4),    // Coefs
            zkey->getSectionData(5),    // pointsA
            zkey->getSectionData(6),    // pointsB1
            zkey->getSectionData(7),    // pointsB2
            zkey->getSectionData(8),    // pointsC
            zkey->getSectionData(9)     // pointsH1
        );

        lastComputedRequestEndTime = 0;

        sem_init(&pendingRequestSem, 0, 0);
        pthread_mutex_init(&mutex, NULL);
        pCurrentRequest = NULL;
        pthread_create(&proverPthread, NULL, proverThread, this);
        pthread_create(&cleanerPthread, NULL, cleanerThread, this);

    } catch (std::exception& e) {
        cerr << "Error: Prover::Prover() got an exception: " << e.what() << '\n';
        exit(-1);
    }
#endif
    // TODO: uncomment when constant polynomials are available
    //Pols2Refs(fr, constPols, constRefs);
}

Prover::~Prover ()
{
    mpz_clear(altBbn128r);
}

void* proverThread(void* arg)
{
    Prover * pProver = (Prover *)arg;
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
        ProverRequest * pProverRequest = pProver->pCurrentRequest;
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

void* cleanerThread(void* arg)
{
    Prover * pProver = (Prover *)arg;
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
            for (uint64_t i=0; i<pProver->completedRequests.size(); i++)
            {
                if (now - pProver->completedRequests[i]->endTime > (int64_t)pProver->config.requestsPersistence)
                {
                    cout << "cleanerThread() deleting request with uuid: " << pProver->completedRequests[i]->uuid << endl;
                    ProverRequest * pProverRequest = pProver->completedRequests[i];
                    pProver->completedRequests.erase(pProver->completedRequests.begin() + i);
                    delete(pProverRequest);
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

string Prover::submitRequest (ProverRequest * pProverRequest) // returns UUID for this request
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

ProverRequest * Prover::waitForRequestToComplete (const string & uuid, const uint64_t timeoutInSeconds) // wait for the request with this UUID to complete; returns NULL if UUID is invalid
{
    zkassert(uuid.size() > 0);
    cout << "Prover::waitForRequestToComplete() waiting for request with UUID: " << uuid << endl;
    
    // We will store here the address of the prove request corresponding to this UUID
    ProverRequest * pProverRequest = NULL;

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

    // Delete the completed request from the completed requests queue
    /*vector<ProveRequest *>::iterator it2;
    it2 = find( completedRequests.begin(), completedRequests.end(), pProveRequest);
    if (it2 == completedRequests.end())
    {
        cerr << "Prover::waitForRequest() failed searching in completed request queue for request of uuid: " << uuid << endl;
    }
    else
    {
        completedRequests.erase(it2);
    }*/
    // TODO: When should we delete the completed request from the completed request queue?

    // Return the request pointer
    return pProverRequest;
}

void Prover::execute (ProverRequest * pProverRequest)
{
    TimerStart(PROVER_EXECUTE);
    bool bFastMode = true;
    zkassert(pProverRequest!=NULL);

    cout << "Prover::execute() timestamp: " << pProverRequest->timestamp << endl;
    cout << "Prover::execute() UUID: " << pProverRequest->uuid << endl;

    // Allocate an area of memory, to store the main and byte4 committed polynomials,
    // and create them using the allocated address
    void * pMainAddress = malloc(MainCommitPols::size()*2);
    zkassert(pMainAddress!=NULL);
    memset(pMainAddress, 0, MainCommitPols::size()*2);
    MainCommitPols mainPols(pMainAddress,2);
    void * pByte4Address = malloc(Byte4CommitPols::size()*2);
    zkassert(pByte4Address!=NULL);
    memset(pByte4Address, 0, Byte4CommitPols::size()*2);
    Byte4CommitPols byte4Pols(pByte4Address,2);

    // Execute the program
    TimerStart(EXECUTOR_EXECUTE);
    MainExecRequired mainExecRequired;
    executor.execute(pProverRequest->input, mainPols, pProverRequest->db, pProverRequest->counters, mainExecRequired, bFastMode);
    TimerStopAndLog(EXECUTOR_EXECUTE);
    
    // Free committed polynomials address space
    free(pMainAddress);
    free(pByte4Address);

    TimerStopAndLog(PROVER_EXECUTE);
}

void Prover::prove (ProverRequest * pProverRequest)
{
    TimerStart(PROVER_PROVE);
    zkassert(pProverRequest!=NULL);

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
    void * pAddress = mapFile(config.cmPolsFile, CommitPols::size(), true);
    cout << "Successfully mapped " << CommitPols::size() << " bytes to file " << config.cmPolsFile << endl;
    CommitPols cmPols(pAddress);

    // This instance will store all data required to execute the rest of State Machines
    MainExecRequired required;

    // Execute the Main State Machine
    TimerStart(EXECUTOR_EXECUTE);
    executor.execute(pProverRequest->input, cmPols.Main, pProverRequest->db, pProverRequest->counters, required);
    TimerStopAndLog(EXECUTOR_EXECUTE);

    // Execute the Storage State Machine
    TimerStart(STORAGE_SM_EXECUTE);
    storageExecutor.execute(required.Storage, cmPols.Storage);
    TimerStopAndLog(STORAGE_SM_EXECUTE);

    // Execute the Byte4 State Machine
    TimerStart(BYTE4_SM_EXECUTE);
    byte4Executor.execute(required.Byte4, cmPols.Byte4);
    TimerStopAndLog(BYTE4_SM_EXECUTE);

    // Execute the Arith State Machine
    TimerStart(ARITH_SM_EXECUTE);
    arithExecutor.execute(required.Arith, cmPols.Arith);
    TimerStopAndLog(ARITH_SM_EXECUTE);

    // Execute the Binary State Machine
    TimerStart(BINARY_SM_EXECUTE);
    binaryExecutor.execute(required.Binary, cmPols.Binary);
    TimerStopAndLog(BINARY_SM_EXECUTE);

    // TODO: Execute the MemAlign State Machine
    
    // Execute the Memory State Machine
    TimerStart(MEMORY_SM_EXECUTE);
    memoryExecutor.execute(required.Memory, cmPols.Mem);
    TimerStopAndLog(MEMORY_SM_EXECUTE);

    // Execute the PaddingKK State Machine
    TimerStart(PADDING_KK_SM_EXECUTE);
    paddingKKExecutor.execute(required.PaddingKK, cmPols.PaddingKK, required.PaddingKKBit);
    TimerStopAndLog(PADDING_KK_SM_EXECUTE);

    // Execute the PaddingKKBit State Machine
    TimerStart(PADDING_KK_BIT_SM_EXECUTE);
    paddingKKBitExecutor.execute(required.PaddingKKBit, cmPols.PaddingKKBit, required.Nine2One);
    TimerStopAndLog(PADDING_KK_BIT_SM_EXECUTE);

    // Execute the Nine2One State Machine
    TimerStart(NINE2ONE_SM_EXECUTE);
    nine2OneExecutor.execute(required.Nine2One, cmPols.Nine2One, required.KeccakF);
    TimerStopAndLog(NINE2ONE_SM_EXECUTE);

    // Execute the KeccakF State Machine
    TimerStart(KECCAK_F_SM_EXECUTE);
    keccakFExecutor.execute(required.KeccakF, cmPols.KeccakF, required.NormGate9);
    TimerStopAndLog(KECCAK_F_SM_EXECUTE);

    // Execute the NormGate9 State Machine
    TimerStart(NORM_GATE_9_SM_EXECUTE);
    normGate9Executor.execute(required.NormGate9, cmPols.NormGate9);
    TimerStopAndLog(NORM_GATE_9_SM_EXECUTE);

    // TODO: Execute the Padding PG State Machine

    // TODO: Execute the PoseidonG State Machine
    
    // Save input to <timestamp>.input.json, after execution
    json inputJsonEx;
    pProverRequest->input.save(inputJsonEx, pProverRequest->db);
    json2file(inputJsonEx, pProverRequest->inputFileEx);

    // Save public.json file
    TimerStart(SAVE_PUBLIC_JSON);
    json publicJson;
    publicJson[0] = fr.toString(cmPols.Main.FREE0[0]);
    json2file(publicJson, pProverRequest->publicFile);
    TimerStopAndLog(SAVE_PUBLIC_JSON);

    /***********************/
    /* STARK Batch Machine */
    /***********************/

    /* TODO: Undo this
    TimerStart(MEM_ALLOC);
    Mem mem;
    MemAlloc(mem, fr, script, cmPols, constRefs, config.constantsTreeFile);
    TimerStopAndLog(MEM_ALLOC);*/

    TimerStart(BATCH_MACHINE_EXECUTOR);
    json starkProof;
    // TODO: Migrate BME to new finite fields library / golden prime
    //BatchMachineExecutor bme(fr, script);
    //bme.execute(mem, starkProof);
    json stark;
    stark["proof"] = starkProof;
    json globalHash;
    globalHash["globalHash"] = fr.toString(cmPols.Main.FREE0[0]);
    stark["publics"] = globalHash;

    TimerStopAndLog(BATCH_MACHINE_EXECUTOR);

    // If stark file present (i.e. enabled) save stark.json file to disk
    if (config.starkFile.size()>0)
    {
        TimerStart(SAVE_STARK_PROOF);
        json2file(stark, config.starkFile);
        TimerStopAndLog(SAVE_STARK_PROOF);
    }

    /****************/
    /* Proof 2 zkIn */
    /****************/

    TimerStart(PROOF2ZKIN);
    json zkin;
    proof2zkin(stark, zkin);
    zkin["globalHash"] = fr.toString(cmPols.Main.FREE0[0]);
    TimerStopAndLog(PROOF2ZKIN);

    // If stark file present (i.e. enabled) save stark.zkin.json file to disk
    if (config.starkFile.size()>0)
    {
        TimerStart(SAVE_ZKIN_PROOF);
        string zkinFile = config.starkFile;
        zkinFile.erase(zkinFile.find_last_not_of(".json")+1);
        zkinFile += ".zkin.json";
        json2file(zkin, zkinFile);
        TimerStopAndLog(SAVE_ZKIN_PROOF);
    }

#ifdef PROVER_INJECT_ZKIN_JSON
    TimerStart(PROVER_INJECT_ZKIN_JSON);
    zkin.clear();
    std::ifstream zkinStream("/home/fractasy/git/zkproverc/testvectors/zkin.json");
    if (!zkinStream.good())
    {
        cerr << "Error: failed loading zkin.json file " << endl;
        exit(-1);
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
    if (ctx->getRemaingInputsToBeSet()!=0)
    {
        cerr << "Error: Not all inputs have been set. Only " << get_main_input_signal_no()-ctx->getRemaingInputsToBeSet() << " out of " << get_main_input_signal_no() << endl;
        exit(-1);
    }
    TimerStopAndLog(CIRCOM_LOAD_JSON);

    // If present, save witness file
    if (config.witnessFile.size()>0)
    {
        TimerStart(CIRCOM_WRITE_BIN_WITNESS);
        writeBinWitness(ctx, config.witnessFile); // No need to write the file to disk, 12-13M fe, in binary, in wtns format
        TimerStopAndLog(CIRCOM_WRITE_BIN_WITNESS);
    }

    TimerStart(CIRCOM_GET_BIN_WITNESS);
    AltBn128::FrElement * pWitness = NULL;
    uint64_t witnessSize = 0;
    getBinWitness(ctx, pWitness, witnessSize);
    TimerStopAndLog(CIRCOM_GET_BIN_WITNESS);

#ifdef PROVER_USE_PROOF_GOOD_JSON
    // Load and parse a good proof JSON file, just for development and testing purposes
    string goodProofFile = "../testvectors/proof.good.json";
    std::ifstream goodProofStream(goodProofFile);
    if (!goodProofStream.good())
    {
        cerr << "Error: failed loading a good proof JSON file " << goodProofFile << endl;
        exit(-1);
    }
    json jsonProof;
    goodProofStream >> jsonProof;
    goodProofStream.close();
#else
    // Generate Groth16 via rapid SNARK
    TimerStart(RAPID_SNARK);
    json jsonProof;
    try
    {
        auto proof = groth16Prover->prove(pWitness);
        jsonProof = proof->toJson();
    }
    catch (std::exception& e)
    {
        cerr << "Error: Prover::Prove() got exception in rapid SNARK:" << e.what() << '\n';
        exit(-1);
    }
    TimerStopAndLog(RAPID_SNARK);
#endif

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

    /* TODO: Undo this
    TimerStart(MEM_FREE);
    MemFree(mem);
    TimerStopAndLog(MEM_FREE);*/

    free(pWitness);

    // Unmap committed polynomials address
    unmapFile(pAddress, CommitPols::size());

    //cout << "Prover::prove() done" << endl;

    TimerStopAndLog(PROVER_PROVE);
}
