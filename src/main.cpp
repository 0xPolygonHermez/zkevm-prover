#include <iostream>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iomanip>
#include <sys/time.h>
#include "goldilocks_base_field.hpp"
#include "utils.hpp"
#include "config.hpp"
#include "version.hpp"
#include "proof2zkin.hpp"
#include "calcwit.hpp"
#include "circom.hpp"
#include "main.hpp"
#include "prover.hpp"
#include "service/executor/executor_server.hpp"
#include "service/executor/executor_client.hpp"
#include "service/aggregator/aggregator_server.hpp"
#include "service/aggregator/aggregator_client.hpp"
#include "service/aggregator/aggregator_client_mock.hpp"
#include "sm/keccak_f/keccak.hpp"
#include "sm/keccak_f/keccak_executor_test.hpp"
#include "sm/storage/storage_executor.hpp"
#include "sm/storage/storage_test.hpp"
#include "sm/binary/binary_test.hpp"
#include "sm/mem_align/mem_align_test.hpp"
#include "timer.hpp"
#include "statedb/statedb_server.hpp"
#include "service/statedb/statedb_test.hpp"
#include "service/statedb/statedb.hpp"
#include "sha256_test.hpp"
#include "blake_test.hpp"
#include "goldilocks_precomputed.hpp"

using namespace std;
using json = nlohmann::json;

/*
    Prover (available via GRPC service)
    |\
    | Executor (available via GRPC service)
    | |\
    | | Main State Machine
    | | Byte4 State Machine
    | | Binary State Machine
    | | Memory State Machine
    | | Mem Align State Machine
    | | Arithmetic State Machine
    | | Storage State Machine------\
    | |                             |--> Poseidon G State Machine
    | | Padding PG State Machine---/
    | | Padding KK SM -> Padding KK Bit -> Bits 2 Field SM -> Keccak-f SM
    |  \
    |   State DB (available via GRPC service)
    |   |\
    |   | SMT
    |    \
    |     Database
    |\
    | Stark
    |\
    | Circom
*/

void runFileGenBatchProof(Goldilocks fr, Prover &prover, Config &config)
{
    // Load and parse input JSON file
    TimerStart(INPUT_LOAD);
    // Create and init an empty prover request
    ProverRequest proverRequest(fr, config, prt_genBatchProof);
    if (config.inputFile.size() > 0)
    {
        json inputJson;
        file2json(config.inputFile, inputJson);
        zkresult zkResult = proverRequest.input.load(inputJson);
        if (zkResult != ZKR_SUCCESS)
        {
            cerr << "Error: runFileGenBatchProof() failed calling proverRequest.input.load() zkResult=" << zkResult << "=" << zkresult2string(zkResult) << endl;
            exitProcess();
        }
    }
    TimerStopAndLog(INPUT_LOAD);
    
    // Create full tracer based on fork ID
    proverRequest.CreateFullTracer();
    if (proverRequest.result != ZKR_SUCCESS)
    {
        cerr << "Error: runFileGenBatchProof() failed calling proverRequest.CreateFullTracer() zkResult=" << proverRequest.result << "=" << zkresult2string(proverRequest.result) << endl;
        exitProcess();
    }

    // Call the prover
    prover.genBatchProof(&proverRequest);
}

void runFileGenAggregatedProof(Goldilocks fr, Prover &prover, Config &config)
{
    // Load and parse input JSON file
    TimerStart(INPUT_LOAD);
    // Create and init an empty prover request
    ProverRequest proverRequest(fr, config, prt_genAggregatedProof);
    if (config.inputFile.size() > 0)
    {
        file2json(config.inputFile, proverRequest.aggregatedProofInput1);
    }
    if (config.inputFile2.size() > 0)
    {
        file2json(config.inputFile2, proverRequest.aggregatedProofInput2);
    }
    TimerStopAndLog(INPUT_LOAD);

    // Call the prover
    prover.genAggregatedProof(&proverRequest);
}

void runFileGenFinalProof(Goldilocks fr, Prover &prover, Config &config)
{
    // Load and parse input JSON file
    TimerStart(INPUT_LOAD);
    // Create and init an empty prover request
    ProverRequest proverRequest(fr, config, prt_genFinalProof);
    if (config.inputFile.size() > 0)
    {
        file2json(config.inputFile, proverRequest.finalProofInput);
    }
    TimerStopAndLog(INPUT_LOAD);

    // Call the prover
    prover.genFinalProof(&proverRequest);
}

uint64_t processBatchTotalArith = 0;
uint64_t processBatchTotalBinary = 0;
uint64_t processBatchTotalKeccakF = 0;
uint64_t processBatchTotalMemAlign = 0;
uint64_t processBatchTotalPaddingPG = 0;
uint64_t processBatchTotalPoseidonG = 0;
uint64_t processBatchTotalSteps = 0;

void runFileProcessBatch(Goldilocks fr, Prover &prover, Config &config)
{
    // Load and parse input JSON file
    TimerStart(INPUT_LOAD);
    // Create and init an empty prover request
    ProverRequest proverRequest(fr, config, prt_processBatch);
    if (config.inputFile.size() > 0)
    {
        json inputJson;
        file2json(config.inputFile, inputJson);
        zkresult zkResult = proverRequest.input.load(inputJson);
        if (zkResult != ZKR_SUCCESS)
        {
            cerr << "Error: runFileProcessBatch() failed calling proverRequest.input.load() zkResult=" << zkResult << "=" << zkresult2string(zkResult) << endl;
            exitProcess();
        }
    }
    TimerStopAndLog(INPUT_LOAD);
    
    // Create full tracer based on fork ID
    proverRequest.CreateFullTracer();
    if (proverRequest.result != ZKR_SUCCESS)
    {
        cerr << "Error: runFileProcessBatch() failed calling proverRequest.CreateFullTracer() zkResult=" << proverRequest.result << "=" << zkresult2string(proverRequest.result) << endl;
        exitProcess();
    }

    // Call the prover
    prover.processBatch(&proverRequest);

    processBatchTotalArith += proverRequest.counters.arith;
    processBatchTotalBinary += proverRequest.counters.binary;
    processBatchTotalKeccakF += proverRequest.counters.keccakF;
    processBatchTotalMemAlign += proverRequest.counters.memAlign;
    processBatchTotalPaddingPG += proverRequest.counters.paddingPG;
    processBatchTotalPoseidonG += proverRequest.counters.poseidonG;
    processBatchTotalSteps += proverRequest.counters.steps;

    cout << "runFileProcessBatch(" << config.inputFile << ") got counters: arith=" << proverRequest.counters.arith <<
        " binary=" << proverRequest.counters.binary <<
        " keccakF=" << proverRequest.counters.keccakF <<
        " memAlign=" << proverRequest.counters.memAlign <<
        " paddingPG=" << proverRequest.counters.paddingPG <<
        " poseidonG=" << proverRequest.counters.poseidonG <<
        " steps=" << proverRequest.counters.steps <<
        " totals:" <<
        " arith=" << processBatchTotalArith <<
        " binary=" << processBatchTotalBinary <<
        " keccakF=" << processBatchTotalKeccakF <<
        " memAlign=" << processBatchTotalMemAlign <<
        " paddingPG=" << processBatchTotalPaddingPG <<
        " poseidonG=" << processBatchTotalPoseidonG <<
        " steps=" << processBatchTotalSteps << endl;
 }

class RunFileThreadArguments
{
public:
    Goldilocks &fr;
    Prover &prover;
    Config &config;
    RunFileThreadArguments(Goldilocks &fr, Prover &prover, Config &config) : fr(fr), prover(prover), config(config){};
};

#define RUN_FILE_MULTITHREAD_N_THREADS 100
#define RUN_FILE_MULTITHREAD_N_FILES 100

void *runFileProcessBatchThread(void *arg)
{
    RunFileThreadArguments *pArgs = (RunFileThreadArguments *)arg;

    // For all files
    for (uint64_t i = 0; i < RUN_FILE_MULTITHREAD_N_FILES; i++)
    {
        runFileProcessBatch(pArgs->fr, pArgs->prover, pArgs->config);
    }

    return NULL;
}

void runFileProcessBatchMultithread(Goldilocks &fr, Prover &prover, Config &config)
{
    RunFileThreadArguments args(fr, prover, config);

    pthread_t threads[RUN_FILE_MULTITHREAD_N_THREADS];

    // Launch all threads
    for (uint64_t i = 0; i < RUN_FILE_MULTITHREAD_N_THREADS; i++)
    {
        pthread_create(&threads[i], NULL, runFileProcessBatchThread, &args);
    }

    // Wait for all threads to complete
    for (uint64_t i = 0; i < RUN_FILE_MULTITHREAD_N_THREADS; i++)
    {
        pthread_join(threads[i], NULL);
    }
}

void runFileExecute(Goldilocks fr, Prover &prover, Config &config)
{
    // Load and parse input JSON file
    TimerStart(INPUT_LOAD);
    // Create and init an empty prover request
    ProverRequest proverRequest(fr, config, prt_execute);
    if (config.inputFile.size() > 0)
    {
        json inputJson;
        file2json(config.inputFile, inputJson);
        zkresult zkResult = proverRequest.input.load(inputJson);
        if (zkResult != ZKR_SUCCESS)
        {
            cerr << "Error: runFileExecute() failed calling proverRequest.input.load() zkResult=" << zkResult << "=" << zkresult2string(zkResult) << endl;
            exitProcess();
        }
    }
    TimerStopAndLog(INPUT_LOAD);
    
    // Create full tracer based on fork ID
    proverRequest.CreateFullTracer();
    if (proverRequest.result != ZKR_SUCCESS)
    {
        cerr << "Error: runFileExecute() failed calling proverRequest.CreateFullTracer() zkResult=" << proverRequest.result << "=" << zkresult2string(proverRequest.result) << endl;
        exitProcess();
    }

    // Call the prover
    prover.execute(&proverRequest);
}

int main(int argc, char **argv)
{
    /* CONFIG */

    // Print the zkProver version
    cout << "Version: " << string(ZKEVM_PROVER_VERSION) << endl;

    // Test that stderr is properly logged
    cerr << "Error: Checking error channel; ignore this trace" << endl;

    // Print the number of cores
    cout << "Number of cores=" << getNumberOfCores() << endl;

    // Print the hostname and the IP address
    string ipAddress;
    getIPAddress(ipAddress);
    cout << "IP address=" << ipAddress << endl;

#ifdef DEBUG
    cout << "DEBUG defined" << endl;
#endif

    if (argc == 2)
    {
        if ((strcmp(argv[1], "-v") == 0) || (strcmp(argv[1], "--version") == 0))
        {
            // If requested to only print the version, then exit the program
            return 0;
        }
    }

    TimerStart(WHOLE_PROCESS);

    // Parse the name of the configuration file
    char *pConfigFile = (char *)"config/config.json";
    if (argc == 3)
    {
        if ((strcmp(argv[1], "-c") == 0) || (strcmp(argv[1], "--config") == 0))
        {
            pConfigFile = argv[2];
        }
    }

    // Create one instance of Config based on the contents of the file config.json
    TimerStart(LOAD_CONFIG_JSON);
    json configJson;
    file2json(pConfigFile, configJson);
    Config config;
    config.load(configJson);
    config.print();
    TimerStopAndLog(LOAD_CONFIG_JSON);

    // Check required files presence
    bool bError = false;
    if (!fileExists(config.rom))
    {
        cerr << "Error: required file config.rom=" << config.rom << " does not exist" << endl;
        bError = true;
    }
    if (config.generateProof())
    {
        if (!fileExists(config.zkevmConstPols))
        {
            cerr << "Error: required file config.zkevmConstPols=" << config.zkevmConstPols << " does not exist" << endl;
            bError = true;
        }
        if (!fileExists(config.c12aConstPols))
        {
            cerr << "Error: required file config.c12aConstPols=" << config.c12aConstPols << " does not exist" << endl;
            bError = true;
        }
        if (!fileExists(config.recursive1ConstPols))
        {
            cerr << "Error: required file config.recursive1ConstPols=" << config.recursive1ConstPols << " does not exist" << endl;
            bError = true;
        }
        if (!fileExists(config.recursive2ConstPols))
        {
            cerr << "Error: required file config.recursive2ConstPols=" << config.recursive2ConstPols << " does not exist" << endl;
            bError = true;
        }
        if (!fileExists(config.recursivefConstPols))
        {
            cerr << "Error: required file config.recursivefConstPols=" << config.recursivefConstPols << " does not exist" << endl;
            bError = true;
        }

        if (!fileExists(config.zkevmConstantsTree))
        {
            cerr << "Error: required file config.zkevmConstantsTree=" << config.zkevmConstantsTree << " does not exist" << endl;
            bError = true;
        }
        if (!fileExists(config.c12aConstantsTree))
        {
            cerr << "Error: required file config.c12aConstantsTree=" << config.c12aConstantsTree << " does not exist" << endl;
            bError = true;
        }
        if (!fileExists(config.recursive1ConstantsTree))
        {
            cerr << "Error: required file config.recursive1ConstantsTree=" << config.recursive1ConstantsTree << " does not exist" << endl;
            bError = true;
        }
        if (!fileExists(config.recursive2ConstantsTree))
        {
            cerr << "Error: required file config.recursive2ConstantsTree=" << config.recursive2ConstantsTree << " does not exist" << endl;
            bError = true;
        }
        if (!fileExists(config.recursivefConstantsTree))
        {
            cerr << "Error: required file config.recursivefConstantsTree=" << config.recursivefConstantsTree << " does not exist" << endl;
            bError = true;
        }
        if (!fileExists(config.zkevmVerifier))
        {
            cerr << "Error: required file config.zkevmVerifier=" << config.zkevmVerifier << " does not exist" << endl;
            bError = true;
        }
        if (!fileExists(config.recursive1Verifier))
        {
            cerr << "Error: required file config.recursive1Verifier=" << config.recursive1Verifier << " does not exist" << endl;
            bError = true;
        }
        if (!fileExists(config.recursive2Verifier))
        {
            cerr << "Error: required file config.recursive2Verifier=" << config.recursive2Verifier << " does not exist" << endl;
            bError = true;
        }
        if (!fileExists(config.recursive2Verkey))
        {
            cerr << "Error: required file config.recursive2Verkey=" << config.recursive2Verkey << " does not exist" << endl;
            bError = true;
        }
        if (!fileExists(config.finalVerifier))
        {
            cerr << "Error: required file config.finalVerifier=" << config.finalVerifier << " does not exist" << endl;
            bError = true;
        }
        if (!fileExists(config.recursivefVerifier))
        {
            cerr << "Error: required file config.recursivefVerifier=" << config.recursivefVerifier << " does not exist" << endl;
            bError = true;
        }
        if (!fileExists(config.finalStarkZkey))
        {
            cerr << "Error: required file config.finalStarkZkey=" << config.finalStarkZkey << " does not exist" << endl;
            bError = true;
        }
        if (!fileExists(config.storageRomFile))
        {
            cerr << "Error: required file config.storageRomFile=" << config.storageRomFile << " does not exist" << endl;
            bError = true;
        }
        if (!fileExists(config.zkevmStarkInfo))
        {
            cerr << "Error: required file config.zkevmStarkInfo=" << config.zkevmStarkInfo << " does not exist" << endl;
            bError = true;
        }
        if (!fileExists(config.c12aStarkInfo))
        {
            cerr << "Error: required file config.c12aStarkInfo=" << config.c12aStarkInfo << " does not exist" << endl;
            bError = true;
        }
        if (!fileExists(config.recursive1StarkInfo))
        {
            cerr << "Error: required file config.recursive1StarkInfo=" << config.recursive1StarkInfo << " does not exist" << endl;
            bError = true;
        }
        if (!fileExists(config.recursive2StarkInfo))
        {
            cerr << "Error: required file config.recursive2StarkInfo=" << config.recursive2StarkInfo << " does not exist" << endl;
            bError = true;
        }
        if (!fileExists(config.recursivefStarkInfo))
        {
            cerr << "Error: required file config.recursivefStarkInfo=" << config.recursivefStarkInfo << " does not exist" << endl;
            bError = true;
        }
        if (!fileExists(config.c12aExec))
        {
            cerr << "Error: required file config.c12aExec=" << config.c12aExec << " does not exist" << endl;
            bError = true;
        }
        if (!fileExists(config.recursive1Exec))
        {
            cerr << "Error: required file config.recursive1Exec=" << config.recursive1Exec << " does not exist" << endl;
            bError = true;
        }
        if (!fileExists(config.recursive2Exec))
        {
            cerr << "Error: required file config.recursive2Exec=" << config.recursive2Exec << " does not exist" << endl;
            bError = true;
        }
        if (!fileExists(config.recursivefExec))
        {
            cerr << "Error: required file config.recursivefExec=" << config.recursivefExec << " does not exist" << endl;
            bError = true;
        }
    }
    if (bError)
        exitProcess();

    // Create one instance of the Goldilocks finite field instance
    Goldilocks fr;

    // Create one instance of the Poseidon hash library
    PoseidonGoldilocks poseidon;

    // Init goldilocks precomputed
    TimerStart(GOLDILOCKS_PRECOMPUTED_INIT);
    glp.init();
    TimerStopAndLog(GOLDILOCKS_PRECOMPUTED_INIT);    

    /* TOOLS */

    // Generate Keccak SM script
    if (config.runKeccakScriptGenerator)
    {
        KeccakGenerateScript(config);
    }

    /* TESTS */

    // Test Keccak SM
    if (config.runKeccakTest)
    {
        // Keccak2Test();
        KeccakSMTest();
        KeccakSMExecutorTest(fr, config);
    }

    // Test Storage SM
    if (config.runStorageSMTest)
    {
        StorageSMTest(fr, poseidon, config);
    }

    // Test Binary SM
    if (config.runBinarySMTest)
    {
        BinarySMTest(fr, config);
    }

    // Test MemAlign SM
    if (config.runMemAlignSMTest)
    {
        MemAlignSMTest(fr, config);
    }

    // Test SHA256
    if (config.runSHA256Test)
    {
        SHA256Test(fr, config);
    }

    // Test Blake
    if (config.runBlakeTest)
    {
        Blake2b256_Test(fr, config);
    }

    // If there is nothing else to run, exit normally
    if (!config.runExecutorServer && !config.runExecutorClient && !config.runExecutorClientMultithread &&
        !config.runStateDBServer && !config.runStateDBTest &&
        !config.runAggregatorServer && !config.runAggregatorClient && !config.runAggregatorClientMock &&
        !config.runFileGenBatchProof && !config.runFileGenAggregatedProof && !config.runFileGenFinalProof &&
        !config.runFileProcessBatch && !config.runFileProcessBatchMultithread && !config.runFileExecute)
    {
        exit(0);
    }

#if 0
    BatchMachineExecutor::batchInverseTest(fr);
#endif

    // Create output directory, if specified; otherwise, current working directory will be used to store output files
    if (config.outputPath.size() > 0)
    {
        ensureDirectoryExists(config.outputPath);
    }

    // Create an instace of the Prover
    TimerStart(PROVER_CONSTRUCTOR);
    Prover prover(fr,
                  poseidon,
                  config);
    TimerStopAndLog(PROVER_CONSTRUCTOR);

#ifdef DATABASE_USE_CACHE

    /* INIT DB CACHE */
    Database::dbMTCache.setName("MTCache");
    Database::dbProgramCache.setName("ProgramCache");
    Database::dbMTCache.setMaxSize(config.dbMTCacheSize*1024*1024);
    Database::dbProgramCache.setMaxSize(config.dbProgramCacheSize*1024*1024);

    if (config.databaseURL != "local") // remote DB
    {

        if (config.loadDBToMemCache && (config.runAggregatorClient || config.runExecutorServer || config.runStateDBServer))
        {
            TimerStart(DB_CACHE_LOAD);
            // if we have a db cache enabled
            if ((Database::dbMTCache.enabled()) || (Database::dbProgramCache.enabled()))
            {
                if (config.loadDBToMemCacheInParallel) {
                    // Run thread that loads the DB into the dbCache
                    std::thread loadDBThread (loadDb2MemCache, config);
                    loadDBThread.detach();
                } else {
                    loadDb2MemCache(config);
                }
            }
            TimerStopAndLog(DB_CACHE_LOAD);
        }
    }
    
#endif // DATABASE_USE_CACHE

    /* SERVERS */

    // Create the StateDB server and run it, if configured
    StateDBServer *pStateDBServer = NULL;
    if (config.runStateDBServer)
    {
        pStateDBServer = new StateDBServer(fr, config);
        zkassert(pStateDBServer != NULL);
        cout << "Launching StateDB server thread..." << endl;
        pStateDBServer->runThread();
    }

    // Create the executor server and run it, if configured
    ExecutorServer *pExecutorServer = NULL;
    if (config.runExecutorServer)
    {
        pExecutorServer = new ExecutorServer(fr, prover, config);
        zkassert(pExecutorServer != NULL);
        cout << "Launching executor server thread..." << endl;
        pExecutorServer->runThread();
    }

    // Create the aggregator server and run it, if configured
    AggregatorServer *pAggregatorServer = NULL;
    if (config.runAggregatorServer)
    {
        pAggregatorServer = new AggregatorServer(fr, config);
        zkassert(pAggregatorServer != NULL);
        cout << "Launching aggregator server thread..." << endl;
        pAggregatorServer->runThread();
        sleep(5);
    }

    /* FILE-BASED INPUT */

    // Generate a batch proof from the input file
    if (config.runFileGenBatchProof)
    {
        if (config.inputFile.back() == '/') // Process all input files in the folder
        {
            Config tmpConfig = config;
            // Get files sorted alphabetically from the folder
            vector<string> files = getFolderFiles(config.inputFile, true);
            // Process each input file in order
            for (size_t i = 0; i < files.size(); i++)
            {
                tmpConfig.inputFile = config.inputFile + files[i];
                cout << "runFileGenBatchProof inputFile=" << tmpConfig.inputFile << endl;
                // Call the prover
                runFileGenBatchProof(fr, prover, tmpConfig);
            }
        }
        else
        {
            // Call the prover
            runFileGenBatchProof(fr, prover, config);
        }
    }

    // Generate an aggregated proof from the input file
    if (config.runFileGenAggregatedProof)
    {
        if (config.inputFile.back() == '/') // Process all input files in the folder
        {
            Config tmpConfig = config;
            // Get files sorted alphabetically from the folder
            vector<string> files = getFolderFiles(config.inputFile, true);
            // Process each input file in order
            for (size_t i = 0; i < files.size(); i++)
            {
                tmpConfig.inputFile = config.inputFile + files[i];
                cout << "runFileGenAggregatedProof inputFile=" << tmpConfig.inputFile << endl;
                // Call the prover
                runFileGenAggregatedProof(fr, prover, tmpConfig);
            }
        }
        else
        {
            // Call the prover
            runFileGenAggregatedProof(fr, prover, config);
        }
    }

    // Generate a final proof from the input file
    if (config.runFileGenFinalProof)
    {
        if (config.inputFile.back() == '/') // Process all input files in the folder
        {
            Config tmpConfig = config;
            // Get files sorted alphabetically from the folder
            vector<string> files = getFolderFiles(config.inputFile, true);
            // Process each input file in order
            for (size_t i = 0; i < files.size(); i++)
            {
                tmpConfig.inputFile = config.inputFile + files[i];
                cout << "runFileGenFinalProof inputFile=" << tmpConfig.inputFile << endl;
                // Call the prover
                runFileGenFinalProof(fr, prover, tmpConfig);
            }
        }
        else
        {
            // Call the prover
            runFileGenFinalProof(fr, prover, config);
        }
    }

    // Execute (no proof generation) the input file
    if (config.runFileProcessBatch)
    {
        if (config.inputFile.back() == '/')
        {
            Config tmpConfig = config;
            // Get files sorted alphabetically from the folder
            vector<string> files = getFolderFiles(config.inputFile, true);
            // Process each input file in order
            for (size_t i = 0; i < files.size(); i++)
            {
                tmpConfig.inputFile = config.inputFile + files[i];
                cout << "runFileProcessBatch inputFile=" << tmpConfig.inputFile << endl;
                // Call the prover
                runFileProcessBatch(fr, prover, tmpConfig);
            }
        }
        else
        {
            runFileProcessBatch(fr, prover, config);
        }
    }

    // Execute (no proof generation) the input file, in a multithread way
    if (config.runFileProcessBatchMultithread)
    {
        runFileProcessBatchMultithread(fr, prover, config);
    }

    // Execute (no proof generation) the input file, in all SMs
    if (config.runFileExecute)
    {
        runFileExecute(fr, prover, config);
    }

    /* CLIENTS */

    // Create the executor client and run it, if configured
    ExecutorClient *pExecutorClient = NULL;
    if (config.runExecutorClient)
    {
        pExecutorClient = new ExecutorClient(fr, config);
        zkassert(pExecutorClient != NULL);
        cout << "Launching executor client thread..." << endl;
        pExecutorClient->runThread();
    }

    // Run the executor client multithread, if configured
    if (config.runExecutorClientMultithread)
    {
        if (pExecutorClient == NULL)
        {
            pExecutorClient = new ExecutorClient(fr, config);
            zkassert(pExecutorClient != NULL);
        }
        cout << "Launching executor client threads..." << endl;
        pExecutorClient->runThreads();
    }

    // Run the stateDB test, if configured
    if (config.runStateDBTest)
    {
        cout << "Launching StateDB test thread..." << endl;
        runStateDBTest(config);
    }

    // Create the aggregator client and run it, if configured
    AggregatorClient *pAggregatorClient = NULL;
    if (config.runAggregatorClient)
    {
        pAggregatorClient = new AggregatorClient(fr, config, prover);
        zkassert(pAggregatorClient != NULL);
        cout << "Launching aggregator client thread..." << endl;
        pAggregatorClient->runThread();
    }

    // Create the aggregator client and run it, if configured
    AggregatorClientMock * pAggregatorClientMock = NULL;
    if (config.runAggregatorClientMock)
    {
        pAggregatorClientMock = new AggregatorClientMock(fr, config);
        zkassert(pAggregatorClientMock != NULL);
        cout << "Launching aggregator client mock thread..." << endl;
        pAggregatorClientMock->runThread();
    }

    /* WAIT FOR CLIENT THREADS COMPETION */

    // Wait for the executor client thread to end
    if (config.runExecutorClient)
    {
        zkassert(pExecutorClient != NULL);
        pExecutorClient->waitForThread();
        sleep(1);
        return 0;
    }

    // Wait for the executor client thread to end
    if (config.runExecutorClientMultithread)
    {
        zkassert(pExecutorClient != NULL);
        pExecutorClient->waitForThreads();
        cout << "All executor client threads have completed" << endl;
        sleep(1);
        return 0;
    }

    // Wait for the executor server thread to end
    if (config.runExecutorServer)
    {
        zkassert(pExecutorServer != NULL);
        pExecutorServer->waitForThread();
    }

    // Wait for StateDBServer thread to end
    if (config.runStateDBServer && !config.runStateDBTest)
    {
        zkassert(pStateDBServer != NULL);
        pStateDBServer->waitForThread();
    }

    // Wait for the aggregator client thread to end
    if (config.runAggregatorClient)
    {
        zkassert(pAggregatorClient != NULL);
        pAggregatorClient->waitForThread();
        sleep(1);
        return 0;
    }

    // Wait for the aggregator client mock thread to end
    if (config.runAggregatorClientMock)
    {
        zkassert(pAggregatorClientMock != NULL);
        pAggregatorClientMock->waitForThread();
        sleep(1);
        return 0;
    }

    // Wait for the aggregator server thread to end
    if (config.runAggregatorServer)
    {
        zkassert(pAggregatorServer != NULL);
        pAggregatorServer->waitForThread();
    }

    // Clean up
    if (pExecutorClient != NULL)
    {
        delete pExecutorClient;
        pExecutorClient = NULL;
    }
    if (pAggregatorClient != NULL)
    {
        delete pAggregatorClient;
        pAggregatorClient = NULL;
    }
    if (pExecutorServer != NULL)
    {
        delete pExecutorServer;
        pExecutorServer = NULL;
    }
    if (pStateDBServer != NULL)
    {
        delete pStateDBServer;
        pStateDBServer = NULL;
    }
    if (pAggregatorServer != NULL)
    {
        delete pAggregatorServer;
        pAggregatorServer = NULL;
    }

    TimerStopAndLog(WHOLE_PROCESS);

    cout << "Done" << endl;
}