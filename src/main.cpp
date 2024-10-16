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
#include "definitions.hpp"
#if (PROVER_FORK_ID == 13)
    #include "fork_13/calcwit.hpp"
    #include "fork_13/circom.hpp"
#else
    #error "Invalid PROVER_FORK_ID"
#endif
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
#include "sm/climb_key/climb_key_executor.hpp"
#include "sm/climb_key/climb_key_test.hpp"
#include "sm/binary/binary_test.hpp"
#include "sm/mem_align/mem_align_test.hpp"
#include "timer.hpp"
#include "hashdb/hashdb_server.hpp"
#include "service/hashdb/hashdb_test.hpp"
#include "service/hashdb/hashdb.hpp"
#include "sha256_test.hpp"
#include "blake_test.hpp"
#include "goldilocks_precomputed.hpp"
#include "zklog.hpp"
#include "ecrecover_test.hpp"
#include "hashdb_singleton.hpp"
#include "unit_test.hpp"
#include "database_cache_test.hpp"
#include "state_manager.hpp"
#include "check_tree_test.hpp"
#include "database_performance_test.hpp"
#include "sha256.hpp"
#include "zkglobals.hpp"

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
            zklog.error("runFileGenBatchProof() failed calling proverRequest.input.load() zkResult=" + to_string(zkResult) + "=" + zkresult2string(zkResult));
            exitProcess();
        }
    }
    TimerStopAndLog(INPUT_LOAD);

    // Create full tracer based on fork ID
    proverRequest.CreateFullTracer();
    if (proverRequest.result != ZKR_SUCCESS)
    {
        zklog.error("runFileGenBatchProof() failed calling proverRequest.CreateFullTracer() zkResult=" + to_string(proverRequest.result) + "=" + zkresult2string(proverRequest.result));
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
    TimerStart(INPUT_LOAD);
    
    // Create and init an empty prover request
    ProverRequest proverRequest(fr, config, prt_processBatch);

    // Load and parse input JSON file
    if (config.inputFile.empty())
    {
        zklog.error("runFileProcessBatch() found config.inputFile empty");
        exitProcess();
    }
    json inputJson;
    file2json(config.inputFile, inputJson);
    zkresult zkResult = proverRequest.input.load(inputJson);
    if (zkResult != ZKR_SUCCESS)
    {
        zklog.error("runFileProcessBatch() failed calling proverRequest.input.load() zkResult=" + to_string(zkResult) + "=" + zkresult2string(zkResult));
        exitProcess();
    }

    TimerStopAndLog(INPUT_LOAD);

    // Create full tracer based on fork ID
    proverRequest.CreateFullTracer();
    if (proverRequest.result != ZKR_SUCCESS)
    {
        zklog.error("runFileProcessBatch() failed calling proverRequest.CreateFullTracer() zkResult=" + to_string(proverRequest.result) + "=" + zkresult2string(proverRequest.result));
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

    zklog.info("runFileProcessBatch(" + config.inputFile + ") got counters: arith=" + to_string(proverRequest.counters.arith) +
        " binary=" + to_string(proverRequest.counters.binary) +
        " keccakF=" + to_string(proverRequest.counters.keccakF) +
        " memAlign=" + to_string(proverRequest.counters.memAlign) +
        " paddingPG=" + to_string(proverRequest.counters.paddingPG) +
        " poseidonG=" + to_string(proverRequest.counters.poseidonG) +
        " steps=" + to_string(proverRequest.counters.steps) +
        " totals:" +
        " arith=" + to_string(processBatchTotalArith) +
        " binary=" + to_string(processBatchTotalBinary) +
        " keccakF=" + to_string(processBatchTotalKeccakF) +
        " memAlign=" + to_string(processBatchTotalMemAlign) +
        " paddingPG=" + to_string(processBatchTotalPaddingPG) +
        " poseidonG=" + to_string(processBatchTotalPoseidonG) +
        " steps=" + to_string(processBatchTotalSteps));
}

class RunFileThreadArguments
{
public:
    Goldilocks &fr;
    Prover &prover;
    Config &config;
    RunFileThreadArguments(Goldilocks &fr, Prover &prover, Config &config) : fr(fr), prover(prover), config(config){};
};

#define RUN_FILE_MULTITHREAD_N_THREADS 10
#define RUN_FILE_MULTITHREAD_N_FILES 10

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
            zklog.error("runFileExecute() failed calling proverRequest.input.load() zkResult=" + to_string(zkResult) + "=" + zkresult2string(zkResult));
            exitProcess();
        }
    }
    TimerStopAndLog(INPUT_LOAD);

    // Create full tracer based on fork ID
    proverRequest.CreateFullTracer();
    if (proverRequest.result != ZKR_SUCCESS)
    {
        zklog.error("runFileExecute() failed calling proverRequest.CreateFullTracer() zkResult=" + to_string(proverRequest.result) + "=" + zkresult2string(proverRequest.result));
        exitProcess();
    }

    // Call the prover
    prover.execute(&proverRequest);
}

int main(int argc, char **argv)
{
    /* CONFIG */

    if (argc == 2)
    {
        if ((strcmp(argv[1], "-v") == 0) || (strcmp(argv[1], "--version") == 0))
        {
            // If requested to only print the version, then exit the program
            return -1;
        }
    }

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
    json configJson;
    file2json(pConfigFile, configJson);
    config.load(configJson);
    zklog.setJsonLogs(config.jsonLogs);
    zklog.setPID(config.proverID.substr(0, 7)); // Set the logs prefix

    // Print the zkProver version
    zklog.info("Version: " + string(ZKEVM_PROVER_VERSION));

    // Test that stderr is properly logged
    cerr << "Checking error channel; ignore this trace\n";
    zklog.warning("Checking warning channel; ignore this trace");

    // Print the configuration file name
    string configFileName = pConfigFile;
    zklog.info("Config file: " + configFileName);

    // Print the number of cores
    zklog.info("Number of cores=" + to_string(getNumberOfCores()));

    // Print AVX mode used
#ifdef __AVX512__
    zklog.info("Vectorization based on AVX512");
#else
    zklog.info("Vectorization based on AVX2");
#endif

    // Print the hostname and the IP address
    string ipAddress;
    getIPAddress(ipAddress);
    zklog.info("IP address=" + ipAddress);

#ifdef DEBUG
    zklog.info("DEBUG defined");
#endif

    config.print();

    TimerStart(WHOLE_PROCESS);

    if (config.check())
    {
        zklog.error("main() failed calling config.check()");
        exitProcess();
    }

    // Create one instance of the Goldilocks finite field instance
    Goldilocks fr;

    // Create one instance of the Poseidon hash library
    PoseidonGoldilocks poseidon;

    // Init globals
    zkGlobalsInit();

#ifdef DEBUG
    zklog.info("BN128 p-1 =" + bn128.toString(bn128.negOne(),16) + " = " + bn128.toString(bn128.negOne(),10));
    zklog.info("FQ    p-1 =" + fq.toString(fq.negOne(),16) + " = " + fq.toString(fq.negOne(),10));
    zklog.info("FEC   p-1 =" + fec.toString(fec.negOne(),16) + " = " + fec.toString(fec.negOne(),10));
    zklog.info("FNEC  p-1 =" + fnec.toString(fnec.negOne(),16) + " = " + fnec.toString(fnec.negOne(),10));
#endif

    // Init the HashDB singleton
    hashDBSingleton.init(fr, config);

    // Init the StateManager singleton
    stateManager.init(config);

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

    // Generate SHA256 SM script
    if (config.runSHA256ScriptGenerator)
    {
        SHA256GenerateScript(config);
    }

    /* TESTS */

    // Test Keccak SM
    if (config.runKeccakTest)
    {
        // Keccak2Test();
        KeccakTest();
        KeccakSMExecutorTest(fr, config);
    }

    // Test Storage SM
    if (config.runStorageSMTest)
    {
        StorageSMTest(fr, poseidon, config);
    }

    // Test Storage SM
    if (config.runClimbKeySMTest)
    {
        ClimbKeySMTest(fr, config);
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

    // Test ECRecover
    if (config.runECRecoverTest)
    {
        ECRecoverTest();
    }

    // Test Database cache
    if (config.runDatabaseCacheTest)
    {
        DatabaseCacheTest();
        //DatabaseCacheBenchmark();
    }

    // Test check tree
    if (config.runCheckTreeTest)
    {
        CheckTreeTest(config);
    }

    // Test Database performance
    if (config.runDatabasePerformanceTest)
    {
        DatabasePerformanceTest();
    }

    // Unit test
    if (config.runUnitTest)
    {
        UnitTest(fr, poseidon, config);
    }

    // If there is nothing else to run, exit normally
    if (!config.runExecutorServer && !config.runExecutorClient && !config.runExecutorClientMultithread &&
        !config.runHashDBServer && !config.runHashDBTest &&
        !config.runAggregatorServer && !config.runAggregatorClient && !config.runAggregatorClientMock &&
        !config.runFileGenBatchProof && !config.runFileGenAggregatedProof && !config.runFileGenFinalProof &&
        !config.runFileProcessBatch && !config.runFileProcessBatchMultithread && !config.runFileExecute)
    {
        return 0;
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

    /* SERVERS */

    // Create the HashDB server and run it, if configured
    HashDBServer *pHashDBServer = NULL;
    if (config.runHashDBServer)
    {
        pHashDBServer = new HashDBServer(fr, config);
        zkassert(pHashDBServer != NULL);
        zklog.info("Launching HashDB server thread...");
        pHashDBServer->runThread();
    }

    // Create the executor server and run it, if configured
    ExecutorServer *pExecutorServer = NULL;
    if (config.runExecutorServer)
    {
        pExecutorServer = new ExecutorServer(fr, prover, config);
        zkassert(pExecutorServer != NULL);
        zklog.info("Launching executor server thread...");
        pExecutorServer->runThread();
    }

    // Create the aggregator server and run it, if configured
    AggregatorServer *pAggregatorServer = NULL;
    if (config.runAggregatorServer)
    {
        pAggregatorServer = new AggregatorServer(fr, config);
        zkassert(pAggregatorServer != NULL);
        zklog.info("Launching aggregator server thread...");
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
                zklog.info("runFileGenBatchProof inputFile=" + tmpConfig.inputFile);
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
                zklog.info("runFileGenAggregatedProof inputFile=" + tmpConfig.inputFile);
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
                zklog.info("runFileGenFinalProof inputFile=" + tmpConfig.inputFile);
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
                zklog.info("runFileProcessBatch inputFile=" + tmpConfig.inputFile);
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
        zklog.info("Launching executor client thread...");
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
        zklog.info("Launching executor client threads...");
        pExecutorClient->runThreads();
    }

    // Run the hashDB test, if configured
    if (config.runHashDBTest)
    {
        zklog.info("Launching HashDB test thread...");
        HashDBTest(config);
    }

    // Create the aggregator client and run it, if configured
    AggregatorClient *pAggregatorClient = NULL;
    if (config.runAggregatorClient)
    {
        pAggregatorClient = new AggregatorClient(fr, config, prover);
        zkassert(pAggregatorClient != NULL);
        zklog.info("Launching aggregator client thread...");
        pAggregatorClient->runThread();
    }

    // Create the aggregator client and run it, if configured
    AggregatorClientMock * pAggregatorClientMock = NULL;
    if (config.runAggregatorClientMock)
    {
        pAggregatorClientMock = new AggregatorClientMock(fr, config);
        zkassert(pAggregatorClientMock != NULL);
        zklog.info("Launching aggregator client mock thread...");
        pAggregatorClientMock->runThread();
    }

    /* WAIT FOR CLIENT THREADS COMPETION */

    // Wait for the executor client thread to end
    if (config.runExecutorClient)
    {
        zkassert(pExecutorClient != NULL);
        int64_t iResult = pExecutorClient->waitForThread();
        sleep(1);
        return iResult;
    }

    // Wait for the executor client thread to end
    if (config.runExecutorClientMultithread)
    {
        zkassert(pExecutorClient != NULL);
        int64_t iResult = pExecutorClient->waitForThreads();
        zklog.info("All executor client threads have completed");
        sleep(1);
        return iResult;
    }

    // Wait for the executor server thread to end
    if (config.runExecutorServer)
    {
        zkassert(pExecutorServer != NULL);
        pExecutorServer->waitForThread();
    }

    // Wait for HashDBServer thread to end
    if (config.runHashDBServer && !config.runHashDBTest)
    {
        zkassert(pHashDBServer != NULL);
        pHashDBServer->waitForThread();
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
    if (pHashDBServer != NULL)
    {
        delete pHashDBServer;
        pHashDBServer = NULL;
    }
    if (pAggregatorServer != NULL)
    {
        delete pAggregatorServer;
        pAggregatorServer = NULL;
    }

    TimerStopAndLog(WHOLE_PROCESS);

    zklog.info("Done");

    return 0;
}