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
#include "calcwit.hpp"
#include "circom.hpp"
#include "main.hpp"
#include "main.recursive1.hpp"
#include "prover.hpp"
#include "service/executor/executor_server.hpp"
#include "service/executor/executor_client.hpp"
#include "service/aggregator/aggregator_server.hpp"
#include "service/aggregator/aggregator_client.hpp"
#include "service/aggregator/aggregator_client_mock.hpp"
#include "sm/keccak_f/keccak.hpp"
#include "sm/storage/storage_executor.hpp"
#include "sm/climb_key/climb_key_executor.hpp"

#ifndef __ZKEVM_LIB__
#include "sm/keccak_f/keccak_executor_test.hpp"
#include "sm/storage/storage_test.hpp"
#include "sm/climb_key/climb_key_test.hpp"
#include "sm/binary/binary_test.hpp"
#include "sm/mem_align/mem_align_test.hpp"
#include "service/hashdb/hashdb_test.hpp"
#include "sha256_test.hpp"
#include "blake_test.hpp"
#include "ecrecover_test.hpp"
#include "database_cache_test.hpp"
#include "check_tree_test.hpp"
#include "database_performance_test.hpp"
#include "smt_64_test.hpp"
#include "page_manager_test.hpp"
#include "key_value_tree_test.hpp"
#endif

#include "timer.hpp"
#include "hashdb/hashdb_server.hpp"
#include "service/hashdb/hashdb.hpp"
#include "goldilocks_precomputed.hpp"
#include "zklog.hpp"
#include "hashdb_singleton.hpp"
#include "unit_test.hpp"
#include "main_sm/fork_8/main_exec_c/account.hpp"
#include "state_manager.hpp"
#include "state_manager_64.hpp"
#include "sha256.hpp"
#include "zkglobals.hpp"
#include "ZkevmSteps.hpp"
#include "C12aSteps.hpp"
#include "Recursive1Steps.hpp"
#include "Recursive2Steps.hpp"
#include "RecursiveFSteps.hpp"
#include "proof2zkinStark.hpp"
#include "starks.hpp"
#ifdef __ZKEVM_LIB__
#include "zkevm_sm.h"
#endif
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

void runFileGenBatchProof(Goldilocks fr, Prover &prover, Config &config, void *pStarkInfo = nullptr)
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
    prover.genBatchProof(&proverRequest, pStarkInfo);
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
            zklog.error("runFileProcessBatch() failed calling proverRequest.input.load() zkResult=" + to_string(zkResult) + "=" + zkresult2string(zkResult));
            exitProcess();
        }
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

int zkevm_main(char *configFile, void *pAddress, void **pSMRequests, void *pSMRequestsOut, void *pStarkInfo)
{

    // Create one instance of Config based on the contents of the file config.json
    json configJson;
    file2json(configFile, configJson);
    config.load(configJson);
    zklog.setJsonLogs(config.jsonLogs);
    zklog.setPID(config.proverID.substr(0, 7)); // Set the logs prefix

    // Print the zkProver version
    zklog.info("Version: " + string(ZKEVM_PROVER_VERSION));

    // Test that stderr is properly logged
    cerr << "Checking error channel; ignore this trace\n";
    zklog.warning("Checking warning channel; ignore this trace");

    // Print the configuration file name
    string configFileName = configFile;
    zklog.info("Config file: " + configFileName);

    // Print the number of cores
    zklog.info("Number of cores=" + to_string(getNumberOfCores()));

    // Print the hostname and the IP address
    string ipAddress;
    getIPAddress(ipAddress);
    zklog.info("IP address=" + ipAddress);

#ifdef __DEBUG__
    zklog.info("__DEBUG__ defined");
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

#ifdef DEBUG
    zklog.info("BN128 p-1 =" + bn128.toString(bn128.negOne(), 16) + " = " + bn128.toString(bn128.negOne(), 10));
    zklog.info("FQ    p-1 =" + fq.toString(fq.negOne(), 16) + " = " + fq.toString(fq.negOne(), 10));
    zklog.info("FEC   p-1 =" + fec.toString(fec.negOne(), 16) + " = " + fec.toString(fec.negOne(), 10));
    zklog.info("FNEC  p-1 =" + fnec.toString(fnec.negOne(), 16) + " = " + fnec.toString(fnec.negOne(), 10));
#endif

    // Generate account zero keys
    fork_8::Account::GenerateZeroKey(fr, poseidon);

    // Init the HashDB singleton
    hashDBSingleton.init(fr, config);

    // Init the StateManager singleton
    if (config.hashDB64)
    {
        stateManager64.init();
    }
    else
    {
        stateManager.init(config);
    }

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

#ifdef DATABASE_USE_CACHE

    /* INIT DB CACHE */
    if (config.useAssociativeCache)
    {
        Database::useAssociativeCache = true;
        Database::dbMTACache.postConstruct(config.log2DbMTAssociativeCacheIndexesSize, config.log2DbMTAssociativeCacheSize, "MTACache");
    }
    else
    {
        Database::useAssociativeCache = false;
        Database::dbMTCache.setName("MTCache");
        Database::dbMTCache.setMaxSize(config.dbMTCacheSize * 1024 * 1024);
    }
    Database::dbProgramCache.setName("ProgramCache");
    Database::dbProgramCache.setMaxSize(config.dbProgramCacheSize * 1024 * 1024);

    if (config.databaseURL != "local") // remote DB
    {

        if (config.loadDBToMemCache && (config.runAggregatorClient || config.runExecutorServer || config.runHashDBServer))
        {
            TimerStart(DB_CACHE_LOAD);
            // if we have a db cache enabled
            if ((Database::dbMTCache.enabled()) || (Database::dbProgramCache.enabled()) || (Database::dbMTACache.enabled()))
            {
                if (config.loadDBToMemCacheInParallel)
                {
                    // Run thread that loads the DB into the dbCache
                    std::thread loadDBThread(loadDb2MemCache, config);
                    loadDBThread.detach();
                }
                else
                {
                    loadDb2MemCache(config);
                }
            }
            TimerStopAndLog(DB_CACHE_LOAD);
        }
    }

#endif // DATABASE_USE_CACHE

#ifndef __ZKEVM_LIB__
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
    // Test PageManager
    if (config.runPageManagerTest)
    {
        PageManagerTest();
    }
    // Test KeyValueTree
    if (config.runKeyValueTreeTest)
    {
        KeyValueTreeTest();
    }

    // Test SMT64
    if (config.runSMT64Test)
    {
        Smt64Test(config);
    }

    // Unit test
    if (config.runUnitTest)
    {
        UnitTest(fr, poseidon, config);
    }
#endif

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
                  config,
                  pAddress);
    prover.setSMRequestsPointer(pSMRequests);
    prover.setSMRequestsOutPointer(pSMRequestsOut);
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
                runFileGenBatchProof(fr, prover, tmpConfig, pStarkInfo);
            }
        }
        else
        {
            // Call the prover
            runFileGenBatchProof(fr, prover, config, pStarkInfo);
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

#ifndef __ZKEVM_LIB__
    // Run the hashDB test, if configured
    if (config.runHashDBTest)
    {
        zklog.info("Launching HashDB test thread...");
        HashDBTest(config);
    }
#endif

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
    AggregatorClientMock *pAggregatorClientMock = NULL;
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
        pExecutorClient->waitForThread();
        sleep(1);
        return 0;
    }

    // Wait for the executor client thread to end
    if (config.runExecutorClientMultithread)
    {
        zkassert(pExecutorClient != NULL);
        pExecutorClient->waitForThreads();
        zklog.info("All executor client threads have completed");
        sleep(1);
        return 0;
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
    return 1;
}
int zkevm_delete_sm_requests(void **pSMRequests)
{
    if (pSMRequests != NULL && (*pSMRequests) != NULL)
    {
        delete (PROVER_FORK_NAMESPACE::MainExecRequired *)pSMRequests[0];
    }
    return 0;
}

int zkevm_arith(void *inputs_, int ninputs, void *pAddress)
{
    std::vector<ArithAction> inputs;
    if (ninputs > 0)
    {
        inputs.assign((ArithAction *)inputs_, (ArithAction *)inputs_ + ninputs);
    }
    ArithExecutor arithExecutor(fr, config);
    PROVER_FORK_NAMESPACE::ArithCommitPols pols(pAddress, PROVER_FORK_NAMESPACE::ArithCommitPols::pilDegree());
    arithExecutor.execute(inputs, pols);
    return 0;
}
int zkevm_arith_req(void *pSMRequests, void *pAddress)
{
    ArithExecutor arithExecutor(fr, config);
    PROVER_FORK_NAMESPACE::ArithCommitPols pols(pAddress, PROVER_FORK_NAMESPACE::ArithCommitPols::pilDegree());
    arithExecutor.execute(((PROVER_FORK_NAMESPACE::MainExecRequired *)pSMRequests)->Arith, pols);
    return 0;
}
int zkevm_memory(void *inputs_, int ninputs, void *pAddress)
{
    std::vector<MemoryAccess> inputs;
    if (ninputs > 0)
    {
        inputs.assign((MemoryAccess *)inputs_, (MemoryAccess *)inputs_ + ninputs);
    }
    MemoryExecutor bits2fieldExecutor(fr, config);
    PROVER_FORK_NAMESPACE::MemCommitPols pols(pAddress, PROVER_FORK_NAMESPACE::MemCommitPols::pilDegree());
    bits2fieldExecutor.execute(inputs, pols);
    return 0;
}
int zkevm_memory_req(void *pSMRequests, void *pAddress)
{
    MemoryExecutor bits2fieldExecutor(fr, config);
    PROVER_FORK_NAMESPACE::MemCommitPols pols(pAddress, PROVER_FORK_NAMESPACE::MemCommitPols::pilDegree());
    bits2fieldExecutor.execute(((PROVER_FORK_NAMESPACE::MainExecRequired *)pSMRequests)->Memory, pols);
    return 0;
}
int zkevm_mem_align(void *inputs_, int ninputs, void *pAddress)
{
    MemAlignAction *pinputs = (MemAlignAction *)inputs_;
    std::vector<MemAlignAction> inputs;
    inputs.resize(ninputs);
    for (int i = 0; i < ninputs; i++)
    {
        inputs[i].m0 = pinputs[i].m0;
        inputs[i].m1 = pinputs[i].m1;
        inputs[i].v = pinputs[i].v;
        inputs[i].w0 = pinputs[i].w0;
        inputs[i].w1 = pinputs[i].w1;
        inputs[i].offset = pinputs[i].offset;
        inputs[i].wr8 = pinputs[i].wr8;
        inputs[i].wr256 = pinputs[i].wr256;
    }
    MemAlignExecutor memAlignExecutor(fr, config);
    PROVER_FORK_NAMESPACE::MemAlignCommitPols pols(pAddress, PROVER_FORK_NAMESPACE::MemAlignCommitPols::pilDegree());
    memAlignExecutor.execute(inputs, pols);
    return 0;
}
int zkevm_mem_align_req(void *pSMRequests, void *pAddress)
{
    MemAlignExecutor memAlignExecutor(fr, config);
    PROVER_FORK_NAMESPACE::MemAlignCommitPols pols(pAddress, PROVER_FORK_NAMESPACE::MemAlignCommitPols::pilDegree());
    memAlignExecutor.execute(((PROVER_FORK_NAMESPACE::MainExecRequired *)pSMRequests)->MemAlign, pols);
    return 0;
}
int zkevm_binary_req(void *pSMRequests, void *pAddress)
{
    BinaryExecutor binaryExecutor(fr, config);
    PROVER_FORK_NAMESPACE::BinaryCommitPols pols(pAddress, PROVER_FORK_NAMESPACE::BinaryCommitPols::pilDegree());
    binaryExecutor.execute(((PROVER_FORK_NAMESPACE::MainExecRequired *)pSMRequests)->Binary, pols);
    return 0;
}
int zkevm_padding_sha256(void *inputs_, int ninputs, void *pAddress, void *pSMRequests, void *pSMRequestsOut)
{

    PaddingSha256ExecutorInput::DTO *p_inputs = (PaddingSha256ExecutorInput::DTO *)inputs_;
    std::vector<PaddingSha256ExecutorInput> inputs;
    PaddingSha256ExecutorInput::fromDTO(p_inputs, ninputs, inputs);
    PaddingSha256Executor paddingSha256Executor(fr);
    PROVER_FORK_NAMESPACE::PaddingSha256CommitPols pols(pAddress, PROVER_FORK_NAMESPACE::PaddingSha256CommitPols::pilDegree());
    std::vector<PaddingSha256BitExecutorInput> &required = ((PROVER_FORK_NAMESPACE::MainExecRequired *)pSMRequests)->PaddingSha256Bit;
    required.clear();
    paddingSha256Executor.execute(inputs, pols, required);
#ifdef __ZKEVM_LIB__
    add_padding_sha256_bit_inputs(pSMRequestsOut, (void *)required.data(), (uint64_t)required.size());
#endif
    return 0;
}
int zkevm_padding_sha256_req(void *pSMRequests, void *pAddress)
{
    PaddingSha256Executor paddingSha256Executor(fr);
    PROVER_FORK_NAMESPACE::PaddingSha256CommitPols pols(pAddress, PROVER_FORK_NAMESPACE::PaddingSha256CommitPols::pilDegree());
    ((PROVER_FORK_NAMESPACE::MainExecRequired *)pSMRequests)->PaddingSha256Bit.clear();
    paddingSha256Executor.execute(((PROVER_FORK_NAMESPACE::MainExecRequired *)pSMRequests)->PaddingSha256,
                                  pols,
                                  ((PROVER_FORK_NAMESPACE::MainExecRequired *)pSMRequests)->PaddingSha256Bit);
    return 0;
}
int zkevm_padding_sha256_bit(void *inputs_, int ninputs, void *pAddress, void *pSMRequests, void *pSMRequestsOut)
{
    std::vector<PaddingSha256BitExecutorInput> inputs;
    if (ninputs > 0)
    {
        inputs.assign((PaddingSha256BitExecutorInput *)inputs_, (PaddingSha256BitExecutorInput *)inputs_ + ninputs);
    }
    PaddingSha256BitExecutor paddingSha256BitExecutor(fr);
    PROVER_FORK_NAMESPACE::PaddingSha256BitCommitPols pols(pAddress, PROVER_FORK_NAMESPACE::PaddingSha256BitCommitPols::pilDegree());
    vector<Bits2FieldSha256ExecutorInput> &required = ((PROVER_FORK_NAMESPACE::MainExecRequired *)pSMRequests)->Bits2FieldSha256;
    required.clear();
    paddingSha256BitExecutor.execute(inputs, pols, required);
#ifdef __ZKEVM_LIB__
    add_bits_2_field_sha256_inputs(pSMRequestsOut, (void *)required.data(), (uint64_t)required.size());
#endif
    return 0;
}
int zkevm_padding_sha256_bit_req(void *pSMRequests, void *pAddress)
{
    PaddingSha256BitExecutor paddingSha256BitExecutor(fr);
    PROVER_FORK_NAMESPACE::PaddingSha256BitCommitPols pols(pAddress, PROVER_FORK_NAMESPACE::PaddingSha256BitCommitPols::pilDegree());
    ((PROVER_FORK_NAMESPACE::MainExecRequired *)pSMRequests)->Bits2FieldSha256.clear();
    paddingSha256BitExecutor.execute(((PROVER_FORK_NAMESPACE::MainExecRequired *)pSMRequests)->PaddingSha256Bit, pols, ((PROVER_FORK_NAMESPACE::MainExecRequired *)pSMRequests)->Bits2FieldSha256);
    return 0;
}
int zkevm_bits2field_sha256(void *inputs_, int ninputs, void *pAddress, void *pSMRequests, void *pSMRequestsOut)
{
    std::vector<Bits2FieldSha256ExecutorInput> inputs;
    if (ninputs > 0)
    {
        inputs.assign((Bits2FieldSha256ExecutorInput *)inputs_, (Bits2FieldSha256ExecutorInput *)inputs_ + ninputs);
    }
    Bits2FieldSha256Executor bits2fieldSha256Executor(fr);
    PROVER_FORK_NAMESPACE::Bits2FieldSha256CommitPols pols(pAddress, PROVER_FORK_NAMESPACE::Bits2FieldSha256CommitPols::pilDegree());
    vector<Sha256FExecutorInput> &required = ((PROVER_FORK_NAMESPACE::MainExecRequired *)pSMRequests)->Sha256F;
    required.clear();
    bits2fieldSha256Executor.execute(inputs, pols, required);
#ifdef __ZKEVM_LIB__
    Sha256FExecutorInput::DTO *dto = Sha256FExecutorInput::toDTO(required);
    add_sha256_f_inputs(pSMRequestsOut, dto, (uint64_t)required.size());
#endif
    return 0;
}
int zkevm_bits2field_sha256_req(void *pSMRequests, void *pAddress)
{
    Bits2FieldSha256Executor bits2fieldSha256Executor(fr);
    PROVER_FORK_NAMESPACE::Bits2FieldSha256CommitPols pols(pAddress, PROVER_FORK_NAMESPACE::Bits2FieldSha256CommitPols::pilDegree());
    ((PROVER_FORK_NAMESPACE::MainExecRequired *)pSMRequests)->Sha256F.clear();
    bits2fieldSha256Executor.execute(((PROVER_FORK_NAMESPACE::MainExecRequired *)pSMRequests)->Bits2FieldSha256, pols, ((PROVER_FORK_NAMESPACE::MainExecRequired *)pSMRequests)->Sha256F);
    return 0;
}
int zkevm_sha256_f(void *inputs_, int ninputs, void *pAddress)
{
    Sha256FExecutorInput::DTO *p_inputs = (Sha256FExecutorInput::DTO *)inputs_;
    std::vector<Sha256FExecutorInput> inputs;
    Sha256FExecutorInput::fromDTO(p_inputs, ninputs, inputs);
    Sha256FExecutor sha256FExecutor(fr, config);
    PROVER_FORK_NAMESPACE::Sha256FCommitPols pols(pAddress, PROVER_FORK_NAMESPACE::Sha256FCommitPols::pilDegree());
    sha256FExecutor.execute(inputs, pols);
    return 0;
}
int zkevm_sha256_f_req(void *pSMRequests, void *pAddress)
{
    Sha256FExecutor sha256FExecutor(fr, config);
    PROVER_FORK_NAMESPACE::Sha256FCommitPols pols(pAddress, PROVER_FORK_NAMESPACE::Sha256FCommitPols::pilDegree());
    sha256FExecutor.execute(((PROVER_FORK_NAMESPACE::MainExecRequired *)pSMRequests)->Sha256F, pols);
    return 0;
}
int zkevm_padding_kk(void *inputs_, int ninputs, void *pAddress, void *pSMRequests, void *pSMRequestsOut)
{
    PaddingKKExecutorInput::DTO *p_inputs = (PaddingKKExecutorInput::DTO *)inputs_;
    std::vector<PaddingKKExecutorInput> inputs;
    PaddingKKExecutorInput::fromDTO(p_inputs, ninputs, inputs);
    PaddingKKExecutor paddingKKExecutor(fr);
    PROVER_FORK_NAMESPACE::PaddingKKCommitPols pols(pAddress, PROVER_FORK_NAMESPACE::PaddingKKCommitPols::pilDegree());
    std::vector<PaddingKKBitExecutorInput> &required = ((PROVER_FORK_NAMESPACE::MainExecRequired *)pSMRequests)->PaddingKKBit;
    required.clear();

    paddingKKExecutor.execute(inputs, pols, required);
#ifdef __ZKEVM_LIB__
    add_padding_kk_bit_inputs(pSMRequestsOut, (void *)required.data(), (uint64_t)required.size());
#endif
    return 0;
}
int zkevm_padding_kk_req(void *pSMRequests, void *pAddress)
{
    PaddingKKExecutor paddingKKExecutor(fr);
    PROVER_FORK_NAMESPACE::PaddingKKCommitPols pols(pAddress, PROVER_FORK_NAMESPACE::PaddingKKCommitPols::pilDegree());
    ((PROVER_FORK_NAMESPACE::MainExecRequired *)pSMRequests)->PaddingKKBit.clear();
    paddingKKExecutor.execute(((PROVER_FORK_NAMESPACE::MainExecRequired *)pSMRequests)->PaddingKK,
                              pols,
                              ((PROVER_FORK_NAMESPACE::MainExecRequired *)pSMRequests)->PaddingKKBit);
    return 0;
}
int zkevm_padding_kk_bit(void *inputs_, int ninputs, void *pAddress, void *pSMRequests, void *pSMRequestsOut)
{
    std::vector<PaddingKKBitExecutorInput> inputs;
    if (ninputs > 0)
    {
        inputs.assign((PaddingKKBitExecutorInput *)inputs_, (PaddingKKBitExecutorInput *)inputs_ + ninputs);
    }
    PaddingKKBitExecutor paddingKKBitExecutor(fr);
    PROVER_FORK_NAMESPACE::PaddingKKBitCommitPols pols(pAddress, PROVER_FORK_NAMESPACE::PaddingKKBitCommitPols::pilDegree());
    vector<Bits2FieldExecutorInput> &required = ((PROVER_FORK_NAMESPACE::MainExecRequired *)pSMRequests)->Bits2Field;
    required.clear();
    paddingKKBitExecutor.execute(inputs, pols, required);
#ifdef __ZKEVM_LIB__
    add_bits_2_field_inputs(pSMRequestsOut, (void *)required.data(), (uint64_t)required.size());
#endif
    return 0;
}
int zkevm_padding_kk_bit_req(void *pSMRequests, void *pAddress)
{
    PaddingKKBitExecutor paddingKKBitExecutor(fr);
    PROVER_FORK_NAMESPACE::PaddingKKBitCommitPols pols(pAddress, PROVER_FORK_NAMESPACE::PaddingKKBitCommitPols::pilDegree());
    ((PROVER_FORK_NAMESPACE::MainExecRequired *)pSMRequests)->Bits2Field.clear();
    paddingKKBitExecutor.execute(((PROVER_FORK_NAMESPACE::MainExecRequired *)pSMRequests)->PaddingKKBit, pols, ((PROVER_FORK_NAMESPACE::MainExecRequired *)pSMRequests)->Bits2Field);
    return 0;
}
int zkevm_bits2field_kk(void *inputs_, int ninputs, void *pAddress, void *pSMRequests, void *pSMRequestsOut)
{
    std::vector<Bits2FieldExecutorInput> inputs;
    if (ninputs > 0)
    {
        inputs.assign((Bits2FieldExecutorInput *)inputs_, (Bits2FieldExecutorInput *)inputs_ + ninputs);
    }
    Bits2FieldExecutor bits2fieldExecutor(fr);
    PROVER_FORK_NAMESPACE::Bits2FieldCommitPols pols(pAddress, PROVER_FORK_NAMESPACE::Bits2FieldCommitPols::pilDegree());
    vector<vector<Goldilocks::Element>> &required = ((PROVER_FORK_NAMESPACE::MainExecRequired *)pSMRequests)->KeccakF;
    required.clear();
    bits2fieldExecutor.execute(inputs, pols, required);
#ifdef __ZKEVM_LIB__
    KeccakFExecutorInput::DTO *dto = KeccakFExecutorInput::toDTO(required);
    add_keccak_f_inputs(pSMRequestsOut, dto, (uint64_t)required.size());
#endif
    return 0;
}
int zkevm_bits2field_kk_req(void *pSMRequests, void *pAddress)
{
    Bits2FieldExecutor bits2fieldExecutor(fr);
    PROVER_FORK_NAMESPACE::Bits2FieldCommitPols pols(pAddress, PROVER_FORK_NAMESPACE::Bits2FieldCommitPols::pilDegree());
    ((PROVER_FORK_NAMESPACE::MainExecRequired *)pSMRequests)->KeccakF.clear();
    bits2fieldExecutor.execute(((PROVER_FORK_NAMESPACE::MainExecRequired *)pSMRequests)->Bits2Field, pols, ((PROVER_FORK_NAMESPACE::MainExecRequired *)pSMRequests)->KeccakF);
    return 0;
}
int zkevm_keccak_f(void *inputs_, int ninputs, void *pAddress)
{
    KeccakFExecutorInput::DTO *p_inputs = (KeccakFExecutorInput::DTO *)inputs_;
    std::vector<std::vector<Goldilocks::Element>> inputs;
    KeccakFExecutorInput::fromDTO(p_inputs, ninputs, inputs);
    KeccakFExecutor keccakFExecutor(fr, config);
    PROVER_FORK_NAMESPACE::KeccakFCommitPols pols(pAddress, PROVER_FORK_NAMESPACE::KeccakFCommitPols::pilDegree());
    keccakFExecutor.execute(inputs, pols);
    return 0;
}
int zkevm_keccak_f_req(void *pSMRequests, void *pAddress)
{
    KeccakFExecutor keccakFExecutor(fr, config);
    PROVER_FORK_NAMESPACE::KeccakFCommitPols pols(pAddress, PROVER_FORK_NAMESPACE::KeccakFCommitPols::pilDegree());
    keccakFExecutor.execute(((PROVER_FORK_NAMESPACE::MainExecRequired *)pSMRequests)->KeccakF, pols);
    return 0;
}

int zkevm_storage_req(void *pSMRequests, void *pAddress)
{
    StorageExecutor storageExecutor(fr, poseidon, config);
    PROVER_FORK_NAMESPACE::StorageCommitPols pols(pAddress, PROVER_FORK_NAMESPACE::StorageCommitPols::pilDegree());
    storageExecutor.execute(((PROVER_FORK_NAMESPACE::MainExecRequired *)pSMRequests)->Storage, pols,
                            ((PROVER_FORK_NAMESPACE::MainExecRequired *)pSMRequests)->PoseidonGFromST, ((PROVER_FORK_NAMESPACE::MainExecRequired *)pSMRequests)->ClimbKey);
    return 0;
}

int zkevm_padding_pg_req(void *pSMRequests, void *pAddress)
{
    PaddingPGExecutor paddingPGExecutor(fr, poseidon);
    PROVER_FORK_NAMESPACE::PaddingPGCommitPols pols(pAddress, PROVER_FORK_NAMESPACE::PaddingPGCommitPols::pilDegree());
    paddingPGExecutor.execute(((PROVER_FORK_NAMESPACE::MainExecRequired *)pSMRequests)->PaddingPG, pols, ((PROVER_FORK_NAMESPACE::MainExecRequired *)pSMRequests)->PoseidonGFromPG);
    return 0;
}

int zkevm_climb_key_req(void *pSMRequests, void *pAddress)
{
    ClimbKeyExecutor climbKeyExecutor(fr, config);
    PROVER_FORK_NAMESPACE::ClimbKeyCommitPols pols(pAddress, PROVER_FORK_NAMESPACE::ClimbKeyCommitPols::pilDegree());
    climbKeyExecutor.execute(((PROVER_FORK_NAMESPACE::MainExecRequired *)pSMRequests)->ClimbKey, pols);
    return 0;
}

int zkevm_poseidon_g_req(void *pSMRequests, void *pAddress)
{
    PoseidonGExecutor poseidonGExecutor(fr, poseidon);
    PROVER_FORK_NAMESPACE::PoseidonGCommitPols pols(pAddress, PROVER_FORK_NAMESPACE::PoseidonGCommitPols::pilDegree());
    poseidonGExecutor.execute(
        ((PROVER_FORK_NAMESPACE::MainExecRequired *)pSMRequests)->PoseidonG,
        ((PROVER_FORK_NAMESPACE::MainExecRequired *)pSMRequests)->PoseidonGFromPG,
        ((PROVER_FORK_NAMESPACE::MainExecRequired *)pSMRequests)->PoseidonGFromST,
        pols);
    return 0;
}

void save_proof(void *pStarkInfo, void *pFriProof, unsigned long numPublicInputs, void *pPublicInputs, char *publicsOutputFile, char *filePrefix)
{
    auto friProof = (FRIProof<Goldilocks::Element> *)pFriProof;
    Goldilocks::Element *publicInputs = (Goldilocks::Element *)pPublicInputs;

    // Generate publics
    json publicStarkJson;
    for (uint64_t i = 0; i < numPublicInputs; i++)
    {
        publicStarkJson[i] = Goldilocks::toString(publicInputs[i]);
    }

    nlohmann::ordered_json jProofRecursive1 = friProof->proofs.proof2json();
    nlohmann::ordered_json zkinRecursive1 = proof2zkinStark(jProofRecursive1, *(StarkInfo *)pStarkInfo);
    zkinRecursive1["publics"] = publicStarkJson;

    // save publics to filestarks
    json2file(publicStarkJson, publicsOutputFile);

    // Save output to file
    if (config.saveOutputToFile)
    {
        json2file(zkinRecursive1, string(filePrefix) + "batch_proof.output.json");
    }
    // Save proof to file
    if (config.saveProofToFile)
    {
        jProofRecursive1["publics"] = publicStarkJson;
        json2file(jProofRecursive1, string(filePrefix) + "batch_proof.proof.json");
    }
}

void *zkevm_steps_new()
{
    ZkevmSteps *zkevmSteps = new ZkevmSteps();
    return zkevmSteps;
}

void zkevm_steps_free(void *pZkevmSteps)
{
    ZkevmSteps *zkevmSteps = (ZkevmSteps *)pZkevmSteps;
    delete zkevmSteps;
}

void *c12a_steps_new()
{
    C12aSteps *c12aSteps = new C12aSteps();
    return c12aSteps;
}
void c12a_steps_free(void *pC12aSteps)
{
    C12aSteps *c12aSteps = (C12aSteps *)pC12aSteps;
    delete c12aSteps;
}
void *recursive1_steps_new()
{
    Recursive1Steps *recursive1Steps = new Recursive1Steps();
    return recursive1Steps;
}
void recursive1_steps_free(void *pRecursive1Steps)
{
    Recursive1Steps *recursive1Steps = (Recursive1Steps *)pRecursive1Steps;
    delete recursive1Steps;
}
void *recursive2_steps_new()
{
    Recursive2Steps *recursive2Steps = new Recursive2Steps();
    return recursive2Steps;
}

void recursive2_steps_free(void *pRecursive2Steps)
{
    Recursive2Steps *recursive2Steps = (Recursive2Steps *)pRecursive2Steps;
    delete recursive2Steps;
}

void *generic_steps_new()
{
    CHelpersSteps *genericSteps = new CHelpersSteps();
    return genericSteps;
}
void generic_steps_free(void *pGenericSteps)
{
    CHelpersSteps *genericSteps = (CHelpersSteps *)pGenericSteps;
    delete genericSteps;
}

void *fri_proof_new(void *pStarks)
{
    Starks<Goldilocks::Element> *starks = (Starks<Goldilocks::Element> *)pStarks;
    FRIProof<Goldilocks::Element> *friProof = new FRIProof<Goldilocks::Element>(starks->starkInfo);

    return friProof;
}

void *fri_proof_get_root(void *pFriProof, uint64_t root_index, uint64_t root_subindex)
{
    FRIProof<Goldilocks::Element> *friProof = (FRIProof<Goldilocks::Element> *)pFriProof;
    return &friProof->proofs.roots[root_index][root_subindex];
}

void *fri_proof_get_tree_root(void *pFriProof, uint64_t tree_index, uint64_t root_index)
{
    FRIProof<Goldilocks::Element> *friProof = (FRIProof<Goldilocks::Element> *)pFriProof;
    return &friProof->proofs.fri.trees[tree_index].root[root_index];
}

void fri_proof_free(void *pFriProof)
{
    FRIProof<Goldilocks::Element> *friProof = (FRIProof<Goldilocks::Element> *)pFriProof;
    delete friProof;
}

void *config_new(char *filename)
{
    Config *config = new Config();
    json configJson;
    file2json(filename, configJson);
    config->load(configJson);

    return config;
}

void config_free(void *pConfig)
{
    Config *config = (Config *)pConfig;
    delete config;
}

void *starkinfo_new(char *filename)
{
    auto starkInfo = new StarkInfo(filename);

    return starkInfo;
}

uint64_t get_mapTotalN(void *pStarkInfo)
{
    return ((StarkInfo *)pStarkInfo)->mapTotalN;
}

void set_mapOffsets(void *pStarkInfo, void *pChelpers)
{
    auto starkInfo = (StarkInfo *)pStarkInfo;
    auto cHelpers = (CHelpers *)pChelpers;
    starkInfo->setMapOffsets(cHelpers->hints);
}

uint64_t get_map_offsets(void *pStarkInfo, char *stage, bool flag)
{
    auto starkInfo = (StarkInfo *)pStarkInfo;
    return starkInfo->mapOffsets[std::make_pair(stage, flag)];
}

uint64_t get_map_sections_n(void *pStarkInfo, char *stage)
{
    auto starkInfo = (StarkInfo *)pStarkInfo;
    return starkInfo->mapSectionsN[stage];
}

void starkinfo_free(void *pStarkInfo)
{
    auto starkInfo = (StarkInfo *)pStarkInfo;
    delete starkInfo;
}

void *starks_new(void *pConfig, char *constPols, bool mapConstPolsFile, char *constantsTree, void *starkInfo, void *cHelpers, void *pAddress)
{
    return new Starks<Goldilocks::Element>(*(Config *)pConfig, {constPols, mapConstPolsFile, constantsTree}, pAddress, *(StarkInfo *)starkInfo, *(CHelpers *)cHelpers, false);
}

void *get_stark_info(void *pStarks)
{
    return &((Starks<Goldilocks::Element> *)pStarks)->starkInfo;
}

void starks_free(void *pStarks)
{
    Starks<Goldilocks::Element> *starks = (Starks<Goldilocks::Element> *)pStarks;
    delete starks;
}

void *chelpers_new(char *cHelpers)
{
    cout << "loading " << cHelpers << endl;
    return new CHelpers(cHelpers);
}

void chelpers_free(void *pChelpers)
{
    CHelpers *cHelpers = (CHelpers *)pChelpers;
    delete cHelpers;
}

void init_hints()
{
    HintHandlerBuilder::registerBuilder(H1H2HintHandler::getName(), std::make_unique<H1H2HintHandlerBuilder>());
    HintHandlerBuilder::registerBuilder(GProdHintHandler::getName(), std::make_unique<GProdHintHandlerBuilder>());
    HintHandlerBuilder::registerBuilder(GSumHintHandler::getName(), std::make_unique<GSumHintHandlerBuilder>());
    HintHandlerBuilder::registerBuilder(SubproofValueHintHandler::getName(), std::make_unique<SubproofValueHintHandlerBuilder>());
}

void *steps_params_new(void *pStarks, void *pChallenges, void *pSubproofValues, void *pEvals, void *pPublicInputs)
{
    Starks<Goldilocks::Element> *starks = (Starks<Goldilocks::Element> *)pStarks;
    Goldilocks::Element *challenges = (Goldilocks::Element *)pChallenges;
    Goldilocks::Element *subproofValues = (Goldilocks::Element *)pSubproofValues;
    Goldilocks::Element *evals = (Goldilocks::Element *)pEvals;
    Goldilocks::Element *publicInputs = (Goldilocks::Element *)pPublicInputs;

    return starks->ffi_create_steps_params(challenges, subproofValues, evals, publicInputs);
}

void *get_steps_params_field(void *pStepsParams, char *name)
{
    StepsParams *stepsParams = (StepsParams *)pStepsParams;

    if (strcmp(name, "q_2ns") == 0)
    {
        return stepsParams->q_2ns;
    }
    else if (strcmp(name, "f_2ns") == 0)
    {
        return stepsParams->f_2ns;
    }
    else
    {
        return NULL;
    }
}

void steps_params_free(void *pStepsParams)
{
    StepsParams *stepsParams = (StepsParams *)pStepsParams;

    delete stepsParams;
}

void extend_and_merkelize(void *pStarks, uint64_t step, void *pParams, void *pProof)
{
    auto starks = (Starks<Goldilocks::Element> *)pStarks;
    auto params = (StepsParams *)pParams;
    auto proof = (FRIProof<Goldilocks::Element> *)pProof;

    starks->ffi_extend_and_merkelize(step, params, proof);
}

void treesGL_get_root(void *pStarks, uint64_t index, void *dst)
{
    Starks<Goldilocks::Element> *starks = (Starks<Goldilocks::Element> *)pStarks;

    starks->ffi_treesGL_get_root(index, (Goldilocks::Element *)dst);
}

void compute_stage(void *pStarks, uint32_t elementType, uint64_t step, void *pParams, void *pProof, void *pTranscript, void *pChelpersSteps)
{
    // type == 1 => Goldilocks
    // type == 2 => BN128
    switch (elementType)
    {
    case 1:
        ((Starks<Goldilocks::Element> *)pStarks)->computeStage(step, *(StepsParams *)pParams, *(FRIProof<Goldilocks::Element> *)pProof, *(TranscriptGL *)pTranscript, (CHelpersSteps *)pChelpersSteps);
        break;
    default:
        cerr << "Invalid elementType: " << elementType << endl;
        break;
    }
}

void compute_evals(void *pStarks, void *pParams, void *pProof)
{
    Starks<Goldilocks::Element> *starks = (Starks<Goldilocks::Element> *)pStarks;
    starks->computeEvals(*(StepsParams *)pParams, *(FRIProof<Goldilocks::Element> *)pProof);
}

void compute_fri_pol(void *pStarks, uint64_t step, void *pParams, void *cHelpersSteps)
{
    Starks<Goldilocks::Element> *starks = (Starks<Goldilocks::Element> *)pStarks;
    starks->computeFRIPol(step, *(StepsParams *)pParams, (CHelpersSteps *)cHelpersSteps);
}

void compute_fri_folding(void *pStarks, void *pProof, void *pFriPol, uint64_t step, void *pChallenge)
{
    Starks<Goldilocks::Element> *starks = (Starks<Goldilocks::Element> *)pStarks;
    starks->computeFRIFolding(*(FRIProof<Goldilocks::Element> *)pProof, (Goldilocks::Element *)pFriPol, step, (Goldilocks::Element *)pChallenge);
}

void compute_fri_queries(void *pStarks, void *pProof, uint64_t *friQueries)
{
    Starks<Goldilocks::Element> *starks = (Starks<Goldilocks::Element> *)pStarks;
    starks->computeFRIQueries(*(FRIProof<Goldilocks::Element> *)pProof, friQueries);
}

void *get_proof_root(void *pProof, uint64_t stage_id, uint64_t index)
{
    FRIProof<Goldilocks::Element> *proof = (FRIProof<Goldilocks::Element> *)pProof;

    return &proof->proofs.roots[stage_id][index];
}

void *get_vector_pointer(void *pStarks, char *name)
{
    Starks<Goldilocks::Element> *starks = (Starks<Goldilocks::Element> *)pStarks;
    return starks->ffi_get_vector_pointer(name);
}

void resize_vector(void *pVector, uint64_t newSize, bool value)
{
    std::vector<bool> *vector = (std::vector<bool> *)pVector;
    vector->resize(newSize, value);
}

void set_bool_vector_value(void *pVector, uint64_t index, bool value)
{
    std::vector<bool> *vector = (std::vector<bool> *)pVector;
    vector->at(index) = value;
}

void clean_symbols_calculated(void *pStarks)
{
    Starks<Goldilocks::Element> *starks = (Starks<Goldilocks::Element> *)pStarks;
    starks->cleanSymbolsCalculated();
}

void set_symbol_calculated(void *pStarks, uint32_t operand, uint64_t id)
{
    Starks<Goldilocks::Element> *starks = (Starks<Goldilocks::Element> *)pStarks;
    starks->ffi_set_symbol_calculated(operand, id);
}

void calculate_hash(void *pStarks, void *pHhash, void *pBuffer, uint64_t nElements)
{
    Starks<Goldilocks::Element> *starks = (Starks<Goldilocks::Element> *)pStarks;
    starks->calculateHash((Goldilocks::Element *)pHhash, (Goldilocks::Element *)pBuffer, nElements);
}

void *commit_pols_starks_new(void *pAddress, uint64_t degree, uint64_t nCommitedPols)
{
    return new CommitPolsStarks(pAddress, degree, nCommitedPols);
}

void commit_pols_starks_free(void *pCommitPolsStarks)
{
    CommitPolsStarks *commitPolsStarks = (CommitPolsStarks *)pCommitPolsStarks;
    delete commitPolsStarks;
}

void circom_get_commited_pols(void *pCommitPolsStarks, char *zkevmVerifier, char *execFile, void *zkin, uint64_t N, uint64_t nCols)
{
    nlohmann::json *zkinJson = (nlohmann::json *)zkin;
    Circom::getCommitedPols((CommitPolsStarks *)pCommitPolsStarks, zkevmVerifier, execFile, *zkinJson, N, nCols);
}

void circom_recursive1_get_commited_pols(void *pCommitPolsStarks, char *zkevmVerifier, char *execFile, void *zkin, uint64_t N, uint64_t nCols)
{
    nlohmann::json *zkinJson = (nlohmann::json *)zkin;
    CircomRecursive1::getCommitedPols((CommitPolsStarks *)pCommitPolsStarks, zkevmVerifier, execFile, *zkinJson, N, nCols);
}

void *zkin_new(void *pStarkInfo, void *pFriProof, unsigned long numPublicInputs, void *pPublicInputs, unsigned long numRootC, void *pRootC)
{
    auto friProof = (FRIProof<Goldilocks::Element> *)pFriProof;
    auto publicInputs = (Goldilocks::Element *)pPublicInputs;
    auto rootC = (Goldilocks::Element *)pRootC;

    // Generate publics
    json publicStarkJson;
    for (uint64_t i = 0; i < numPublicInputs; i++)
    {
        publicStarkJson[i] = Goldilocks::toString(publicInputs[i]);
    }

    json xrootC;
    for (uint64_t i = 0; i < numRootC; i++)
    {
        xrootC[i] = Goldilocks::toString(rootC[i]);
    }

    nlohmann::ordered_json *jProof = new nlohmann::ordered_json();
    nlohmann::json *zkin = new nlohmann::json();
    *jProof = friProof->proofs.proof2json();

    *zkin = proof2zkinStark(*jProof, *(StarkInfo *)pStarkInfo);
    (*zkin)["publics"] = publicStarkJson;
    if (numRootC != 0)
        (*zkin)["rootC"] = xrootC;

    return zkin;
}

void *transcript_new(uint32_t elementType, uint64_t arity, bool custom)
{
    // type == 1 => Goldilocks
    // type == 2 => BN128
    switch (elementType)
    {
    case 1:
        return new TranscriptGL(arity, custom);
    case 2:
        return new TranscriptBN128(arity, custom);
    default:
        return NULL;
    }
}

void transcript_add(void *pTranscript, void *pInput, uint64_t size)
{
    auto transcript = (TranscriptGL *)pTranscript;
    auto input = (Goldilocks::Element *)pInput;

    transcript->put(input, size);
}

void transcript_add_polinomial(void *pTranscript, void *pPolinomial)
{
    auto transcript = (TranscriptGL *)pTranscript;
    auto pol = (Polinomial *)pPolinomial;

    for (uint64_t i = 0; i < pol->degree(); i++)
    {
        transcript->put(pol->operator[](i), pol->dim());
    }
}

void transcript_free(void *pTranscript, uint32_t elementType)
{
    switch (elementType)
    {
    case 1:
        delete (TranscriptGL *)pTranscript;
        break;
    case 2:
        delete (TranscriptBN128 *)pTranscript;
        break;
    }
}

void get_challenge(void *pStarks, void *pTranscript, void *pElement)
{
    TranscriptGL *transcript = (TranscriptGL *)pTranscript;
    ((Starks<Goldilocks::Element> *)pStarks)->getChallenge(*transcript, *(Goldilocks::Element *)pElement);
}

void get_permutations(void *pTranscript, uint64_t *res, uint64_t n, uint64_t nBits)
{
    TranscriptGL *transcript = (TranscriptGL *)pTranscript;
    transcript->getPermutations(res, n, nBits);
}

void *polinomial_new(uint64_t degree, uint64_t dim, char *name)
{
    auto pol = new Polinomial(degree, dim, string(name));
    return (void *)pol;
}

void *polinomial_get_p_element(void *pPolinomial, uint64_t index)
{
    Polinomial *polinomial = (Polinomial *)pPolinomial;
    return polinomial->operator[](index);
}

void polinomial_free(void *pPolinomial)
{
    Polinomial *polinomial = (Polinomial *)pPolinomial;
    delete polinomial;
}

void goldilocks_linear_hash(void *pInput, void *pOutput)
{
    Goldilocks::Element input[12];

    memcpy(input, pInput, 8 * sizeof(Goldilocks::Element));
    memset(&input[8], 0, 4 * sizeof(Goldilocks::Element));

    PoseidonGoldilocks::hash(*(Goldilocks::Element(*)[4])pOutput, input);
}
