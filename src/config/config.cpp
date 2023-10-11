#include <string>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include "definitions.hpp"
#include "config.hpp"
#include "zkassert.hpp"
#include "utils.hpp"
#include "zklog.hpp"
#include "scalar.hpp"

using namespace std;
using json = nlohmann::json;

void ParseEnvironmentBool (const char *pEnv, bool &variable)
{
    const char * pInput = getenv(pEnv);
    if (pInput != NULL)
    {
        string input(pInput);
        if (stringToLower(input) == "true") variable = true;
        else if (stringToLower(input) == "false") variable = false;
        else zklog.error("ParseEnvironmentBool() found invalid environment name=" + string(pEnv) + " value=" + input);
    }
}

void ParseEnvironmentString (const char *pEnv, string &variable)
{
    const char * pInput = getenv(pEnv);
    if (pInput != NULL)
    {
        variable = pInput;
    }
}

void ParseEnvironmentU64 (const char *pEnv, uint64_t &variable)
{
    const char * pInput = getenv(pEnv);
    if (pInput != NULL)
    {
        variable = atoi(pInput);
    }
}

void ParseEnvironmentS64 (const char *pEnv, int64_t &variable)
{
    const char * pInput = getenv(pEnv);
    if (pInput != NULL)
    {
        variable = atoi(pInput);
    }
}

void ParseEnvironmentU16 (const char *pEnv, uint16_t &variable)
{
    const char * pInput = getenv(pEnv);
    if (pInput != NULL)
    {
        variable = atoi(pInput);
    }
}

void ParseBool (const json & config, const char *pJsonName, const char *pEnv, bool &variable, bool defaultValue)
{
    variable = defaultValue;
    if (config.contains(pJsonName) && config[pJsonName].is_boolean())
    {
        variable = config[pJsonName];
    }
    ParseEnvironmentBool(pEnv, variable);
}

void ParseString (const json & config, const char *pJsonName, const char *pEnv, string &variable, const string &defaultValue)
{
    variable = defaultValue;
    if (config.contains(pJsonName) && config[pJsonName].is_string())
    {
        variable = config[pJsonName];
    }
    ParseEnvironmentString(pEnv, variable);
}

void ParseU64 (const json & config, const char *pJsonName, const char *pEnv, uint64_t &variable, const uint64_t &defaultValue)
{
    variable = defaultValue;
    if (config.contains(pJsonName) && config[pJsonName].is_number())
    {
        variable = config[pJsonName];
    }
    ParseEnvironmentU64(pEnv, variable);
}

void ParseS64 (const json & config, const char *pJsonName, const char *pEnv, int64_t &variable, const int64_t &defaultValue)
{
    variable = defaultValue;
    if (config.contains(pJsonName) && config[pJsonName].is_number())
    {
        variable = config[pJsonName];
    }
    ParseEnvironmentS64(pEnv, variable);
}

void ParseU16 (const json & config, const char *pJsonName, const char *pEnv, uint16_t &variable, const uint16_t &defaultValue)
{
    variable = defaultValue;
    if (config.contains(pJsonName) && config[pJsonName].is_number())
    {
        variable = config[pJsonName];
    }
    ParseEnvironmentU16(pEnv, variable);
}

void Config::load(json &config)
{
    zkassert(proverID == "");
    proverID = getUUID();

    // Servers and clients
    ParseBool(config, "runExecutorServer", "RUN_EXECUTOR_SERVER", runExecutorServer, true);
    ParseBool(config, "runExecutorClient", "RUN_EXECUTOR_CLIENT", runExecutorClient, false);
    ParseBool(config, "runExecutorClientMultithread", "RUN_EXECUTOR_CLIENT_MULTITHREAD", runExecutorClientMultithread, false);
    ParseBool(config, "runHashDBServer", "RUN_HASHDB_SERVER", runHashDBServer, true);
    ParseBool(config, "runHashDBTest", "RUN_HASHDB_TEST", runHashDBTest, false);
    ParseBool(config, "runAggregatorServer", "RUN_AGGREGATOR_SERVER", runAggregatorServer, false);
    ParseBool(config, "runAggregatorClient", "RUN_AGGREGATOR_CLIENT", runAggregatorClient, false);
    ParseBool(config, "runAggregatorClientMock", "RUN_AGGREGATOR_CLIENT_MOCK", runAggregatorClientMock, false);

    // Run file
    ParseBool(config, "runFileGenBatchProof", "RUN_FILE_GEN_BATCH_PROOF", runFileGenBatchProof, false);
    ParseBool(config, "runFileGenAggregatedProof", "RUN_FILE_GEN_AGGREGATED_PROOF", runFileGenAggregatedProof, false);
    ParseBool(config, "runFileGenFinalProof", "RUN_FILE_GEN_FINAL_PROOF", runFileGenFinalProof, false);
    ParseBool(config, "runFileProcessBatch", "RUN_FILE_PROCESS_BATCH", runFileProcessBatch, false);
    ParseBool(config, "runFileProcessBatchMultithread", "RUN_FILE_PROCESS_BATCH_MULTITHREAD", runFileProcessBatchMultithread, false);
    ParseBool(config, "runFileExecute", "RUN_FILE_EXECUTE", runFileExecute, false);

    // Tests
    ParseBool(config, "runKeccakScriptGenerator", "RUN_KECCAK_SCRIPT_GENERATOR", runKeccakScriptGenerator, false);
    ParseBool(config, "runKeccakTest", "RUN_KECCAK_TEST", runKeccakTest, false);
    ParseBool(config, "runStorageSMTest", "RUN_STORAGE_SM_TEST", runStorageSMTest, false);
    ParseBool(config, "runBinarySMTest", "RUN_BINARY_SM_TEST", runBinarySMTest, false);
    ParseBool(config, "runMemAlignSMTest", "RUN_MEM_ALIGN_SM_TEST", runMemAlignSMTest, false);
    ParseBool(config, "runSHA256Test", "RUN_SHA256_TEST", runSHA256Test, false);
    ParseBool(config, "runBlakeTest", "RUN_BLAKE_TEST", runBlakeTest, false);
    ParseBool(config, "runECRecoverTest", "RUN_ECRECOVER_TEST", runECRecoverTest, false);
    ParseBool(config, "runDatabaseCacheTest", "RUN_DATABASE_CACHE_TEST", runDatabaseCacheTest, false);
    ParseBool(config, "runDatabaseAssociativeCacheTest", "RUN_DATABASE_ASSOCIATIVE_CACHE_TEST", runDatabaseAssociativeCacheTest, false);
    ParseBool(config, "runCheckTreeTest", "RUN_CHECK_TREE_TEST", runCheckTreeTest, false);
    ParseString(config, "checkTreeRoot", "CHECK_TREE_ROOT", checkTreeRoot, "auto");
    ParseBool(config, "runDatabasePerformanceTest", "RUN_DATABASE_PERFORMANCE_TEST", runDatabasePerformanceTest, false);
    ParseBool(config, "runUnitTest", "RUN_UNIT_TEST", runUnitTest, false);

    // Main SM executor
    ParseBool(config, "useMainExecGenerated", "USE_MAIN_EXEC_GENERATED", useMainExecGenerated, true);
    ParseBool(config, "useMainExecC", "USE_MAIN_EXEC_C", useMainExecC, false);
    ParseBool(config, "executeInParallel", "EXECUTE_IN_PARALLEL", executeInParallel, true);

    // Save to file
    ParseBool(config, "saveDbReadsToFile", "SAVE_DB_READS_TO_FILE", saveDbReadsToFile, false);
    ParseBool(config, "saveRequestToFile", "SAVE_REQUESTS_TO_FILE", saveRequestToFile, false);
    ParseBool(config, "saveDbReadsToFileOnChange", "SAVE_DB_READS_TO_FILE_ON_CHANGE", saveDbReadsToFileOnChange, false);
    ParseBool(config, "saveInputToFile", "SAVE_INPUT_TO_FILE", saveInputToFile, false);
    ParseBool(config, "saveResponseToFile", "SAVE_RESPONSE_TO_FILE", saveResponseToFile, false);
    ParseBool(config, "saveOutputToFile", "SAVE_OUTPUT_TO_FILE", saveOutputToFile, false);
    ParseBool(config, "saveProofToFile", "SAVE_PROOF_TO_FILE", saveProofToFile, false);
    ParseBool(config, "saveFilesInSubfolders", "SAVE_FILES_IN_SUBFOLDERS", saveFilesInSubfolders, false);

    // Load DB to mem cache TODO: Discontinue this functionality
    ParseBool(config, "loadDBToMemCache", "LOAD_DB_TO_MEM_CACHE", loadDBToMemCache, false);
    ParseBool(config, "loadDBToMemCacheInParallel", "LOAD_DB_TO_MEM_CACHE_IN_PARALLEL", loadDBToMemCacheInParallel, false);
    ParseU64(config, "loadDBToMemTimeout", "LOAD_DB_TO_MEM_TIMEOUT", loadDBToMemTimeout, 30*1000*1000); // Default = 30 seconds

    // Server and client ports, hosts, etc.
    ParseU16(config, "executorServerPort", "EXECUTOR_SERVER_PORT", executorServerPort, 50071);
    ParseU16(config, "executorClientPort", "EXECUTOR_CLIENT_PORT", executorClientPort, 50071);
    ParseString(config, "executorClientHost", "EXECUTOR_CLIENT_HOST", executorClientHost, "127.0.0.1");
    ParseU64(config, "executorClientLoops", "EXECUTOR_CLIENT_LOOPS", executorClientLoops, 1);
    ParseBool(config, "executorClientCheckNewStateRoot", "EXECUTOR_CLIENT_CHECK_NEW_STATE_ROOT", executorClientCheckNewStateRoot, false);
    ParseU16(config, "hashDBServerPort", "HASHDB_SERVER_PORT", hashDBServerPort, 50061);
    ParseString(config, "hashDBURL", "HASHDB_URL", hashDBURL, "local");
    ParseBool(config, "hashDB64", "HASHDB64", hashDB64, false);
    ParseString(config, "dbCacheSynchURL", "DB_CACHE_SYNCH_URL", dbCacheSynchURL, "");
    ParseU16(config, "aggregatorServerPort", "AGGREGATOR_SERVER_PORT", aggregatorServerPort, 50081);
    ParseU16(config, "aggregatorClientPort", "AGGREGATOR_CLIENT_PORT", aggregatorClientPort, 50081);
    ParseString(config, "aggregatorClientHost", "AGGREGATOR_CLIENT_HOST", aggregatorClientHost, "127.0.0.1");
    ParseU64(config, "aggregatorClientMockTimeout", "AGGREGATOR_CLIENT_MOCK_TIMEOUT", aggregatorClientMockTimeout, 60 * 1000 * 1000);
    ParseU64(config, "aggregatorClientWatchdogTimeout", "AGGREGATOR_CLIENT_WATCHDOG_TIMEOUT", aggregatorClientWatchdogTimeout, 60 * 1000 * 1000);
    ParseU64(config, "aggregatorClientMaxStreams", "AGGREGATOR_CLIENT_MAX_STREAMS", aggregatorClientMaxStreams, 0);

    // MT cache
    ParseS64(config, "dbMTCacheSize", "DB_MT_CACHE_SIZE", dbMTCacheSize, 8*1024); // Default = 8 GB

    // MT associative cache
    ParseBool(config, "useAssociativeCache", "USE_ASSOCIATIVE_CACHE", useAssociativeCache, false);
    ParseS64(config, "log2DbMTAssociativeCacheSize", "LOG2_DB_MT_ASSOCIATIVE_CACHE_SIZE", log2DbMTAssociativeCacheSize, 24);
    ParseS64(config, "log2DbMTAssociativeCacheIndexesSize", "LOG2_DB_MT_ASSOCIATIVE_CACHE_INDEXES_SIZE", log2DbMTAssociativeCacheIndexesSize, 28);

    // Program (SC) cache
    ParseS64(config, "dbProgramCacheSize", "DB_PROGRAM_CACHE_SIZE", dbProgramCacheSize, 1*1024); // Default = 1 GB

    // Logs
    ParseBool(config, "executorROMLineTraces", "EXECUTOR_ROM_LINE_TRACES", executorROMLineTraces, false);
    ParseBool(config, "executorTimeStatistics", "EXECUTOR_TIME_STATISTICS", executorTimeStatistics, false);
    ParseBool(config, "opcodeTracer", "OPCODE_TRACER", opcodeTracer, false);
    ParseBool(config, "logRemoteDbReads", "LOG_REMOTE_DB_READS", logRemoteDbReads, false);
    ParseBool(config, "logExecutorServerInput", "LOG_EXECUTOR_SERVER_INPUT", logExecutorServerInput, false);
    ParseBool(config, "logExecutorServerInputJson", "LOG_EXECUTOR_SERVER_INPUT_JSON", logExecutorServerInputJson, false);
    ParseU64(config, "logExecutorServerInputGasThreshold", "LOG_EXECUTOR_SERVER_INPUT_GAS_THRESHOLD", logExecutorServerInputGasThreshold, 0);
    ParseBool(config, "logExecutorServerResponses", "LOG_EXECUTOR_SERVER_RESPONSES", logExecutorServerResponses, false);
    ParseBool(config, "logExecutorServerTxs", "LOG_EXECUTOR_SERVER_TXS", logExecutorServerTxs, true);
    ParseBool(config, "dontLoadRomOffsets", "DONT_LOAD_ROM_OFFSETS", dontLoadRomOffsets, false);

    // Files and paths
    ParseString(config, "inputFile", "INPUT_FILE", inputFile, "testvectors/batchProof/input_executor_0.json");
    ParseString(config, "inputFile2", "INPUT_FILE_2", inputFile2, "");
    ParseString(config, "outputPath", "OUTPUT_PATH", outputPath, "output");
    ParseString(config, "configPath", "CONFIG_PATH", configPath, "config");
    ParseString(config, "rom", "ROM", rom, string("src/main_sm/") + string(PROVER_FORK_NAMESPACE_STRING) + string("/scripts/rom.json"));
    ParseString(config, "keccakScriptFile", "KECCAK_SCRIPT_FILE", keccakScriptFile, configPath + "/scripts/keccak_script.json");
    ParseString(config, "storageRomFile", "STORAGE_ROM_FILE", storageRomFile, configPath + "/scripts/storage_sm_rom.json");
    ParseString(config, "zkevmConstPols", "ZKEVM_CONST_POLS", zkevmConstPols, configPath + "/zkevm/zkevm.const");
    ParseString(config, "zkevmConstantsTree", "ZKEVM_CONSTANTS_TREE", zkevmConstantsTree, configPath + "/zkevm/zkevm.consttree");
    ParseString(config, "zkevmStarkInfo", "ZKEVM_STARK_INFO", zkevmStarkInfo, configPath + "/zkevm/zkevm.starkinfo.json");
    ParseString(config, "zkevmVerifier", "ZKEVM_VERIFIER", zkevmVerifier, configPath + "/zkevm/zkevm.verifier.dat");
    ParseString(config, "c12aConstPols", "C12A_CONST_POLS", c12aConstPols, configPath + "/c12a/c12a.const");
    ParseString(config, "c12aConstantsTree", "C12A_CONSTANTS_TREE", c12aConstantsTree, configPath + "/c12a/c12a.consttree");
    ParseString(config, "c12aExec", "C12A_EXEC", c12aExec, configPath + "/c12a/c12a.exec");
    ParseString(config, "c12aStarkInfo", "C12A_STARK_INFO", c12aStarkInfo, configPath + "/c12a/c12a.starkinfo.json");
    ParseString(config, "recursive1ConstPols", "RECURSIVE1_CONST_POLS", recursive1ConstPols, configPath + "/recursive1/recursive1.const");
    ParseString(config, "recursive1ConstantsTree", "RECURSIVE1_CONSTANTS_TREE", recursive1ConstantsTree, configPath + "/recursive1/recursive1.consttree");
    ParseString(config, "recursive1Exec", "RECURSIVE1_EXEC", recursive1Exec, configPath + "/recursive1/recursive1.exec");
    ParseString(config, "recursive1StarkInfo", "RECURSIVE1_STARK_INFO", recursive1StarkInfo, configPath + "/recursive1/recursive1.starkinfo.json");
    ParseString(config, "recursive1Verifier", "RECURSIVE1_VERIFIER", recursive1Verifier, configPath + "/recursive1/recursive1.verifier.dat");
    ParseString(config, "recursive2ConstPols", "RECURSIVE2_CONST_POLS", recursive2ConstPols, configPath + "/recursive2/recursive2.const");
    ParseString(config, "recursive2ConstantsTree", "RECURSIVE2_CONSTANTS_TREE", recursive2ConstantsTree, configPath + "/recursive2/recursive2.consttree");
    ParseString(config, "recursive2Exec", "RECURSIVE2_EXEC", recursive2Exec, configPath + "/recursive2/recursive2.exec");
    ParseString(config, "recursive2StarkInfo", "RECURSIVE2_STARK_INFO", recursive2StarkInfo, configPath + "/recursive2/recursive2.starkinfo.json");
    ParseString(config, "recursive2Verifier", "RECURSIVE2_VERIFIER", recursive2Verifier, configPath + "/recursive2/recursive2.verifier.dat");
    ParseString(config, "recursive2Verkey", "RECURSIVE2_VERKEY", recursive2Verkey, configPath + "/recursive2/recursive2.verkey.json");
    ParseString(config, "recursivefConstPols", "RECURSIVEF_CONST_POLS", recursivefConstPols, configPath + "/recursivef/recursivef.const");
    ParseString(config, "recursivefConstantsTree", "RECURSIVEF_CONSTANTS_TREE", recursivefConstantsTree, configPath + "/recursivef/recursivef.consttree");
    ParseString(config, "recursivefExec", "RECURSIVEF_EXEC", recursivefExec, configPath + "/recursivef/recursivef.exec");
    ParseString(config, "recursivefStarkInfo", "RECURSIVEF_STARK_INFO", recursivefStarkInfo, configPath + "/recursivef/recursivef.starkinfo.json");
    ParseString(config, "recursivefVerifier", "RECURSIVEF_VERIFIER", recursivefVerifier, configPath + "/recursivef/recursivef.verifier.dat");
    ParseString(config, "finalVerifier", "FINAL_VERIFIER", finalVerifier, configPath + "/final/final.verifier.dat");
    ParseString(config, "finalVerkey", "FINAL_VERKEY", finalVerkey, configPath + "/final/final.fflonk.verkey.json");
    ParseString(config, "finalStarkZkey", "FINAL_STARK_ZKEY", finalStarkZkey, configPath + "/final/final.fflonk.zkey");
    ParseString(config, "zkevmCmPols", "ZKEVM_CM_POLS", zkevmCmPols, "");
    ParseString(config, "zkevmCmPolsAfterExecutor", "ZKEVM_CM_POLS_AFTER_EXECUTOR", zkevmCmPolsAfterExecutor, "");
    ParseString(config, "c12aCmPols", "C12A_CM_POLS", c12aCmPols, "");
    ParseString(config, "recursive1CmPols", "RECURSIVE1_CM_POLS", recursive1CmPols, "");
    ParseBool(config, "mapConstPolsFile", "MAP_CONST_POLS_FILE", mapConstPolsFile, false);
    ParseBool(config, "mapConstantsTreeFile", "MAP_CONSTANTS_TREE_FILE", mapConstantsTreeFile, false);
    ParseString(config, "proofFile", "PROOF_FILE", proofFile, "proof.json");
    ParseString(config, "publicsOutput", "PUBLICS_OUTPUT", publicsOutput, "public.json");
    ParseString(config, "keccakPolsFile", "KECCAK_POLS_FILE", keccakPolsFile, "keccak_pols.json");
    ParseString(config, "keccakConnectionsFile", "KECCAK_CONNECTIONS_FILE", keccakConnectionsFile, "keccak_connections.json");

    // Database
    ParseString(config, "databaseURL", "DATABASE_URL", databaseURL, "local");
    ParseString(config, "dbNodesTableName", "DB_NODES_TABLE_NAME", dbNodesTableName, "state.nodes");
    ParseString(config, "dbProgramTableName", "DB_PROGRAM_TABLE_NAME", dbProgramTableName, "state.program");
    ParseBool(config, "dbMultiWrite", "DB_MULTIWRITE", dbMultiWrite, true);
    ParseU64(config, "dbMultiWriteSingleQuerySize", "DB_MULTIWRITE_SINGLE_QUERY_SIZE", dbMultiWriteSingleQuerySize, 20*1024*1024);
    ParseBool(config, "dbConnectionsPool", "DB_CONNECTIONS_POOL", dbConnectionsPool, true);
    ParseU64(config, "dbNumberOfPoolConnections", "DB_NUMBER_OF_POOL_CONNECTIONS", dbNumberOfPoolConnections, 30);
    ParseBool(config, "dbMetrics", "DB_METRICS", dbMetrics, true);
    ParseBool(config, "dbClearCache", "DB_CLEAR_CACHE", dbClearCache, false);
    ParseBool(config, "dbGetTree", "DB_GET_TREE", dbGetTree, true);
    ParseBool(config, "dbReadOnly", "DB_READ_ONLY", dbReadOnly, false);
    ParseU64(config, "dbReadRetryCounter", "DB_READ_RETRY_COUNTER", dbReadRetryCounter, 10);
    ParseU64(config, "dbReadRetryDelay", "DB_READ_RETRY_DELAY", dbReadRetryDelay, 100*1000);

    // State Manager
    ParseBool(config, "stateManager", "STATE_MANAGER", stateManager, true);
    ParseBool(config, "stateManagerPurge", "STATE_MANAGER_PURGE", stateManagerPurge, true);
    ParseBool(config, "stateManagerPurgeTxs", "STATE_MANAGER_PURGE_TXS", stateManagerPurgeTxs, true);

    // Threads
    ParseU64(config, "cleanerPollingPeriod", "CLEANER_POLLING_PERIOD", cleanerPollingPeriod, 600);
    ParseU64(config, "requestsPersistence", "REQUESTS_PERSISTENCE", requestsPersistence, 3600);
    ParseU64(config, "maxExecutorThreads", "MAX_EXECUTOR_THREADS", maxExecutorThreads, 20);
    ParseU64(config, "maxProverThreads", "MAX_PROVER_THREADS", maxProverThreads, 8);
    ParseU64(config, "maxHashDBThreads", "MAX_HASHDB_THREADS", maxHashDBThreads, 8);

    // Prover name, name of this instance as per configuration
    ParseString(config, "proverName", "PROVER_NAME", proverName, "UNSPECIFIED");

    // Memory allocation
    ParseU64(config, "fullTracerTraceReserveSize", "FULL_TRACER_TRACE_RESERVE_SIZE", fullTracerTraceReserveSize, 256*1024);

    // ECRecover
    ParseBool(config, "ECRecoverPrecalc", "ECRECOVER_PRECALC", ECRecoverPrecalc, false);
    ParseU64(config, "ECRecoverPrecalcNThreads", "ECRECOVER_PRECALC_N_THREADS", ECRecoverPrecalcNThreads, 16);
}

void Config::print(void)
{
    zklog.info("Configuration:");

    zklog.info("    proverID=" + proverID);
    zklog.info("    proverName=" + proverName);

    zklog.info(toJson().dump(4));
}

json Config::toJson(void)
{
    json j;

    j["proverID"] = proverID;

    // Servers and clients
    j["runExecutorServer"] = runExecutorServer; // ParseBool(config, "runExecutorServer", "RUN_EXECUTOR_SERVER", runExecutorServer, true);
    j["runExecutorClient"] = runExecutorClient; // ParseBool(config, "runExecutorClient", "RUN_EXECUTOR_CLIENT", runExecutorClient, false);
    j["runExecutorClientMultithread"] = runExecutorClientMultithread; // ParseBool(config, "runExecutorClientMultithread", "RUN_EXECUTOR_CLIENT_MULTITHREAD", runExecutorClientMultithread, false);
    j["runHashDBServer"] = runHashDBServer; // ParseBool(config, "runHashDBServer", "RUN_HASHDB_SERVER", runHashDBServer, true);
    j["runHashDBTest"] = runHashDBTest; // ParseBool(config, "runHashDBTest", "RUN_HASHDB_TEST", runHashDBTest, false);
    j["runAggregatorServer"] = runAggregatorServer; // ParseBool(config, "runAggregatorServer", "RUN_AGGREGATOR_SERVER", runAggregatorServer, false);
    j["runAggregatorClient"] = runAggregatorClient; // ParseBool(config, "runAggregatorClient", "RUN_AGGREGATOR_CLIENT", runAggregatorClient, false);
    j["runAggregatorClientMock"] = runAggregatorClientMock; // ParseBool(config, "runAggregatorClientMock", "RUN_AGGREGATOR_CLIENT_MOCK", runAggregatorClientMock, false);

    // Run file
    j["runFileGenBatchProof"] = runFileGenBatchProof; // ParseBool(config, "runFileGenBatchProof", "RUN_FILE_GEN_BATCH_PROOF", runFileGenBatchProof, false);
    j["runFileGenAggregatedProof"] = runFileGenAggregatedProof; // ParseBool(config, "runFileGenAggregatedProof", "RUN_FILE_GEN_AGGREGATED_PROOF", runFileGenAggregatedProof, false);
    j["runFileGenFinalProof"] = runFileGenFinalProof; // ParseBool(config, "runFileGenFinalProof", "RUN_FILE_GEN_FINAL_PROOF", runFileGenFinalProof, false);
    j["runFileProcessBatch"] = runFileProcessBatch; // ParseBool(config, "runFileProcessBatch", "RUN_FILE_PROCESS_BATCH", runFileProcessBatch, false);
    j["runFileProcessBatchMultithread"] = runFileProcessBatchMultithread; // ParseBool(config, "runFileProcessBatchMultithread", "RUN_FILE_PROCESS_BATCH_MULTITHREAD", runFileProcessBatchMultithread, false);
    j["runFileExecute"] = runFileExecute; // ParseBool(config, "runFileExecute", "RUN_FILE_EXECUTE", runFileExecute, false);

    // Tests
    j["runKeccakScriptGenerator"] = runKeccakScriptGenerator; // ParseBool(config, "runKeccakScriptGenerator", "RUN_KECCAK_SCRIPT_GENERATOR", runKeccakScriptGenerator, false);
    j["runKeccakTest"] = runKeccakTest; // ParseBool(config, "runKeccakTest", "RUN_KECCAK_TEST", runKeccakTest, false);
    j["runStorageSMTest"] = runStorageSMTest; // ParseBool(config, "runStorageSMTest", "RUN_STORAGE_SM_TEST", runStorageSMTest, false);
    j["runBinarySMTest"] = runBinarySMTest; // ParseBool(config, "runBinarySMTest", "RUN_BINARY_SM_TEST", runBinarySMTest, false);
    j["runMemAlignSMTest"] = runMemAlignSMTest; // ParseBool(config, "runMemAlignSMTest", "RUN_MEM_ALIGN_SM_TEST", runMemAlignSMTest, false);
    j["runSHA256Test"] = runSHA256Test; // ParseBool(config, "runSHA256Test", "RUN_SHA256_TEST", runSHA256Test, false);
    j["runBlakeTest"] = runBlakeTest; // ParseBool(config, "runBlakeTest", "RUN_BLAKE_TEST", runBlakeTest, false);
    j["runECRecoverTest"] = runECRecoverTest; // ParseBool(config, "runECRecoverTest", "RUN_ECRECOVER_TEST", runECRecoverTest, false);
    j["runDatabaseCacheTest"] = runDatabaseCacheTest; // ParseBool(config, "runDatabaseCacheTest", "RUN_DATABASE_CACHE_TEST", runDatabaseCacheTest, false);
    j["runDatabaseAssociativeCacheTest"] = runDatabaseAssociativeCacheTest; // ParseBool(config, "runDatabaseAssociativeCacheTest", "RUN_DATABASE_ASSOCIATIVE_CACHE_TEST", runDatabaseAssociativeCacheTest, false);
    j["runCheckTreeTest"] = runCheckTreeTest; // ParseBool(config, "runCheckTreeTest", "RUN_CHECK_TREE_TEST", runCheckTreeTest, false);
    j["checkTreeRoot"] = checkTreeRoot; // ParseString(config, "checkTreeRoot", "CHECK_TREE_ROOT", checkTreeRoot, "auto");
    j["runDatabasePerformanceTest"] = runDatabasePerformanceTest; // ParseBool(config, "runDatabasePerformanceTest", "RUN_DATABASE_PERFORMANCE_TEST", runDatabasePerformanceTest, false);
    j["runUnitTest"] = runUnitTest; // ParseBool(config, "runUnitTest", "RUN_UNIT_TEST", runUnitTest, false);

    // Main SM executor
    j["useMainExecGenerated"] = useMainExecGenerated; // ParseBool(config, "useMainExecGenerated", "USE_MAIN_EXEC_GENERATED", useMainExecGenerated, true);
    j["useMainExecC"] = useMainExecC; // ParseBool(config, "useMainExecC", "USE_MAIN_EXEC_C", useMainExecC, false);
    j["executeInParallel"] = executeInParallel; // ParseBool(config, "executeInParallel", "EXECUTE_IN_PARALLEL", executeInParallel, true);

    // Save to file
    j["saveDbReadsToFile"] = saveDbReadsToFile; // ParseBool(config, "saveDbReadsToFile", "SAVE_DB_READS_TO_FILE", saveDbReadsToFile, false);
    j["saveRequestToFile"] = saveRequestToFile; // ParseBool(config, "saveRequestToFile", "SAVE_REQUESTS_TO_FILE", saveRequestToFile, false);
    j["saveDbReadsToFileOnChange"] = saveDbReadsToFileOnChange; // ParseBool(config, "saveDbReadsToFileOnChange", "SAVE_DB_READS_TO_FILE_ON_CHANGE", saveDbReadsToFileOnChange, false);
    j["saveInputToFile"] = saveInputToFile; // ParseBool(config, "saveInputToFile", "SAVE_INPUT_TO_FILE", saveInputToFile, false);
    j["saveResponseToFile"] = saveResponseToFile; // ParseBool(config, "saveResponseToFile", "SAVE_RESPONSE_TO_FILE", saveResponseToFile, false);
    j["saveOutputToFile"] = saveOutputToFile; // ParseBool(config, "saveOutputToFile", "SAVE_OUTPUT_TO_FILE", saveOutputToFile, false);
    j["saveProofToFile"] = saveProofToFile; // ParseBool(config, "saveProofToFile", "SAVE_PROOF_TO_FILE", saveProofToFile, false);
    j["saveFilesInSubfolders"] = saveFilesInSubfolders; // ParseBool(config, "saveFilesInSubfolders", "SAVE_FILES_IN_SUBFOLDERS", saveFilesInSubfolders, false);

    // Load DB to mem cache TODO: Discontinue this functionality
    j["loadDBToMemCache"] = loadDBToMemCache; // ParseBool(config, "loadDBToMemCache", "LOAD_DB_TO_MEM_CACHE", loadDBToMemCache, false);
    j["loadDBToMemCacheInParallel"] = loadDBToMemCacheInParallel; // ParseBool(config, "loadDBToMemCacheInParallel", "LOAD_DB_TO_MEM_CACHE_IN_PARALLEL", loadDBToMemCacheInParallel, false);
    j["loadDBToMemTimeout"] = loadDBToMemTimeout; // ParseU64(config, "loadDBToMemTimeout", "LOAD_DB_TO_MEM_TIMEOUT", loadDBToMemTimeout, 30*1000*1000); // Default = 30 seconds

    // Server and client ports, hosts, etc.
    j["executorServerPort"] = executorServerPort; // ParseU16(config, "executorServerPort", "EXECUTOR_SERVER_PORT", executorServerPort, 50071);
    j["executorClientPort"] = executorClientPort; // ParseU16(config, "executorClientPort", "EXECUTOR_CLIENT_PORT", executorClientPort, 50071);
    j["executorClientHost"] = executorClientHost; // ParseString(config, "executorClientHost", "EXECUTOR_CLIENT_HOST", executorClientHost, "127.0.0.1");
    j["executorClientLoops"] = executorClientLoops; // ParseU64(config, "executorClientLoops", "EXECUTOR_CLIENT_LOOPS", executorClientLoops, 1);
    j["executorClientCheckNewStateRoot"] = executorClientCheckNewStateRoot; // ParseBool(config, "executorClientCheckNewStateRoot", "EXECUTOR_CLIENT_CHECK_NEW_STATE_ROOT", executorClientCheckNewStateRoot, false);
    j["hashDBServerPort"] = hashDBServerPort; // ParseU16(config, "hashDBServerPort", "HASHDB_SERVER_PORT", hashDBServerPort, 50061);
    j["hashDBURL"] = hashDBURL; // ParseString(config, "hashDBURL", "HASHDB_URL", hashDBURL, "local");
    j["hashDB64"] = hashDB64; // ParseBool(config, "hashDB64", "HASHDB64", hashDB64, false);
    j["dbCacheSynchURL"] = dbCacheSynchURL; // ParseString(config, "dbCacheSynchURL", "DB_CACHE_SYNCH_URL", dbCacheSynchURL, "");
    j["aggregatorServerPort"] = aggregatorServerPort; // ParseU16(config, "aggregatorServerPort", "AGGREGATOR_SERVER_PORT", aggregatorServerPort, 50081);
    j["aggregatorClientPort"] = aggregatorClientPort; // ParseU16(config, "aggregatorClientPort", "AGGREGATOR_CLIENT_PORT", aggregatorClientPort, 50081);
    j["aggregatorClientHost"] = aggregatorClientHost; // ParseString(config, "aggregatorClientHost", "AGGREGATOR_CLIENT_HOST", aggregatorClientHost, "127.0.0.1");
    j["aggregatorClientMockTimeout"] = aggregatorClientMockTimeout; // ParseU64(config, "aggregatorClientMockTimeout", "AGGREGATOR_CLIENT_MOCK_TIMEOUT", aggregatorClientMockTimeout, 60 * 1000 * 1000);
    j["aggregatorClientWatchdogTimeout"] = aggregatorClientWatchdogTimeout; // ParseU64(config, "aggregatorClientWatchdogTimeout", "AGGREGATOR_CLIENT_WATCHDOG_TIMEOUT", aggregatorClientWatchdogTimeout, 60 * 1000 * 1000);
    j["aggregatorClientMaxStreams"] = aggregatorClientMaxStreams; // ParseU64(config, "aggregatorClientMaxStreams", "AGGREGATOR_CLIENT_MAX_STREAMS", aggregatorClientMaxStreams, 0);

    // MT cache
    j["dbMTCacheSize"] = dbMTCacheSize; // ParseS64(config, "dbMTCacheSize", "DB_MT_CACHE_SIZE", dbMTCacheSize, 8*1024); // Default = 8 GB

    // MT associative cache
    j["useAssociativeCache"] = useAssociativeCache; // ParseBool(config, "useAssociativeCache", "USE_ASSOCIATIVE_CACHE", useAssociativeCache, false);
    j["log2DbMTAssociativeCacheSize"] = log2DbMTAssociativeCacheSize; // ParseS64(config, "log2DbMTAssociativeCacheSize", "LOG2_DB_MT_ASSOCIATIVE_CACHE_SIZE", log2DbMTAssociativeCacheSize, 24);
    j["log2DbMTAssociativeCacheIndexesSize"] = log2DbMTAssociativeCacheIndexesSize; // ParseS64(config, "log2DbMTAssociativeCacheIndexesSize", "LOG2_DB_MT_ASSOCIATIVE_CACHE_INDEXES_SIZE", log2DbMTAssociativeCacheIndexesSize, 28);

    // Program (SC) cache
    j["dbProgramCacheSize"] = dbProgramCacheSize; // ParseS64(config, "dbProgramCacheSize", "DB_PROGRAM_CACHE_SIZE", dbProgramCacheSize, 1*1024); // Default = 1 GB

    // Logs
    j["executorROMLineTraces"] = executorROMLineTraces; // ParseBool(config, "executorROMLineTraces", "EXECUTOR_ROM_LINE_TRACES", executorROMLineTraces, false);
    j["executorTimeStatistics"] = executorTimeStatistics; // ParseBool(config, "executorTimeStatistics", "EXECUTOR_TIME_STATISTICS", executorTimeStatistics, false);
    j["opcodeTracer"] = opcodeTracer; // ParseBool(config, "opcodeTracer", "OPCODE_TRACER", opcodeTracer, false);
    j["logRemoteDbReads"] = logRemoteDbReads; // ParseBool(config, "logRemoteDbReads", "LOG_REMOTE_DB_READS", logRemoteDbReads, false);
    j["logExecutorServerInput"] = logExecutorServerInput; // ParseBool(config, "logExecutorServerInput", "LOG_EXECUTOR_SERVER_INPUT", logExecutorServerInput, false);
    j["logExecutorServerInputJson"] = logExecutorServerInputJson; // ParseBool(config, "logExecutorServerInputJson", "LOG_EXECUTOR_SERVER_INPUT_JSON", logExecutorServerInputJson, false);
    j["logExecutorServerInputGasThreshold"] = logExecutorServerInputGasThreshold; // ParseU64(config, "logExecutorServerInputGasThreshold", "LOG_EXECUTOR_SERVER_INPUT_GAS_THRESHOLD", logExecutorServerInputGasThreshold, 0);
    j["logExecutorServerResponses"] = logExecutorServerResponses; // ParseBool(config, "logExecutorServerResponses", "LOG_EXECUTOR_SERVER_RESPONSES", logExecutorServerResponses, false);
    j["logExecutorServerTxs"] = logExecutorServerTxs; // ParseBool(config, "logExecutorServerTxs", "LOG_EXECUTOR_SERVER_TXS", logExecutorServerTxs, true);
    j["dontLoadRomOffsets"] = dontLoadRomOffsets; // ParseBool(config, "dontLoadRomOffsets", "DONT_LOAD_ROM_OFFSETS", dontLoadRomOffsets, false);

    // Files and paths
    j["inputFile"] = inputFile; // ParseString(config, "inputFile", "INPUT_FILE", inputFile, "testvectors/batchProof/input_executor_0.json");
    j["inputFile2"] = inputFile2; // ParseString(config, "inputFile2", "INPUT_FILE_2", inputFile2, "");
    j["outputPath"] = outputPath; // ParseString(config, "outputPath", "OUTPUT_PATH", outputPath, "output");
    j["configPath"] = configPath; // ParseString(config, "configPath", "CONFIG_PATH", configPath, "config");
    j["rom"] = rom; // ParseString(config, "rom", "ROM", rom, string("src/main_sm/") + string(PROVER_FORK_NAMESPACE_STRING) + string("/scripts/rom.json"));
    j["keccakScriptFile"] = keccakScriptFile; // ParseString(config, "keccakScriptFile", "KECCAK_SCRIPT_FILE", keccakScriptFile, configPath + "/scripts/keccak_script.json");
    j["storageRomFile"] = storageRomFile; // ParseString(config, "storageRomFile", "STORAGE_ROM_FILE", storageRomFile, configPath + "/scripts/storage_sm_rom.json");
    j["zkevmConstPols"] = zkevmConstPols; // ParseString(config, "zkevmConstPols", "ZKEVM_CONST_POLS", zkevmConstPols, configPath + "/zkevm/zkevm.const");
    j["zkevmConstantsTree"] = zkevmConstantsTree; // ParseString(config, "zkevmConstantsTree", "ZKEVM_CONSTANTS_TREE", zkevmConstantsTree, configPath + "/zkevm/zkevm.consttree");
    j["zkevmStarkInfo"] = zkevmStarkInfo; // ParseString(config, "zkevmStarkInfo", "ZKEVM_STARK_INFO", zkevmStarkInfo, configPath + "/zkevm/zkevm.starkinfo.json");
    j["zkevmVerifier"] = zkevmVerifier; // ParseString(config, "zkevmVerifier", "ZKEVM_VERIFIER", zkevmVerifier, configPath + "/zkevm/zkevm.verifier.dat");
    j["c12aConstPols"] = c12aConstPols; // ParseString(config, "c12aConstPols", "C12A_CONST_POLS", c12aConstPols, configPath + "/c12a/c12a.const");
    j["c12aConstantsTree"] = c12aConstantsTree; // ParseString(config, "c12aConstantsTree", "C12A_CONSTANTS_TREE", c12aConstantsTree, configPath + "/c12a/c12a.consttree");
    j["c12aExec"] = c12aExec; // ParseString(config, "c12aExec", "C12A_EXEC", c12aExec, configPath + "/c12a/c12a.exec");
    j["c12aStarkInfo"] = c12aStarkInfo; // ParseString(config, "c12aStarkInfo", "C12A_STARK_INFO", c12aStarkInfo, configPath + "/c12a/c12a.starkinfo.json");
    j["recursive1ConstPols"] = recursive1ConstPols; // ParseString(config, "recursive1ConstPols", "RECURSIVE1_CONST_POLS", recursive1ConstPols, configPath + "/recursive1/recursive1.const");
    j["recursive1ConstantsTree"] = recursive1ConstantsTree; // ParseString(config, "recursive1ConstantsTree", "RECURSIVE1_CONSTANTS_TREE", recursive1ConstantsTree, configPath + "/recursive1/recursive1.consttree");
    j["recursive1Exec"] = recursive1Exec; // ParseString(config, "recursive1Exec", "RECURSIVE1_EXEC", recursive1Exec, configPath + "/recursive1/recursive1.exec");
    j["recursive1StarkInfo"] = recursive1StarkInfo; // ParseString(config, "recursive1StarkInfo", "RECURSIVE1_STARK_INFO", recursive1StarkInfo, configPath + "/recursive1/recursive1.starkinfo.json");
    j["recursive1Verifier"] = recursive1Verifier; // ParseString(config, "recursive1Verifier", "RECURSIVE1_VERIFIER", recursive1Verifier, configPath + "/recursive1/recursive1.verifier.dat");
    j["recursive2ConstPols"] = recursive2ConstPols; // ParseString(config, "recursive2ConstPols", "RECURSIVE2_CONST_POLS", recursive2ConstPols, configPath + "/recursive2/recursive2.const");
    j["recursive2ConstantsTree"] = recursive2ConstantsTree; // ParseString(config, "recursive2ConstantsTree", "RECURSIVE2_CONSTANTS_TREE", recursive2ConstantsTree, configPath + "/recursive2/recursive2.consttree");
    j["recursive2Exec"] = recursive2Exec; // ParseString(config, "recursive2Exec", "RECURSIVE2_EXEC", recursive2Exec, configPath + "/recursive2/recursive2.exec");
    j["recursive2StarkInfo"] = recursive2StarkInfo; // ParseString(config, "recursive2StarkInfo", "RECURSIVE2_STARK_INFO", recursive2StarkInfo, configPath + "/recursive2/recursive2.starkinfo.json");
    j["recursive2Verifier"] = recursive2Verifier; // ParseString(config, "recursive2Verifier", "RECURSIVE2_VERIFIER", recursive2Verifier, configPath + "/recursive2/recursive2.verifier.dat");
    j["recursive2Verkey"] = recursive2Verkey; // ParseString(config, "recursive2Verkey", "RECURSIVE2_VERKEY", recursive2Verkey, configPath + "/recursive2/recursive2.verkey.json");
    j["recursivefConstPols"] = recursivefConstPols; // ParseString(config, "recursivefConstPols", "RECURSIVEF_CONST_POLS", recursivefConstPols, configPath + "/recursivef/recursivef.const");
    j["recursivefConstantsTree"] = recursivefConstantsTree; // ParseString(config, "recursivefConstantsTree", "RECURSIVEF_CONSTANTS_TREE", recursivefConstantsTree, configPath + "/recursivef/recursivef.consttree");
    j["recursivefExec"] = recursivefExec; // ParseString(config, "recursivefExec", "RECURSIVEF_EXEC", recursivefExec, configPath + "/recursivef/recursivef.exec");
    j["recursivefStarkInfo"] = recursivefStarkInfo; // ParseString(config, "recursivefStarkInfo", "RECURSIVEF_STARK_INFO", recursivefStarkInfo, configPath + "/recursivef/recursivef.starkinfo.json");
    j["recursivefVerifier"] = recursivefVerifier; // ParseString(config, "recursivefVerifier", "RECURSIVEF_VERIFIER", recursivefVerifier, configPath + "/recursivef/recursivef.verifier.dat");
    j["finalVerifier"] = finalVerifier; // ParseString(config, "finalVerifier", "FINAL_VERIFIER", finalVerifier, configPath + "/final/final.verifier.dat");
    j["finalVerkey"] = finalVerkey; // ParseString(config, "finalVerkey", "FINAL_VERKEY", finalVerkey, configPath + "/final/final.fflonk.verkey.json");
    j["finalStarkZkey"] = finalStarkZkey; // ParseString(config, "finalStarkZkey", "FINAL_STARK_ZKEY", finalStarkZkey, configPath + "/final/final.fflonk.zkey");
    j["zkevmCmPols"] = zkevmCmPols; // ParseString(config, "zkevmCmPols", "ZKEVM_CM_POLS", zkevmCmPols, "");
    j["zkevmCmPolsAfterExecutor"] = zkevmCmPolsAfterExecutor; // ParseString(config, "zkevmCmPolsAfterExecutor", "ZKEVM_CM_POLS_AFTER_EXECUTOR", zkevmCmPolsAfterExecutor, "");
    j["c12aCmPols"] = c12aCmPols; // ParseString(config, "c12aCmPols", "C12A_CM_POLS", c12aCmPols, "");
    j["recursive1CmPols"] = recursive1CmPols; // ParseString(config, "recursive1CmPols", "RECURSIVE1_CM_POLS", recursive1CmPols, "");
    j["mapConstPolsFile"] = mapConstPolsFile; // ParseBool(config, "mapConstPolsFile", "MAP_CONST_POLS_FILE", mapConstPolsFile, false);
    j["mapConstantsTreeFile"] = mapConstantsTreeFile; // ParseBool(config, "mapConstantsTreeFile", "MAP_CONSTANTS_TREE_FILE", mapConstantsTreeFile, false);
    j["proofFile"] = proofFile; // ParseString(config, "proofFile", "PROOF_FILE", proofFile, "proof.json");
    j["publicsOutput"] = publicsOutput; // ParseString(config, "publicsOutput", "PUBLICS_OUTPUT", publicsOutput, "public.json");
    j["keccakPolsFile"] = keccakPolsFile; // ParseString(config, "keccakPolsFile", "KECCAK_POLS_FILE", keccakPolsFile, "keccak_pols.json");
    j["keccakConnectionsFile"] = keccakConnectionsFile; // ParseString(config, "keccakConnectionsFile", "KECCAK_CONNECTIONS_FILE", keccakConnectionsFile, "keccak_connections.json");

    // Database
    j["databaseURL"] = databaseURL; // ParseString(config, "databaseURL", "DATABASE_URL", databaseURL, "local");
    j["dbNodesTableName"] = dbNodesTableName; // ParseString(config, "dbNodesTableName", "DB_NODES_TABLE_NAME", dbNodesTableName, "state.nodes");
    j["dbProgramTableName"] = dbProgramTableName; // ParseString(config, "dbProgramTableName", "DB_PROGRAM_TABLE_NAME", dbProgramTableName, "state.program");
    j["dbMultiWrite"] = dbMultiWrite; // ParseBool(config, "dbMultiWrite", "DB_MULTIWRITE", dbMultiWrite, true);
    j["dbMultiWriteSingleQuerySize"] = dbMultiWriteSingleQuerySize; // ParseU64(config, "dbMultiWriteSingleQuerySize", "DB_MULTIWRITE_SINGLE_QUERY_SIZE", dbMultiWriteSingleQuerySize, 20*1024*1024);
    j["dbConnectionsPool"] = dbConnectionsPool; // ParseBool(config, "dbConnectionsPool", "DB_CONNECTIONS_POOL", dbConnectionsPool, true);
    j["dbNumberOfPoolConnections"] = dbNumberOfPoolConnections; // ParseU64(config, "dbNumberOfPoolConnections", "DB_NUMBER_OF_POOL_CONNECTIONS", dbNumberOfPoolConnections, 30);
    j["dbMetrics"] = dbMetrics; // ParseBool(config, "dbMetrics", "DB_METRICS", dbMetrics, true);
    j["dbClearCache"] = dbClearCache; // ParseBool(config, "dbClearCache", "DB_CLEAR_CACHE", dbClearCache, false);
    j["dbGetTree"] = dbGetTree; // ParseBool(config, "dbGetTree", "DB_GET_TREE", dbGetTree, true);
    j["dbReadOnly"] = dbReadOnly; // ParseBool(config, "dbReadOnly", "DB_READ_ONLY", dbReadOnly, false);
    j["dbReadRetryCounter"] = dbReadRetryCounter; // ParseU64(config, "dbReadRetryCounter", "DB_READ_RETRY_COUNTER", dbReadRetryCounter, 10);
    j["dbReadRetryDelay"] = dbReadRetryDelay; // ParseU64(config, "dbReadRetryDelay", "DB_READ_RETRY_DELAY", dbReadRetryDelay, 100*1000);

    // State Manager
    j["stateManager"] = stateManager; // ParseBool(config, "stateManager", "STATE_MANAGER", stateManager, true);
    j["stateManagerPurge"] = stateManagerPurge; // ParseBool(config, "stateManagerPurge", "STATE_MANAGER_PURGE", stateManagerPurge, true);
    j["stateManagerPurgeTxs"] = stateManagerPurgeTxs; // ParseBool(config, "stateManagerPurgeTxs", "STATE_MANAGER_PURGE_TXS", stateManagerPurgeTxs, true);

    // Threads
    j["cleanerPollingPeriod"] = cleanerPollingPeriod; // ParseU64(config, "cleanerPollingPeriod", "CLEANER_POLLING_PERIOD", cleanerPollingPeriod, 600);
    j["requestsPersistence"] = requestsPersistence; // ParseU64(config, "requestsPersistence", "REQUESTS_PERSISTENCE", requestsPersistence, 3600);
    j["maxExecutorThreads"] = maxExecutorThreads; // ParseU64(config, "maxExecutorThreads", "MAX_EXECUTOR_THREADS", maxExecutorThreads, 20);
    j["maxProverThreads"] = maxProverThreads; // ParseU64(config, "maxProverThreads", "MAX_PROVER_THREADS", maxProverThreads, 8);
    j["maxHashDBThreads"] = maxHashDBThreads; // ParseU64(config, "maxHashDBThreads", "MAX_HASHDB_THREADS", maxHashDBThreads, 8);

    // Prover name, name of this instance as per configuration
    j["proverName"] = proverName; // ParseString(config, "proverName", "PROVER_NAME", proverName, "UNSPECIFIED");

    // Memory allocation
    j["fullTracerTraceReserveSize"] = fullTracerTraceReserveSize; // ParseU64(config, "fullTracerTraceReserveSize", "FULL_TRACER_TRACE_RESERVE_SIZE", fullTracerTraceReserveSize, 256*1024);

    // ECRecover
    j["ECRecoverPrecalc"] = ECRecoverPrecalc; // ParseBool(config, "ECRecoverPrecalc", "ECRECOVER_PRECALC", ECRecoverPrecalc, false);
    j["ECRecoverPrecalcNThreads"] = ECRecoverPrecalcNThreads; // ParseU64(config, "ECRecoverPrecalcNThreads", "ECRECOVER_PRECALC_N_THREADS", ECRecoverPrecalcNThreads, 16);

    return j;
}
