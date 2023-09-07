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
    ParseBool(config, "runSMT64Test", "RUN_SMT64_TEST", runSMT64Test, false);
    ParseBool(config, "runDbKVRemoteTest", "RUN_DBKV_REMOTE_TEST", runDbKVRemoteTest, false);
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
    ParseU64(config, "kvDBMaxVersions", "HASHDB64_MAX_VERSIONS", kvDBMaxVersions, 131072);
    ParseString(config, "dbCacheSynchURL", "DB_CACHE_SYNCH_URL", dbCacheSynchURL, "");
    ParseU16(config, "aggregatorServerPort", "AGGREGATOR_SERVER_PORT", aggregatorServerPort, 50081);
    ParseU16(config, "aggregatorClientPort", "AGGREGATOR_CLIENT_PORT", aggregatorClientPort, 50081);
    ParseString(config, "aggregatorClientHost", "AGGREGATOR_CLIENT_HOST", aggregatorClientHost, "127.0.0.1");
    ParseU64(config, "aggregatorClientMockTimeout", "AGGREGATOR_CLIENT_MOCK_TIMEOUT", aggregatorClientMockTimeout, 60 * 1000 * 1000);
    ParseU64(config, "aggregatorClientWatchdogTimeout", "AGGREGATOR_CLIENT_WATCHDOG_TIMEOUT", aggregatorClientWatchdogTimeout, 60 * 1000 * 1000);
    ParseU64(config, "aggregatorClientMaxStreams", "AGGREGATOR_CLIENT_MAX_STREAMS", aggregatorClientMaxStreams, 0);

    // MT cache
    ParseS64(config, "dbMTCacheSize", "DB_MT_CACHE_SIZE", dbMTCacheSize, 8*1024); // Default = 8 GB

        
    runFileGenBatchProof = false;
    if (config.contains("runFileGenBatchProof") && config["runFileGenBatchProof"].is_boolean())
        runFileGenBatchProof = config["runFileGenBatchProof"];

    runFileGenAggregatedProof = false;
    if (config.contains("runFileGenAggregatedProof") && config["runFileGenAggregatedProof"].is_boolean())
        runFileGenAggregatedProof = config["runFileGenAggregatedProof"];

    runFileGenFinalProof = false;
    if (config.contains("runFileGenFinalProof") && config["runFileGenFinalProof"].is_boolean())
        runFileGenFinalProof = config["runFileGenFinalProof"];

    runFileProcessBatch = false;
    if (config.contains("runFileProcessBatch") && config["runFileProcessBatch"].is_boolean())
        runFileProcessBatch = config["runFileProcessBatch"];

    runFileProcessBatchMultithread = false;
    if (config.contains("runFileProcessBatchMultithread") && config["runFileProcessBatchMultithread"].is_boolean())
        runFileProcessBatchMultithread = config["runFileProcessBatchMultithread"];

    runFileExecute = false;
    if (config.contains("runFileExecute") && config["runFileExecute"].is_boolean())
        runFileExecute = config["runFileExecute"];


    runKeccakScriptGenerator = false;
    if (config.contains("runKeccakScriptGenerator") && config["runKeccakScriptGenerator"].is_boolean())
        runKeccakScriptGenerator = config["runKeccakScriptGenerator"];

    runSHA256ScriptGenerator = false;
    if (config.contains("runSHA256ScriptGenerator") && config["runSHA256ScriptGenerator"].is_boolean())
        runSHA256ScriptGenerator = config["runSHA256ScriptGenerator"];

    runKeccakTest = false;
    if (config.contains("runKeccakTest") && config["runKeccakTest"].is_boolean())
        runKeccakTest = config["runKeccakTest"];

    runStorageSMTest = false;
    if (config.contains("runStorageSMTest") && config["runStorageSMTest"].is_boolean())
        runStorageSMTest = config["runStorageSMTest"];

    runBinarySMTest = false;
    if (config.contains("runBinarySMTest") && config["runBinarySMTest"].is_boolean())
        runBinarySMTest = config["runBinarySMTest"];

    runMemAlignSMTest = false;
    if (config.contains("runMemAlignSMTest") && config["runMemAlignSMTest"].is_boolean())
        runMemAlignSMTest = config["runMemAlignSMTest"];

    runSHA256Test = false;
    if (config.contains("runSHA256Test") && config["runSHA256Test"].is_boolean())
        runSHA256Test = config["runSHA256Test"];

    runBlakeTest = false;
    if (config.contains("runBlakeTest") && config["runBlakeTest"].is_boolean())
        runBlakeTest = config["runBlakeTest"];

    runECRecoverTest = false;
    if (config.contains("runECRecoverTest") && config["runECRecoverTest"].is_boolean())
        runECRecoverTest = config["runECRecoverTest"];

    runDatabaseCacheTest = false;
    if (config.contains("runDatabaseCacheTest") && config["runDatabaseCacheTest"].is_boolean())
        runDatabaseCacheTest = config["runDatabaseCacheTest"];
    
    runDatabaseAssociativeCacheTest = false;
    if (config.contains("runDatabaseAssociativeCacheTest") && config["runDatabaseAssociativeCacheTest"].is_boolean())
        runDatabaseAssociativeCacheTest = config["runDatabaseAssociativeCacheTest"];

    runCheckTreeTest = false;
    if (config.contains("runCheckTreeTest") && config["runCheckTreeTest"].is_boolean())
        runCheckTreeTest = config["runCheckTreeTest"];

    checkTreeRoot = "auto";
    if (config.contains("checkTreeRoot") && config["checkTreeRoot"].is_string())
        checkTreeRoot = config["checkTreeRoot"];

    runDatabasePerformanceTest = false;
    if (config.contains("runDatabasePerformanceTest") && config["runDatabasePerformanceTest"].is_boolean())
        runDatabasePerformanceTest = config["runDatabasePerformanceTest"];

    runUnitTest = false;
    if (config.contains("runUnitTest") && config["runUnitTest"].is_boolean())
        runUnitTest = config["runUnitTest"];

    useMainExecGenerated = false;
    if (config.contains("useMainExecGenerated") && config["useMainExecGenerated"].is_boolean())
        useMainExecGenerated = config["useMainExecGenerated"];

    useMainExecC = false;
    if (config.contains("useMainExecC") && config["useMainExecC"].is_boolean())
        useMainExecC = config["useMainExecC"];

    executeInParallel = false;
    if (config.contains("executeInParallel") && config["executeInParallel"].is_boolean())
        executeInParallel = config["executeInParallel"];

    saveDbReadsToFile = false;
    if (config.contains("saveDbReadsToFile") && config["saveDbReadsToFile"].is_boolean())
        saveDbReadsToFile = config["saveDbReadsToFile"];

    saveRequestToFile = false;
    if (config.contains("saveRequestToFile") && config["saveRequestToFile"].is_boolean())
        saveRequestToFile = config["saveRequestToFile"];

    saveDbReadsToFileOnChange = false;
    if (config.contains("saveDbReadsToFileOnChange") && config["saveDbReadsToFileOnChange"].is_boolean())
        saveDbReadsToFileOnChange = config["saveDbReadsToFileOnChange"];

    saveInputToFile = false;
    if (config.contains("saveInputToFile") && config["saveInputToFile"].is_boolean())
        saveInputToFile = config["saveInputToFile"];

    saveResponseToFile = false;
    if (config.contains("saveResponseToFile") && config["saveResponseToFile"].is_boolean())
        saveResponseToFile = config["saveResponseToFile"];

    saveOutputToFile = false;
    if (config.contains("saveOutputToFile") && config["saveOutputToFile"].is_boolean())
        saveOutputToFile = config["saveOutputToFile"];

    saveProofToFile = false;
    if (config.contains("saveProofToFile") && config["saveProofToFile"].is_boolean())
        saveProofToFile = config["saveProofToFile"];

    saveFilesInSubfolders = false;
    if (config.contains("saveFilesInSubfolders") && config["saveFilesInSubfolders"].is_boolean())
        saveFilesInSubfolders = config["saveFilesInSubfolders"];

    loadDBToMemCache = false;
    if (config.contains("loadDBToMemCache") && config["loadDBToMemCache"].is_boolean())
        loadDBToMemCache = config["loadDBToMemCache"];

    loadDBToMemCacheInParallel = false;
    if (config.contains("loadDBToMemCacheInParallel") && config["loadDBToMemCacheInParallel"].is_boolean())
        loadDBToMemCacheInParallel = config["loadDBToMemCacheInParallel"];

    loadDBToMemTimeout = 30*1000*1000; // Default = 30 seconds
    if (config.contains("loadDBToMemTimeout") && config["loadDBToMemTimeout"].is_number())
        loadDBToMemTimeout = config["loadDBToMemTimeout"];

    dbMTCacheSize = 4*1024;
    if (config.contains("dbMTCacheSize") && config["dbMTCacheSize"].is_number())
        dbMTCacheSize = config["dbMTCacheSize"];
    
    useAssociativeCache = false;
    if (config.contains("useAssociativeCache") && config["useAssociativeCache"].is_boolean())
        useAssociativeCache = config["useAssociativeCache"];

    log2DbMTAssociativeCacheSize = 25;
    if (config.contains("log2DbMTAssociativeCacheSize") && config["log2DbMTAssociativeCacheSize"].is_number())
        log2DbMTAssociativeCacheSize = config["log2DbMTAssociativeCacheSize"];
    
    log2DbMTAssociativeCacheIndexesSize = 28;
    if (config.contains("log2DbMTAssociativeCacheIndexesSize") && config["log2DbMTAssociativeCacheIndexesSize"].is_number())
        log2DbMTAssociativeCacheIndexesSize = config["log2DbMTAssociativeCacheIndexesSize"];

    log2DbKVAssociativeCacheSize = 25;
    if (config.contains("log2DbKVAssociativeCacheSize") && config["log2DbKVAssociativeCacheSize"].is_number())
        log2DbKVAssociativeCacheSize = config["log2DbKVAssociativeCacheSize"];
    
    log2DbKVAssociativeCacheIndexesSize = 28;
    if (config.contains("log2DbKVAssociativeCacheIndexesSize") && config["log2DbKVAssociativeCacheIndexesSize"].is_number())
        log2DbKVAssociativeCacheIndexesSize = config["log2DbKVAssociativeCacheIndexesSize"];

    dbProgramCacheSize = 1*1024;
    if (config.contains("dbProgramCacheSize") && config["dbProgramCacheSize"].is_number())
        dbProgramCacheSize = config["dbProgramCacheSize"];

    opcodeTracer = false;
    if (config.contains("opcodeTracer") && config["opcodeTracer"].is_boolean())
        opcodeTracer = config["opcodeTracer"];

    logRemoteDbReads = false;
    if (config.contains("logRemoteDbReads") && config["logRemoteDbReads"].is_boolean())
        logRemoteDbReads = config["logRemoteDbReads"];

    logExecutorServerInput = false;
    if (config.contains("logExecutorServerInput") && config["logExecutorServerInput"].is_boolean())
        logExecutorServerInput = config["logExecutorServerInput"];

    logExecutorServerInputJson = false;
    if (config.contains("logExecutorServerInputJson") && config["logExecutorServerInputJson"].is_boolean())
        logExecutorServerInputJson = config["logExecutorServerInputJson"];

    logExecutorServerInputGasThreshold = 0;
    if (config.contains("logExecutorServerInputGasThreshold") && config["logExecutorServerInputGasThreshold"].is_number())
        logExecutorServerInputGasThreshold = config["logExecutorServerInputGasThreshold"];

    logExecutorServerResponses = false;
    if (config.contains("logExecutorServerResponses") && config["logExecutorServerResponses"].is_boolean())
        logExecutorServerResponses = config["logExecutorServerResponses"];

    logExecutorServerTxs = true;
    if (config.contains("logExecutorServerTxs") && config["logExecutorServerTxs"].is_boolean())
        logExecutorServerTxs = config["logExecutorServerTxs"];

    dontLoadRomOffsets = false;
    if (config.contains("dontLoadRomOffsets") && config["dontLoadRomOffsets"].is_boolean())
        dontLoadRomOffsets = config["dontLoadRomOffsets"];

    executorServerPort = 50071;
    if (config.contains("executorServerPort") && config["executorServerPort"].is_number())
        executorServerPort = config["executorServerPort"];

    executorROMLineTraces = false;
    if (config.contains("executorROMLineTraces") && config["executorROMLineTraces"].is_boolean())
        executorROMLineTraces = config["executorROMLineTraces"];

    executorTimeStatistics = false;
    if (config.contains("executorTimeStatistics") && config["executorTimeStatistics"].is_boolean())
        executorTimeStatistics = config["executorTimeStatistics"];

    executorClientPort = 50071;
    if (config.contains("executorClientPort") && config["executorClientPort"].is_number())
        executorClientPort = config["executorClientPort"];

    executorClientHost = "127.0.0.1";
    if (config.contains("executorClientHost") && config["executorClientHost"].is_string())
        executorClientHost = config["executorClientHost"];

    executorClientLoops = 1;
    if (config.contains("executorClientLoops") && config["executorClientLoops"].is_number())
        executorClientLoops = config["executorClientLoops"];

    executorClientCheckNewStateRoot = false;
    if (config.contains("executorClientCheckNewStateRoot") && config["executorClientCheckNewStateRoot"].is_boolean())
        executorClientCheckNewStateRoot = config["executorClientCheckNewStateRoot"];

    hashDBServerPort = 50061;
    if (config.contains("hashDBServerPort") && config["hashDBServerPort"].is_number())
        hashDBServerPort = config["hashDBServerPort"];

    hashDBURL = "local";
    if (config.contains("hashDBURL") && config["hashDBURL"].is_string())
        hashDBURL = config["hashDBURL"];

    hashDB64 = false;
    if (config.contains("hashDB64") && config["hashDB64"].is_boolean())
        hashDB64 = config["hashDB64"];

    dbCacheSynchURL = "";
    if (config.contains("dbCacheSynchURL") && config["dbCacheSynchURL"].is_string())
        dbCacheSynchURL = config["dbCacheSynchURL"];

    aggregatorServerPort = 50071;
    if (config.contains("aggregatorServerPort") && config["aggregatorServerPort"].is_number())
        aggregatorServerPort = config["aggregatorServerPort"];

    aggregatorClientPort = 50071;
    if (config.contains("aggregatorClientPort") && config["aggregatorClientPort"].is_number())
        aggregatorClientPort = config["aggregatorClientPort"];

    aggregatorClientHost = "127.0.0.1";
    if (config.contains("aggregatorClientHost") && config["aggregatorClientHost"].is_string())
        aggregatorClientHost = config["aggregatorClientHost"];

    aggregatorClientMockTimeout = 60 * 1000 * 1000;
    if (config.contains("aggregatorClientMockTimeout") && config["aggregatorClientMockTimeout"].is_number())
        aggregatorClientMockTimeout = config["aggregatorClientMockTimeout"];

    aggregatorClientWatchdogTimeout = 60 * 1000 * 1000;
    if (config.contains("aggregatorClientWatchdogTimeout") && config["aggregatorClientWatchdogTimeout"].is_number())
        aggregatorClientWatchdogTimeout = config["aggregatorClientWatchdogTimeout"];

    aggregatorClientMaxStreams = 0;
    if (config.contains("aggregatorClientMaxStreams") && config["aggregatorClientMaxStreams"].is_number())
        aggregatorClientMaxStreams = config["aggregatorClientMaxStreams"];

    if (config.contains("inputFile") && config["inputFile"].is_string())
        inputFile = config["inputFile"];
    if (config.contains("inputFile2") && config["inputFile2"].is_string())
        inputFile2 = config["inputFile2"];

    if (config.contains("outputPath") && config["outputPath"].is_string())
        outputPath = config["outputPath"];

    // Set default config path, and update it if specified
    configPath = "config";
    if (config.contains("configPath") && config["configPath"].is_string())
        configPath = config["configPath"];

    // Set default config files names
    rom = string("src/main_sm/") + string(PROVER_FORK_NAMESPACE_STRING) + string("/scripts/rom.json");
    keccakScriptFile = configPath + "/scripts/keccak_script.json";
    keccakPolsFile = configPath + "/scripts/keccak_pols.json";
    keccakConnectionsFile = configPath + "/scripts/keccak_connections.json";
    sha256ScriptFile = configPath + "/scripts/sha256_script.json";
    sha256PolsFile = configPath + "/scripts/sha256_pols.json";
    sha256ConnectionsFile = configPath + "/scripts/sha256_connections.json";
    storageRomFile = configPath + "/scripts/storage_sm_rom.json";
    zkevmConstPols = configPath + "/zkevm/zkevm.const";
    zkevmConstantsTree = configPath + "/zkevm/zkevm.consttree";
    zkevmStarkInfo = configPath + "/zkevm/zkevm.starkinfo.json";
    zkevmVerifier = configPath + "/zkevm/zkevm.verifier.dat";
    c12aConstPols = configPath + "/c12a/c12a.const";
    c12aConstantsTree = configPath + "/c12a/c12a.consttree";
    c12aExec = configPath + "/c12a/c12a.exec";
    c12aStarkInfo = configPath + "/c12a/c12a.starkinfo.json";
    recursive1ConstPols = configPath + "/recursive1/recursive1.const";
    recursive1ConstantsTree = configPath + "/recursive1/recursive1.consttree";
    recursive1Exec = configPath + "/recursive1/recursive1.exec";
    recursive1StarkInfo = configPath + "/recursive1/recursive1.starkinfo.json";
    recursive1Verifier = configPath + "/recursive1/recursive1.verifier.dat";
    recursive2ConstPols = configPath + "/recursive2/recursive2.const";
    recursive2ConstantsTree = configPath + "/recursive2/recursive2.consttree";
    recursive2Exec = configPath + "/recursive2/recursive2.exec";
    recursive2StarkInfo = configPath + "/recursive2/recursive2.starkinfo.json";
    recursive2Verifier = configPath + "/recursive2/recursive2.verifier.dat";
    recursive2Verkey = configPath + "/recursive2/recursive2.verkey.json";
    recursivefConstPols = configPath + "/recursivef/recursivef.const";
    recursivefConstantsTree = configPath + "/recursivef/recursivef.consttree";
    recursivefExec = configPath + "/recursivef/recursivef.exec";
    recursivefStarkInfo = configPath + "/recursivef/recursivef.starkinfo.json";
    recursivefVerifier = configPath + "/recursivef/recursivef.verifier.dat";
    finalVerifier = configPath + "/final/final.verifier.dat";
    finalVerkey = configPath + "/final/final.fflonk.verkey.json";
    finalStarkZkey = configPath + "/final/final.fflonk.zkey";


    if (config.contains("rom") && config["rom"].is_string())
        rom = config["rom"];

    if (config.contains("zkevmCmPols") && config["zkevmCmPols"].is_string())
        zkevmCmPols = config["zkevmCmPols"];

    if (config.contains("zkevmCmPolsAfterExecutor") && config["zkevmCmPolsAfterExecutor"].is_string())
        zkevmCmPolsAfterExecutor = config["zkevmCmPolsAfterExecutor"];

    if (config.contains("c12aCmPols") && config["c12aCmPols"].is_string())
        c12aCmPols = config["c12aCmPols"];

    if (config.contains("recursive1CmPols") && config["recursive1CmPols"].is_string())
        recursive1CmPols = config["recursive1CmPols"];

    if (config.contains("zkevmConstPols") && config["zkevmConstPols"].is_string())
        zkevmConstPols = config["zkevmConstPols"];

    if (config.contains("c12aConstPols") && config["c12aConstPols"].is_string())
        c12aConstPols = config["c12aConstPols"];

    if (config.contains("recursive1ConstPols") && config["recursive1ConstPols"].is_string())
        recursive1ConstPols = config["recursive1ConstPols"];
    
    if (config.contains("recursive2ConstPols") && config["recursive2ConstPols"].is_string())
        recursive2ConstPols = config["recursive2ConstPols"];
    
    if (config.contains("recursivefConstPols") && config["recursivefConstPols"].is_string())
        recursivefConstPols = config["recursivefConstPols"];

    mapConstPolsFile = true;
    if (config.contains("mapConstPolsFile") && config["mapConstPolsFile"].is_boolean())
        mapConstPolsFile = config["mapConstPolsFile"];

    if (config.contains("zkevmConstantsTree") && config["zkevmConstantsTree"].is_string())
        zkevmConstantsTree = config["zkevmConstantsTree"];

    if (config.contains("c12aConstantsTree") && config["c12aConstantsTree"].is_string())
        c12aConstantsTree = config["c12aConstantsTree"];

    if (config.contains("recursive1ConstantsTree") && config["recursive1ConstantsTree"].is_string())
        recursive1ConstantsTree = config["recursive1ConstantsTree"];

    if (config.contains("recursive2ConstantsTree") && config["recursive2ConstantsTree"].is_string())
        recursive2ConstantsTree = config["recursive2ConstantsTree"];

    if (config.contains("recursivefConstantsTree") && config["recursivefConstantsTree"].is_string())
        recursivefConstantsTree = config["recursivefConstantsTree"];

    mapConstantsTreeFile = true;
    if (config.contains("mapConstantsTreeFile") && config["mapConstantsTreeFile"].is_boolean())
        mapConstantsTreeFile = config["mapConstantsTreeFile"];

    if (config.contains("finalVerkey") && config["finalVerkey"].is_string())
        finalVerkey = config["finalVerkey"];

    if (config.contains("zkevmVerifier") && config["zkevmVerifier"].is_string())
        zkevmVerifier = config["zkevmVerifier"];

    if (config.contains("recursive1Verifier") && config["recursive1Verifier"].is_string())
        recursive1Verifier = config["recursive1Verifier"];

    if (config.contains("recursive2Verifier") && config["recursive2Verifier"].is_string())
        recursive2Verifier = config["recursive2Verifier"];
    
    if (config.contains("recursive2Verkey") && config["recursive2Verkey"].is_string())
        recursive2Verkey = config["recursive2Verkey"];

    if (config.contains("finalVerifier") && config["finalVerifier"].is_string())
        finalVerifier = config["finalVerifier"];

    if (config.contains("recursivefVerifier") && config["recursivefVerifier"].is_string())
        recursivefVerifier = config["recursivefVerifier"];

    if (config.contains("c12aExec") && config["c12aExec"].is_string())
        c12aExec = config["c12aExec"];

    if (config.contains("recursive1Exec") && config["recursive1Exec"].is_string())
        recursive1Exec = config["recursive1Exec"];
    
    if (config.contains("recursive2Exec") && config["recursive2Exec"].is_string())
        recursive2Exec = config["recursive2Exec"];
    
    if (config.contains("recursivefExec") && config["recursivefExec"].is_string())
        recursivefExec = config["recursivefExec"];

    if (config.contains("finalStarkZkey") && config["finalStarkZkey"].is_string())
        finalStarkZkey = config["finalStarkZkey"];

    if (config.contains("proofFile") && config["proofFile"].is_string())
        proofFile = config["proofFile"];

    if (config.contains("publicsOutput") && config["publicsOutput"].is_string())
        publicsOutput = config["publicsOutput"];

    if (config.contains("keccakScriptFile") && config["keccakScriptFile"].is_string())
        keccakScriptFile = config["keccakScriptFile"];

    if (config.contains("sha256ScriptFile") && config["sha256ScriptFile"].is_string())
        sha256ScriptFile = config["sha256ScriptFile"];

    if (config.contains("keccakPolsFile") && config["keccakPolsFile"].is_string())
        keccakPolsFile = config["keccakPolsFile"];

    if (config.contains("sha256PolsFile") && config["sha256PolsFile"].is_string())
        sha256PolsFile = config["sha256PolsFile"];

    if (config.contains("keccakConnectionsFile") && config["keccakConnectionsFile"].is_string())
        keccakConnectionsFile = config["keccakConnectionsFile"];

    if (config.contains("sha256ConnectionsFile") && config["sha256ConnectionsFile"].is_string())
        sha256ConnectionsFile = config["sha256ConnectionsFile"];

    if (config.contains("storageRomFile") && config["storageRomFile"].is_string())
        storageRomFile = config["storageRomFile"];

    if (config.contains("zkevmStarkInfo") && config["zkevmStarkInfo"].is_string())
        zkevmStarkInfo = config["zkevmStarkInfo"];

    if (config.contains("c12aStarkInfo") && config["c12aStarkInfo"].is_string())
        c12aStarkInfo = config["c12aStarkInfo"];

    if (config.contains("recursive1StarkInfo") && config["recursive1StarkInfo"].is_string())
        recursive1StarkInfo = config["recursive1StarkInfo"];

    if (config.contains("recursive2StarkInfo") && config["recursive2StarkInfo"].is_string())
        recursive2StarkInfo = config["recursive2StarkInfo"];

    if (config.contains("recursivefStarkInfo") && config["recursivefStarkInfo"].is_string())
        recursivefStarkInfo = config["recursivefStarkInfo"];

    if (config.contains("databaseURL") && config["databaseURL"].is_string())
        databaseURL = config["databaseURL"];

    if (config.contains("dbNodesTableName") && config["dbNodesTableName"].is_string())
        dbNodesTableName = config["dbNodesTableName"];

    if (config.contains("dbProgramTableName") && config["dbProgramTableName"].is_string())
        dbProgramTableName = config["dbProgramTableName"];

    dbMultiWrite = false;
    if (config.contains("dbMultiWrite") && config["dbMultiWrite"].is_boolean())
        dbMultiWrite = config["dbMultiWrite"];

    dbMultiWriteSingleQuerySize = 20*1024*1024;
    if (config.contains("dbMultiWriteSingleQuerySize") && config["dbMultiWriteSingleQuerySize"].is_number())
        dbMultiWriteSingleQuerySize = config["dbMultiWriteSingleQuerySize"];

    dbConnectionsPool = false;
    if (config.contains("dbConnectionsPool") && config["dbConnectionsPool"].is_boolean())
        dbConnectionsPool = config["dbConnectionsPool"];

    dbNumberOfPoolConnections = 25;
    if (config.contains("dbNumberOfPoolConnections") && config["dbNumberOfPoolConnections"].is_number())
        dbNumberOfPoolConnections = config["dbNumberOfPoolConnections"];

    dbMetrics = false;
    if (config.contains("dbMetrics") && config["dbMetrics"].is_boolean())
        dbMetrics = config["dbMetrics"];

    dbClearCache = false;
    if (config.contains("dbClearCache") && config["dbClearCache"].is_boolean())
        dbClearCache = config["dbClearCache"];

    dbGetTree = false;
    if (config.contains("dbGetTree") && config["dbGetTree"].is_boolean())
        dbGetTree = config["dbGetTree"];

    dbReadOnly = false;
    if (config.contains("dbReadOnly") && config["dbReadOnly"].is_boolean())
        dbReadOnly = config["dbReadOnly"];

    dbReadRetryCounter = 10;
    if (config.contains("dbReadRetryCounter") && config["dbReadRetryCounter"].is_number())
        dbReadRetryCounter = config["dbReadRetryCounter"];

    dbReadRetryDelay = 100*1000;
    if (config.contains("dbReadRetryDelay") && config["dbReadRetryDelay"].is_number())
        dbReadRetryDelay = config["dbReadRetryDelay"];

    stateManager = false;
    if (config.contains("stateManager") && config["stateManager"].is_boolean())
        stateManager = config["stateManager"];

    stateManagerPurge = true;
    if (config.contains("stateManagerPurge") && config["stateManagerPurge"].is_boolean())
        stateManagerPurge = config["stateManagerPurge"];

    if (config.contains("cleanerPollingPeriod") && config["cleanerPollingPeriod"].is_number())
        cleanerPollingPeriod = config["cleanerPollingPeriod"];

    if (config.contains("requestsPersistence") && config["requestsPersistence"].is_number())
        requestsPersistence = config["requestsPersistence"];

    maxExecutorThreads = 16;
    if (config.contains("maxExecutorThreads") && config["maxExecutorThreads"].is_number())
        maxExecutorThreads = config["maxExecutorThreads"];

    maxProverThreads = 16;
    if (config.contains("maxProverThreads") && config["maxProverThreads"].is_number())
        maxProverThreads = config["maxProverThreads"];

    maxHashDBThreads = 16;
    if (config.contains("maxHashDBThreads") && config["maxHashDBThreads"].is_number())
        maxHashDBThreads = config["maxHashDBThreads"];

    proverName = "UNSPECIFIED";
    if (config.contains("proverName") && config["proverName"].is_string())
        proverName = config["proverName"];

    fullTracerTraceReserveSize = 256*1024;
    if (config.contains("fullTracerTraceReserveSize") && config["fullTracerTraceReserveSize"].is_number())
        fullTracerTraceReserveSize = config["fullTracerTraceReserveSize"];

    ECRecoverPrecalc = true;
    if (config.contains("ECRecoverPrecalc") && config["ECRecoverPrecalc"].is_boolean())
        ECRecoverPrecalc = config["ECRecoverPrecalc"];

    ECRecoverPrecalcNThreads = 16;
    if (config.contains("ECRecoverPrecalcNThreads") && config["ECRecoverPrecalcNThreads"].is_number())
        ECRecoverPrecalcNThreads = config["ECRecoverPrecalcNThreads"];
    // MT associative cache
    ParseBool(config, "useAssociativeCache", "USE_ASSOCIATIVE_CACHE", useAssociativeCache, false);
    ParseS64(config, "log2DbMTAssociativeCacheSize", "LOG2_DB_MT_ASSOCIATIVE_CACHE_SIZE", log2DbMTAssociativeCacheSize, 25);
    ParseS64(config, "log2DbMTAssociativeCacheIndexesSize", "LOG2_DB_MT_ASSOCIATIVE_CACHE_INDEXES_SIZE", log2DbMTAssociativeCacheIndexesSize, 28);
    ParseS64(config, "log2DbKVAssociativeCacheSize", "LOG2_DB_KV_ASSOCIATIVE_CACHE_SIZE", log2DbKVAssociativeCacheSize, 25);
    ParseS64(config, "log2DbMTAssociativeCacheIndexesSize", "LOG2_DB_KV_ASSOCIATIVE_CACHE_INDEXES_SIZE", log2DbKVAssociativeCacheIndexesSize, 28);

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
    ParseString(config, "dbKeyValueTableName", "DB_KEYVALUE_TABLE_NAME", dbKeyValueTableName, "state.keyvalue");
    ParseString(config, "dbKeyVersionTableName", "DB_VERSION_TABLE_NAME", dbVersionTableName, "state.version");
    ParseString(config, "dbLatestVersionTableName", "DB_LATEST_VERSION_TABLE_NAME", dbLatestVersionTableName, "state.latestversion");
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

    zklog.info("    runExecutorServer=" + to_string(runExecutorServer));
    if (runExecutorClient)
        zklog.info("    runExecutorClient=true");
    if (runExecutorClientMultithread)
        zklog.info("    runExecutorClientMultithread=true");
    zklog.info("    runHashDBServer=" + to_string(runHashDBServer));
    if (runHashDBTest)
        zklog.info("    runHashDBTest=true");
    if (runAggregatorServer)
        zklog.info("    runAggregatorServer=true");
    zklog.info("    runAggregatorClient=" + to_string(runAggregatorClient));
    if (runAggregatorClientMock)        
        zklog.info("    runAggregatorClientMock=true");
    if (runFileGenBatchProof)
        zklog.info("    runFileGenBatchProof=true");
    if (runFileGenAggregatedProof)
        zklog.info("    runFileGenAggregatedProof=true");
    if (runFileGenFinalProof)
        zklog.info("    runFileGenFinalProof=true");
    if (runFileProcessBatch)
        zklog.info("    runFileProcessBatch=true");
    if (runFileProcessBatchMultithread)
        zklog.info("    runFileProcessBatchMultithread=true");
    if (runFileExecute)
        zklog.info("    runFileExecute=true");

    if (runKeccakScriptGenerator)
        zklog.info("    runKeccakScriptGenerator=true");
    if (runSHA256ScriptGenerator)
        zklog.info("    runSHA256ScriptGenerator=true");
    if (runKeccakTest)
        zklog.info("    runKeccakTest=true");
    if (runStorageSMTest)
        zklog.info("    runStorageSMTest=true");
    if (runBinarySMTest)
        zklog.info("    runBinarySMTest=true");
    if (runMemAlignSMTest)
        zklog.info("    runMemAlignSMTest=true");
    if (runSHA256Test)
        zklog.info("    runSHA256Test=true");
    if (runBlakeTest)
        zklog.info("    runBlakeTest=true");
    if (runECRecoverTest)
        zklog.info("    runECRecoverTest=true");
    if (runDatabaseCacheTest)
        zklog.info("    runDatabaseCacheTest=true");
    if (runDatabaseAssociativeCacheTest)
        zklog.info("    runDatabaseAssociativeCacheTest=true");
    if (runCheckTreeTest)
    {
        zklog.info("    runCheckTreeTest=true");
        zklog.info("    checkTreeRoot=" + checkTreeRoot);
    }
    if (runDatabasePerformanceTest)
        zklog.info("    runDatabasePerformanceTest=true");
    if (runSMT64Test)
        zklog.info("    runSMT64Test=true");
    if (runDbKVRemoteTest)
        zklog.info("    runDbKVRemoteTest=true");
    if (runUnitTest)
        zklog.info("    runUnitTest=true");

    zklog.info("    executeInParallel=" + to_string(executeInParallel));
    zklog.info("    useMainExecGenerated=" + to_string(useMainExecGenerated));
    zklog.info("    useMainExecC=" + to_string(useMainExecC));

    if (executorROMLineTraces)
        zklog.info("    executorROMLineTraces=true");

    zklog.info("    executorTimeStatistics=" + to_string(executorTimeStatistics));

    if (saveRequestToFile)
        zklog.info("    saveRequestToFile=true");
    if (saveInputToFile)
        zklog.info("    saveInputToFile=true");
    if (saveDbReadsToFile)
        zklog.info("    saveDbReadsToFile=true");
    if (saveDbReadsToFileOnChange)
        zklog.info("    saveDbReadsToFileOnChange=true");
    if (saveOutputToFile)
        zklog.info("    saveOutputToFile=true");
    if (saveProofToFile)
        zklog.info("    saveProofToFile=true");
    if (saveResponseToFile)
        zklog.info("    saveResponseToFile=true");
    zklog.info("    loadDBToMemCache=" + to_string(loadDBToMemCache));
    if (loadDBToMemCacheInParallel)
        zklog.info("    loadDBToMemCacheInParallel=true");
    if (opcodeTracer)
        zklog.info("    opcodeTracer=true");
    if (logRemoteDbReads)
        zklog.info("    logRemoteDbReads=true");
    if (logExecutorServerInput)
        zklog.info("    logExecutorServerInput=true");
    if (logExecutorServerInputJson)
        zklog.info("    logExecutorServerInputJson=true");
    if (logExecutorServerInputGasThreshold)
        zklog.info("    logExecutorServerInputGasThreshold=true");
    if (logExecutorServerResponses)
        zklog.info("    logExecutorServerResponses=true");
    if (logExecutorServerTxs)
        zklog.info("    logExecutorServerTxs=true");
    if (dontLoadRomOffsets)
        zklog.info("    dontLoadRomOffsets=true");

    zklog.info("    executorServerPort=" + to_string(executorServerPort));
    zklog.info("    executorClientPort=" + to_string(executorClientPort));
    zklog.info("    executorClientHost=" + executorClientHost);
    zklog.info("    executorClientLoops=" + to_string(executorClientLoops));
    zklog.info("    executorClientCheckNewStateRoot=" + to_string(executorClientCheckNewStateRoot));
    zklog.info("    hashDBServerPort=" + to_string(hashDBServerPort));
    zklog.info("    hashDBURL=" + hashDBURL);
    zklog.info("    hashDB64=" + to_string(hashDB64));
    zklog.info("    kvDBMaxVersions=" + to_string(kvDBMaxVersions));
    zklog.info("    dbCacheSynchURL=" + dbCacheSynchURL);
    zklog.info("    aggregatorServerPort=" + to_string(aggregatorServerPort));
    zklog.info("    aggregatorClientPort=" + to_string(aggregatorClientPort));
    zklog.info("    aggregatorClientHost=" + aggregatorClientHost);
    zklog.info("    aggregatorClientMockTimeout=" + to_string(aggregatorClientMockTimeout));
    zklog.info("    aggregatorClientWatchdogTimeout=" + to_string(aggregatorClientWatchdogTimeout));
    zklog.info("    aggregatorClientMaxStreams=" + to_string(aggregatorClientMaxStreams));

    zklog.info("    inputFile=" + inputFile);
    zklog.info("    inputFile2=" + inputFile2);
    zklog.info("    outputPath=" + outputPath);
    zklog.info("    configPath=" + configPath);
    zklog.info("    rom=" + rom);
    zklog.info("    zkevmCmPols=" + zkevmCmPols);
    zklog.info("    c12aCmPols=" + c12aCmPols);
    zklog.info("    recursive1CmPols=" + recursive1CmPols);
    zklog.info("    zkevmConstPols=" + zkevmConstPols);
    zklog.info("    c12aConstPols=" + c12aConstPols);
    zklog.info("    mapConstPolsFile=" + to_string(mapConstPolsFile));
    zklog.info("    zkevmConstantsTree=" + zkevmConstantsTree);
    zklog.info("    c12aConstantsTree=" + c12aConstantsTree);
    zklog.info("    mapConstantsTreeFile=" + to_string(mapConstantsTreeFile));
    zklog.info("    finalVerkey=" + finalVerkey);
    zklog.info("    zkevmVerifier=" + zkevmVerifier);
    zklog.info("    recursive1Verifier=" + recursive1Verifier);
    zklog.info("    recursive2Verifier=" + recursive2Verifier);
    zklog.info("    recursive2Verkey=" + recursive2Verkey);
    zklog.info("    recursivefVerifier=" + recursivefVerifier);
    zklog.info("    finalVerifier=" + finalVerifier);
    zklog.info("    finalStarkZkey=" + finalStarkZkey);
    zklog.info("    publicsOutput=" + publicsOutput);
    zklog.info("    proofFile=" + proofFile);
    zklog.info("    keccakScriptFile=" + keccakScriptFile);
    zklog.info("    sha256ScriptFile=" + sha256ScriptFile);
    zklog.info("    sha256ScriptFile=" + sha256ScriptFile);
    zklog.info("    keccakPolsFile=" + keccakPolsFile);
    zklog.info("    sha256PolsFile=" + sha256PolsFile);
    zklog.info("    sha256PolsFile=" + sha256PolsFile);
    zklog.info("    keccakConnectionsFile=" + keccakConnectionsFile);
    zklog.info("    storageRomFile=" + storageRomFile);
    zklog.info("    zkevmStarkInfo=" + zkevmStarkInfo);
    zklog.info("    c12aStarkInfo=" + c12aStarkInfo);
    zklog.info("    databaseURL=" + databaseURL.substr(0, 5) + "...");
    zklog.info("    dbNodesTableName=" + dbNodesTableName);
    zklog.info("    dbProgramTableName=" + dbProgramTableName);
    zklog.info("    dbKeyValueTableName=" + dbKeyValueTableName);
    zklog.info("    dbVersionTableName=" + dbVersionTableName);
    zklog.info("    dbLatestVersionTableName=" + dbLatestVersionTableName);
    zklog.info("    dbMultiWrite=" + to_string(dbMultiWrite));
    zklog.info("    dbMultiWriteSingleQuerySize=" + to_string(dbMultiWriteSingleQuerySize));
    zklog.info("    dbConnectionsPool=" + to_string(dbConnectionsPool));
    zklog.info("    dbNumberOfPoolConnections=" + to_string(dbNumberOfPoolConnections));
    zklog.info("    dbMetrics=" + to_string(dbMetrics));
    zklog.info("    dbClearCache=" + to_string(dbClearCache));
    zklog.info("    dbGetTree=" + to_string(dbGetTree));
    zklog.info("    dbReadOnly=" + to_string(dbReadOnly));
    zklog.info("    dbReadRetryCounter=" + to_string(dbReadRetryCounter));
    zklog.info("    dbReadRetryDelay=" + to_string(dbReadRetryDelay));
    zklog.info("    stateManager=" + to_string(stateManager));
    zklog.info("    stateManagerPurge=" + to_string(stateManagerPurge));
    zklog.info("    stateManagerPurgeTxs=" + to_string(stateManagerPurgeTxs));
    zklog.info("    cleanerPollingPeriod=" + to_string(cleanerPollingPeriod));
    zklog.info("    requestsPersistence=" + to_string(requestsPersistence));
    zklog.info("    maxExecutorThreads=" + to_string(maxExecutorThreads));
    zklog.info("    maxProverThreads=" + to_string(maxProverThreads));
    zklog.info("    maxHashDBThreads=" + to_string(maxHashDBThreads));
    zklog.info("    dbMTCacheSize=" + to_string(dbMTCacheSize));
    zklog.info("    useAssociativeCache=" + to_string(useAssociativeCache));
    zklog.info("    log2DbMTAssociativeCacheSize=" + to_string(log2DbMTAssociativeCacheSize));
    zklog.info("    log2DbMTAssociativeCacheIndexesSize=" + to_string(log2DbMTAssociativeCacheIndexesSize));
    zklog.info("    log2DbKVAssociativeCacheSize=" + to_string(log2DbKVAssociativeCacheSize));
    zklog.info("    log2DbKVAssociativeCacheIndexesSize=" + to_string(log2DbKVAssociativeCacheIndexesSize));
    zklog.info("    dbProgramCacheSize=" + to_string(dbProgramCacheSize));
    zklog.info("    loadDBToMemTimeout=" + to_string(loadDBToMemTimeout));
    zklog.info("    fullTracerTraceReserveSize=" + to_string(fullTracerTraceReserveSize));
    zklog.info("    ECRecoverPrecalc=" + to_string(ECRecoverPrecalc));
    zklog.info("    ECRecoverPrecalcNThreads=" + to_string(ECRecoverPrecalcNThreads));


}