#include <string>
#include <nlohmann/json.hpp>
#include "definitions.hpp"
#include "config.hpp"
#include "zkassert.hpp"
#include "utils.hpp"
#include "zklog.hpp"

using namespace std;
using json = nlohmann::json;

void Config::load(json &config)
{
    zkassert(proverID == "");
    proverID = getUUID();

    runExecutorServer = false;
    if (config.contains("runExecutorServer") && config["runExecutorServer"].is_boolean())
        runExecutorServer = config["runExecutorServer"];

    runExecutorClient = false;
    if (config.contains("runExecutorClient") && config["runExecutorClient"].is_boolean())
        runExecutorClient = config["runExecutorClient"];

    runExecutorClientMultithread = false;
    if (config.contains("runExecutorClientMultithread") && config["runExecutorClientMultithread"].is_boolean())
        runExecutorClientMultithread = config["runExecutorClientMultithread"];

    runHashDBServer = false;
    if (config.contains("runHashDBServer") && config["runHashDBServer"].is_boolean())
        runHashDBServer = config["runHashDBServer"];

    runHashDBTest = false;
    if (config.contains("runHashDBTest") && config["runHashDBTest"].is_boolean())
        runHashDBTest = config["runHashDBTest"];

    runAggregatorServer = false;
    if (config.contains("runAggregatorServer") && config["runAggregatorServer"].is_boolean())
        runAggregatorServer = config["runAggregatorServer"];

    runAggregatorClient = false;
    if (config.contains("runAggregatorClient") && config["runAggregatorClient"].is_boolean())
        runAggregatorClient = config["runAggregatorClient"];

    runAggregatorClientMock = false;
    if (config.contains("runAggregatorClientMock") && config["runAggregatorClientMock"].is_boolean())
        runAggregatorClientMock = config["runAggregatorClientMock"];

        
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

    log2DbMTAssociativeCacheSize = 20;
    if (config.contains("log2DbMTAssociativeCacheSize") && config["log2DbMTAssociativeCacheSize"].is_number())
        log2DbMTAssociativeCacheSize = config["log2DbMTAssociativeCacheSize"];
    
    log2DbMTAssociativeCacheIndicesSize = 23;
    if (config.contains("log2DbMTAssociativeCacheIndicesSize") && config["log2DbMTAssociativeCacheIndicesSize"].is_number())
        log2DbMTAssociativeCacheIndicesSize = config["log2DbMTAssociativeCacheIndicesSize"];

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

    if (config.contains("keccakPolsFile") && config["keccakPolsFile"].is_string())
        keccakPolsFile = config["keccakPolsFile"];

    if (config.contains("keccakConnectionsFile") && config["keccakConnectionsFile"].is_string())
        keccakConnectionsFile = config["keccakConnectionsFile"];

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
    zklog.info("    keccakPolsFile=" + keccakPolsFile);
    zklog.info("    keccakConnectionsFile=" + keccakConnectionsFile);
    zklog.info("    storageRomFile=" + storageRomFile);
    zklog.info("    zkevmStarkInfo=" + zkevmStarkInfo);
    zklog.info("    c12aStarkInfo=" + c12aStarkInfo);
    zklog.info("    databaseURL=" + databaseURL.substr(0, 5) + "...");
    zklog.info("    dbNodesTableName=" + dbNodesTableName);
    zklog.info("    dbProgramTableName=" + dbProgramTableName);
    zklog.info("    dbMultiWrite=" + to_string(dbMultiWrite));
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
    zklog.info("    cleanerPollingPeriod=" + to_string(cleanerPollingPeriod));
    zklog.info("    requestsPersistence=" + to_string(requestsPersistence));
    zklog.info("    maxExecutorThreads=" + to_string(maxExecutorThreads));
    zklog.info("    maxProverThreads=" + to_string(maxProverThreads));
    zklog.info("    maxHashDBThreads=" + to_string(maxHashDBThreads));
    zklog.info("    dbMTCacheSize=" + to_string(dbMTCacheSize));
    zklog.info("    useAssociativeCache=" + to_string(useAssociativeCache));
    zklog.info("    log2DbMTAssociativeCacheSize=" + to_string(log2DbMTAssociativeCacheSize));
    zklog.info("    log2DbMTAssociativeCacheIndicesSize=" + to_string(log2DbMTAssociativeCacheIndicesSize));
    zklog.info("    dbProgramCacheSize=" + to_string(dbProgramCacheSize));
    zklog.info("    loadDBToMemTimeout=" + to_string(loadDBToMemTimeout));
    zklog.info("    fullTracerTraceReserveSize=" + to_string(fullTracerTraceReserveSize));
    zklog.info("    ECRecoverPrecalc=" + to_string(ECRecoverPrecalc));
    zklog.info("    ECRecoverPrecalcNThreads=" + to_string(ECRecoverPrecalcNThreads));


}