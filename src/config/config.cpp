#include <string>
#include <nlohmann/json.hpp>
#include "definitions.hpp"
#include "config.hpp"
#include "zkassert.hpp"
#include "utils.hpp"

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

    runStateDBServer = false;
    if (config.contains("runStateDBServer") && config["runStateDBServer"].is_boolean())
        runStateDBServer = config["runStateDBServer"];

    runStateDBTest = false;
    if (config.contains("runStateDBTest") && config["runStateDBTest"].is_boolean())
        runStateDBTest = config["runStateDBTest"];

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

    useMainExecGenerated = false;
    if (config.contains("useMainExecGenerated") && config["useMainExecGenerated"].is_boolean())
        useMainExecGenerated = config["useMainExecGenerated"];

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

    dbMTCacheSize = 4*1024;
    if (config.contains("dbMTCacheSize") && config["dbMTCacheSize"].is_number())
        dbMTCacheSize = config["dbMTCacheSize"];

    dbProgramCacheSize = 1*1024;
    if (config.contains("dbProgramCacheSize") && config["dbProgramCacheSize"].is_number())
        dbProgramCacheSize = config["dbProgramCacheSize"];

    opcodeTracer = false;
    if (config.contains("opcodeTracer") && config["opcodeTracer"].is_boolean())
        opcodeTracer = config["opcodeTracer"];

    logRemoteDbReads = false;
    if (config.contains("logRemoteDbReads") && config["logRemoteDbReads"].is_boolean())
        logRemoteDbReads = config["logRemoteDbReads"];

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

    stateDBServerPort = 50061;
    if (config.contains("stateDBServerPort") && config["stateDBServerPort"].is_number())
        stateDBServerPort = config["stateDBServerPort"];

    stateDBURL = "local";
    if (config.contains("stateDBURL") && config["stateDBURL"].is_string())
        stateDBURL = config["stateDBURL"];

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

    if (config.contains("stateDBURL") && config["stateDBURL"].is_string())
        stateDBURL = config["stateDBURL"];

    if (config.contains("databaseURL") && config["databaseURL"].is_string())
        databaseURL = config["databaseURL"];

    if (config.contains("dbNodesTableName") && config["dbNodesTableName"].is_string())
        dbNodesTableName = config["dbNodesTableName"];

    if (config.contains("dbProgramTableName") && config["dbProgramTableName"].is_string())
        dbProgramTableName = config["dbProgramTableName"];

    dbMultiWrite = false;
    if (config.contains("dbMultiWrite") && config["dbMultiWrite"].is_boolean())
        dbMultiWrite = config["dbMultiWrite"];

    dbFlushInParallel = false;
    if (config.contains("dbFlushInParallel") && config["dbFlushInParallel"].is_boolean())
        dbFlushInParallel = config["dbFlushInParallel"];

    dbConnectionsPool = false;
    if (config.contains("dbConnectionsPool") && config["dbConnectionsPool"].is_boolean())
        dbConnectionsPool = config["dbConnectionsPool"];

    dbNumberOfPoolConnections = 25;
    if (config.contains("dbNumberOfPoolConnections") && config["dbNumberOfPoolConnections"].is_number())
        dbNumberOfPoolConnections = config["dbNumberOfPoolConnections"];

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

    maxStateDBThreads = 16;
    if (config.contains("maxStateDBThreads") && config["maxStateDBThreads"].is_number())
        maxStateDBThreads = config["maxStateDBThreads"];

    proverName = "UNSPECIFIED";
    if (config.contains("proverName") && config["proverName"].is_string())
        proverName = config["proverName"];
}

void Config::print(void)
{
    cout << "Configuration:" << endl;

    cout << "    proverID=" << proverID << endl;
    cout << "    proverName=" << proverName << endl;

    if (runExecutorServer)
        cout << "    runExecutorServer=true" << endl;
    if (runExecutorClient)
        cout << "    runExecutorClient=true" << endl;
    if (runExecutorClientMultithread)
        cout << "    runExecutorClientMultithread=true" << endl;
    if (runStateDBServer)
        cout << "    runStateDBServer=true" << endl;
    if (runStateDBTest)
        cout << "    runStateDBTest=true" << endl;
    if (runAggregatorServer)
        cout << "    runAggregatorServer=true" << endl;
    if (runAggregatorClient)
        cout << "    runAggregatorClient=true" << endl;
    if (runAggregatorClientMock)        
        cout << "    runAggregatorClientMock=true" << endl;
    if (runFileGenBatchProof)
        cout << "    runFileGenBatchProof=true" << endl;
    if (runFileGenAggregatedProof)
        cout << "    runFileGenAggregatedProof=true" << endl;
    if (runFileGenFinalProof)
        cout << "    runFileGenFinalProof=true" << endl;
    if (runFileProcessBatch)
        cout << "    runFileProcessBatch=true" << endl;
    if (runFileProcessBatchMultithread)
        cout << "    runFileProcessBatchMultithread=true" << endl;
    if (runFileExecute)
        cout << "    runFileExecute=true" << endl;

    if (runKeccakScriptGenerator)
        cout << "    runKeccakScriptGenerator=true" << endl;
    if (runKeccakTest)
        cout << "    runKeccakTest=true" << endl;
    if (runStorageSMTest)
        cout << "    runStorageSMTest=true" << endl;
    if (runBinarySMTest)
        cout << "    runBinarySMTest=true" << endl;
    if (runMemAlignSMTest)
        cout << "    runMemAlignSMTest=true" << endl;
    if (runSHA256Test)
        cout << "    runSHA256Test=true" << endl;
    if (runBlakeTest)
        cout << "    runBlakeTest=true" << endl;

    if (executeInParallel)
        cout << "    executeInParallel=true" << endl;
    if (useMainExecGenerated)
        cout << "    useMainExecGenerated=true" << endl;

    if (executorROMLineTraces)
        cout << "    executorROMLineTraces=true" << endl;

    if (executorTimeStatistics)
        cout << "    executorTimeStatistics=true" << endl;

    if (saveRequestToFile)
        cout << "    saveRequestToFile=true" << endl;
    if (saveInputToFile)
        cout << "    saveInputToFile=true" << endl;
    if (saveDbReadsToFile)
        cout << "    saveDbReadsToFile=true" << endl;
    if (saveDbReadsToFileOnChange)
        cout << "    saveDbReadsToFileOnChange=true" << endl;
    if (saveOutputToFile)
        cout << "    saveOutputToFile=true" << endl;
    if (saveProofToFile)
        cout << "    saveProofToFile=true" << endl;
    if (saveResponseToFile)
        cout << "    saveResponseToFile=true" << endl;
    if (loadDBToMemCache)
        cout << "    loadDBToMemCache=true" << endl;
    if (loadDBToMemCacheInParallel)
        cout << "    loadDBToMemCacheInParallel=true" << endl;
    if (opcodeTracer)
        cout << "    opcodeTracer=true" << endl;
    if (logRemoteDbReads)
        cout << "    logRemoteDbReads=true" << endl;
    if (logExecutorServerResponses)
        cout << "    logExecutorServerResponses=true" << endl;
    if (logExecutorServerTxs)
        cout << "    logExecutorServerTxs=true" << endl;
    if (dontLoadRomOffsets)
        cout << "    dontLoadRomOffsets=true" << endl;

    cout << "    executorServerPort=" << to_string(executorServerPort) << endl;
    cout << "    executorClientPort=" << to_string(executorClientPort) << endl;
    cout << "    executorClientHost=" << executorClientHost << endl;
    cout << "    stateDBServerPort=" << to_string(stateDBServerPort) << endl;
    cout << "    stateDBURL=" << stateDBURL << endl;
    cout << "    aggregatorServerPort=" << to_string(aggregatorServerPort) << endl;
    cout << "    aggregatorClientPort=" << to_string(aggregatorClientPort) << endl;
    cout << "    aggregatorClientHost=" << aggregatorClientHost << endl;
    cout << "    aggregatorClientMockTimeout=" << to_string(aggregatorClientMockTimeout) << endl;

    cout << "    inputFile=" << inputFile << endl;
    cout << "    inputFile2=" << inputFile2 << endl;
    cout << "    outputPath=" << outputPath << endl;
    cout << "    configPath=" << configPath << endl;
    cout << "    rom=" << rom << endl;
    cout << "    zkevmCmPols=" << zkevmCmPols << endl;
    cout << "    c12aCmPols=" << c12aCmPols << endl;
    cout << "    recursive1CmPols=" << recursive1CmPols << endl;
    cout << "    zkevmConstPols=" << zkevmConstPols << endl;
    cout << "    c12aConstPols=" << c12aConstPols << endl;
    if (mapConstPolsFile)
        cout << "    mapConstPolsFile=true" << endl;
    cout << "    zkevmConstantsTree=" << zkevmConstantsTree << endl;
    cout << "    c12aConstantsTree=" << c12aConstantsTree << endl;
    if (mapConstantsTreeFile)
        cout << "    mapConstantsTreeFile=true" << endl;
    cout << "    finalVerkey=" << finalVerkey << endl;
    cout << "    zkevmVerifier=" << zkevmVerifier << endl;
    cout << "    recursive1Verifier=" << recursive1Verifier << endl;
    cout << "    recursive2Verifier=" << recursive2Verifier << endl;
    cout << "    recursive2Verkey=" << recursive2Verkey << endl;
    cout << "    recursivefVerifier=" << recursivefVerifier << endl;
    cout << "    finalVerifier=" << finalVerifier << endl;
    cout << "    finalStarkZkey=" << finalStarkZkey << endl;
    cout << "    publicsOutput=" << publicsOutput << endl;
    cout << "    proofFile=" << proofFile << endl;
    cout << "    keccakScriptFile=" << keccakScriptFile << endl;
    cout << "    keccakPolsFile=" << keccakPolsFile << endl;
    cout << "    keccakConnectionsFile=" << keccakConnectionsFile << endl;
    cout << "    storageRomFile=" << storageRomFile << endl;
    cout << "    zkevmStarkInfo=" << zkevmStarkInfo << endl;
    cout << "    c12aStarkInfo=" << c12aStarkInfo << endl;
    cout << "    databaseURL=" << databaseURL << endl;
    cout << "    dbNodesTableName=" << dbNodesTableName << endl;
    cout << "    dbProgramTableName=" << dbProgramTableName << endl;
    cout << "    dbMultiWrite=" << to_string(dbMultiWrite) << endl;
    cout << "    dbFlushInParallel=" << to_string(dbFlushInParallel) << endl;
    cout << "    dbConnectionsPool=" << to_string(dbConnectionsPool) << endl;
    cout << "    dbNumberOfPoolConnections=" << dbNumberOfPoolConnections << endl;
    cout << "    cleanerPollingPeriod=" << cleanerPollingPeriod << endl;
    cout << "    requestsPersistence=" << requestsPersistence << endl;
    cout << "    maxExecutorThreads=" << maxExecutorThreads << endl;
    cout << "    maxProverThreads=" << maxProverThreads << endl;
    cout << "    maxStateDBThreads=" << maxStateDBThreads << endl;
    cout << "    dbMTCacheSize=" << dbMTCacheSize << endl;
    cout << "    dbProgramCacheSize=" << dbProgramCacheSize << endl;
}