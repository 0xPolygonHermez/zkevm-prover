#include <string>
#include <nlohmann/json.hpp>
#include "config.hpp"
#include "zkassert.hpp"
#include "utils.hpp"

using namespace std;
using json = nlohmann::json;

void Config::load(json &config)
{
    zkassert(processID == "");
    processID = getUUID();

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

    loadDBToMemCache = false;
    if (config.contains("loadDBToMemCache") && config["loadDBToMemCache"].is_boolean())
        loadDBToMemCache = config["loadDBToMemCache"];

    opcodeTracer = false;
    if (config.contains("opcodeTracer") && config["opcodeTracer"].is_boolean())
        opcodeTracer = config["opcodeTracer"];

    logRemoteDbReads = false;
    if (config.contains("logRemoteDbReads") && config["logRemoteDbReads"].is_boolean())
        logRemoteDbReads = config["logRemoteDbReads"];

    logExecutorServerResponses = false;
    if (config.contains("logExecutorServerResponses") && config["logExecutorServerResponses"].is_boolean())
        logExecutorServerResponses = config["logExecutorServerResponses"];

    proverServerPort = 50051;
    if (config.contains("proverServerPort") && config["proverServerPort"].is_number())
        proverServerPort = config["proverServerPort"];

    proverServerMockPort = 50052;
    if (config.contains("proverServerMockPort") && config["proverServerMockPort"].is_number())
        proverServerMockPort = config["proverServerMockPort"];

    proverServerMockTimeout = 60 * 1000 * 1000;
    if (config.contains("proverServerMockTimeout") && config["proverServerMockTimeout"].is_number())
        proverServerMockTimeout = config["proverServerMockTimeout"];

    proverClientPort = 50051;
    if (config.contains("proverClientPort") && config["proverClientPort"].is_number())
        proverClientPort = config["proverClientPort"];

    proverClientHost = "127.0.0.1";
    if (config.contains("proverClientHost") && config["proverClientHost"].is_string())
        proverClientHost = config["proverClientHost"];

    executorServerPort = 50071;
    if (config.contains("executorServerPort") && config["executorServerPort"].is_number())
        executorServerPort = config["executorServerPort"];

    executorROMLineTraces = false;
    if (config.contains("executorROMLineTraces") && config["executorROMLineTraces"].is_boolean())
        executorROMLineTraces = config["executorROMLineTraces"];

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

    if (config.contains("inputFile") && config["inputFile"].is_string())
        inputFile = config["inputFile"];
    if (config.contains("inputFile2") && config["inputFile2"].is_string())
        inputFile2 = config["inputFile2"];

    if (config.contains("romFile") && config["romFile"].is_string())
        romFile = config["romFile"];

    if (config.contains("outputPath") && config["outputPath"].is_string())
        outputPath = config["outputPath"];

    if (config.contains("cmPolsFile") && config["cmPolsFile"].is_string())
        cmPolsFile = config["cmPolsFile"];

    if (config.contains("cmPolsFileC12a") && config["cmPolsFileC12a"].is_string())
        cmPolsFileC12a = config["cmPolsFileC12a"];

    if (config.contains("cmPolsFileRecursive1") && config["cmPolsFileRecursive1"].is_string())
        cmPolsFileRecursive1 = config["cmPolsFileRecursive1"];

    if (config.contains("constPolsFile") && config["constPolsFile"].is_string())
        constPolsFile = config["constPolsFile"];

    if (config.contains("constPolsC12aFile") && config["constPolsC12aFile"].is_string())
        constPolsC12aFile = config["constPolsC12aFile"];

    if (config.contains("constPolsRecursive1File") && config["constPolsRecursive1File"].is_string())
        constPolsRecursive1File = config["constPolsRecursive1File"];

    mapConstPolsFile = true;
    if (config.contains("mapConstPolsFile") && config["mapConstPolsFile"].is_boolean())
        mapConstPolsFile = config["mapConstPolsFile"];

    if (config.contains("constantsTreeFile") && config["constantsTreeFile"].is_string())
        constantsTreeFile = config["constantsTreeFile"];

    if (config.contains("constantsTreeC12aFile") && config["constantsTreeC12aFile"].is_string())
        constantsTreeC12aFile = config["constantsTreeC12aFile"];

    if (config.contains("constantsTreeRecursive1File") && config["constantsTreeRecursive1File"].is_string())
        constantsTreeRecursive1File = config["constantsTreeRecursive1File"];

    mapConstantsTreeFile = true;
    if (config.contains("mapConstantsTreeFile") && config["mapConstantsTreeFile"].is_boolean())
        mapConstantsTreeFile = config["mapConstantsTreeFile"];

    if (config.contains("starkFile") && config["starkFile"].is_string())
        starkFile = config["starkFile"];

    if (config.contains("starkFilec12a") && config["starkFilec12a"].is_string())
        starkFilec12a = config["starkFilec12a"];

    if (config.contains("starkFileRecursive1") && config["starkFileRecursive1"].is_string())
        starkFileRecursive1 = config["starkFileRecursive1"];

    if (config.contains("starkZkIn") && config["starkZkIn"].is_string())
        starkZkIn = config["starkZkIn"];

    if (config.contains("starkZkInC12a") && config["starkZkInC12a"].is_string())
        starkZkInC12a = config["starkZkInC12a"];

    if (config.contains("starkZkInRecursive1") && config["starkZkInRecursive1"].is_string())
        starkZkInRecursive1 = config["starkZkInRecursive1"];

    if (config.contains("verifierFile") && config["verifierFile"].is_string())
        verifierFile = config["verifierFile"];

    if (config.contains("verifierFileRecursive1") && config["verifierFileRecursive1"].is_string())
        verifierFileRecursive1 = config["verifierFileRecursive1"];

    if (config.contains("witnessFile") && config["witnessFile"].is_string())
        witnessFile = config["witnessFile"];

    if (config.contains("witnessFileRecursive1") && config["witnessFileRecursive1"].is_string())
        witnessFileRecursive1 = config["witnessFileRecursive1"];

    if (config.contains("execC12aFile") && config["execC12aFile"].is_string())
        execC12aFile = config["execC12aFile"];

    if (config.contains("execRecursive1File") && config["execRecursive1File"].is_string())
        execRecursive1File = config["execRecursive1File"];

    if (config.contains("starkVerifierFile") && config["starkVerifierFile"].is_string())
        starkVerifierFile = config["starkVerifierFile"];

    if (config.contains("proofFile") && config["proofFile"].is_string())
        proofFile = config["proofFile"];

    if (config.contains("publicStarkFile") && config["publicStarkFile"].is_string())
        publicStarkFile = config["publicStarkFile"];

    if (config.contains("publicFile") && config["publicFile"].is_string())
        publicFile = config["publicFile"];

    if (config.contains("keccakScriptFile") && config["keccakScriptFile"].is_string())
        keccakScriptFile = config["keccakScriptFile"];

    if (config.contains("keccakPolsFile") && config["keccakPolsFile"].is_string())
        keccakPolsFile = config["keccakPolsFile"];

    if (config.contains("keccakConnectionsFile") && config["keccakConnectionsFile"].is_string())
        keccakConnectionsFile = config["keccakConnectionsFile"];

    if (config.contains("storageRomFile") && config["storageRomFile"].is_string())
        storageRomFile = config["storageRomFile"];

    if (config.contains("starkInfoFile") && config["starkInfoFile"].is_string())
        starkInfoFile = config["starkInfoFile"];

    if (config.contains("starkInfoC12aFile") && config["starkInfoC12aFile"].is_string())
        starkInfoC12aFile = config["starkInfoC12aFile"];

    if (config.contains("starkInfoRecursive1File") && config["starkInfoRecursive1File"].is_string())
        starkInfoRecursive1File = config["starkInfoRecursive1File"];

    if (config.contains("stateDBURL") && config["stateDBURL"].is_string())
        stateDBURL = config["stateDBURL"];

    if (config.contains("databaseURL") && config["databaseURL"].is_string())
        databaseURL = config["databaseURL"];

    if (config.contains("dbNodesTableName") && config["dbNodesTableName"].is_string())
        dbNodesTableName = config["dbNodesTableName"];

    if (config.contains("dbProgramTableName") && config["dbProgramTableName"].is_string())
        dbProgramTableName = config["dbProgramTableName"];

    dbAsyncWrite = false;
    if (config.contains("dbAsyncWrite") && config["dbAsyncWrite"].is_boolean())
        dbAsyncWrite = config["dbAsyncWrite"];

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
}

void Config::print(void)
{
    cout << "Configuration:" << endl;

    cout << "    processID=" << processID << endl;

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
    if (saveResponseToFile)
        cout << "    saveResponseToFile=true" << endl;
    if (loadDBToMemCache)
        cout << "    loadDBToMemCache=true" << endl;
    if (opcodeTracer)
        cout << "    opcodeTracer=true" << endl;
    if (logRemoteDbReads)
        cout << "    logRemoteDbReads=true" << endl;
    if (logExecutorServerResponses)
        cout << "    logExecutorServerResponses=true" << endl;

    cout << "    proverServerPort=" << to_string(proverServerPort) << endl;
    cout << "    proverServerMockPort=" << to_string(proverServerMockPort) << endl;
    cout << "    proverClientPort=" << to_string(proverClientPort) << endl;
    cout << "    proverClientHost=" << proverClientHost << endl;
    cout << "    executorServerPort=" << to_string(executorServerPort) << endl;
    cout << "    executorClientPort=" << to_string(executorClientPort) << endl;
    cout << "    executorClientHost=" << executorClientHost << endl;
    cout << "    stateDBServerPort=" << to_string(stateDBServerPort) << endl;
    cout << "    stateDBURL=" << stateDBURL << endl;
    cout << "    aggregatorServerPort=" << to_string(aggregatorServerPort) << endl;
    cout << "    aggregatorClientPort=" << to_string(aggregatorClientPort) << endl;
    cout << "    aggregatorClientHost=" << aggregatorClientHost << endl;

    cout << "    inputFile=" << inputFile << endl;
    cout << "    inputFile2=" << inputFile2 << endl;
    cout << "    outputPath=" << outputPath << endl;
    cout << "    romFile=" << romFile << endl;
    cout << "    cmPolsFile=" << cmPolsFile << endl;
    cout << "    cmPolsFileC12a=" << cmPolsFileC12a << endl;
    cout << "    cmPolsFileRecursive1=" << cmPolsFileRecursive1 << endl;
    cout << "    constPolsFile=" << constPolsFile << endl;
    cout << "    constPolsC12aFile=" << constPolsC12aFile << endl;
    if (mapConstPolsFile)
        cout << "    mapConstPolsFile=true" << endl;
    cout << "    constantsTreeFile=" << constantsTreeFile << endl;
    cout << "    constantsTreeC12aFile=" << constantsTreeC12aFile << endl;
    if (mapConstantsTreeFile)
        cout << "    mapConstantsTreeFile=true" << endl;
    cout << "    starkFile=" << starkFile << endl;
    cout << "    starkFilec12a=" << starkFilec12a << endl;
    cout << "    starkFileRecursive1=" << starkFileRecursive1 << endl;
    cout << "    starkZkIn=" << starkZkIn << endl;
    cout << "    starkZkInC12a=" << starkZkInC12a << endl;
    cout << "    starkZkInRecursive1=" << starkZkInRecursive1 << endl;
    cout << "    verifierFile=" << verifierFile << endl;
    cout << "    verifierFileRecursive1=" << verifierFileRecursive1 << endl;
    cout << "    witnessFile=" << witnessFile << endl;
    cout << "    witnessFileRecursive1=" << witnessFileRecursive1 << endl;
    cout << "    starkVerifierFile=" << starkVerifierFile << endl;
    cout << "    publicStarkFile=" << publicStarkFile << endl;
    cout << "    publicFile=" << publicFile << endl;
    cout << "    proofFile=" << proofFile << endl;
    cout << "    keccakScriptFile=" << keccakScriptFile << endl;
    cout << "    keccakPolsFile=" << keccakPolsFile << endl;
    cout << "    keccakConnectionsFile=" << keccakConnectionsFile << endl;
    cout << "    storageRomFile=" << storageRomFile << endl;
    cout << "    starkInfoFile=" << starkInfoFile << endl;
    cout << "    starkInfoC12aFile=" << starkInfoC12aFile << endl;
    cout << "    databaseURL=" << databaseURL << endl;
    cout << "    dbNodesTableName=" << dbNodesTableName << endl;
    cout << "    dbProgramTableName=" << dbProgramTableName << endl;
    cout << "    dbAsyncWrite=" << to_string(dbAsyncWrite) << endl;
    cout << "    cleanerPollingPeriod=" << cleanerPollingPeriod << endl;
    cout << "    requestsPersistence=" << requestsPersistence << endl;
    cout << "    maxExecutorThreads=" << maxExecutorThreads << endl;
    cout << "    maxProverThreads=" << maxProverThreads << endl;
    cout << "    maxStateDBThreads=" << maxStateDBThreads << endl;
}