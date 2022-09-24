#include <string>
#include <nlohmann/json.hpp>
#include "config.hpp"

using namespace std;
using json = nlohmann::json;

void Config::load(json &config)
{
    runProverServer = false;
    if (config.contains("runProverServer") &&
        config["runProverServer"].is_boolean())
    {
        runProverServer = config["runProverServer"];
    }
    runProverServerMock = false;
    if (config.contains("runProverServerMock") &&
        config["runProverServerMock"].is_boolean())
    {
        runProverServerMock = config["runProverServerMock"];
    }
    runProverClient = false;
    if (config.contains("runProverClient") &&
        config["runProverClient"].is_boolean())
    {
        runProverClient = config["runProverClient"];
    }
    runExecutorServer = false;
    if (config.contains("runExecutorServer") &&
        config["runExecutorServer"].is_boolean())
    {
        runExecutorServer = config["runExecutorServer"];
    }
    runExecutorClient = false;
    if (config.contains("runExecutorClient") &&
        config["runExecutorClient"].is_boolean())
    {
        runExecutorClient = config["runExecutorClient"];
    }
    runStateDBServer = false;
    if (config.contains("runStateDBServer") &&
        config["runStateDBServer"].is_boolean())
    {
        runStateDBServer = config["runStateDBServer"];
    }
    runStateDBTest = false;
    if (config.contains("runStateDBTest") &&
        config["runStateDBTest"].is_boolean())
    {
        runStateDBTest = config["runStateDBTest"];
    }
    runFile = false;
    if (config.contains("runFile") &&
        config["runFile"].is_boolean())
    {
        runFile = config["runFile"];
    }
    runFileFast = false;
    if (config.contains("runFileFast") &&
        config["runFileFast"].is_boolean())
    {
        runFileFast = config["runFileFast"];
    }
    runKeccakScriptGenerator = false;
    if (config.contains("runKeccakScriptGenerator") &&
        config["runKeccakScriptGenerator"].is_boolean())
    {
        runKeccakScriptGenerator = config["runKeccakScriptGenerator"];
    }
    runKeccakTest = false;
    if (config.contains("runKeccakTest") &&
        config["runKeccakTest"].is_boolean())
    {
        runKeccakTest = config["runKeccakTest"];
    }
    runStorageSMTest = false;
    if (config.contains("runStorageSMTest") &&
        config["runStorageSMTest"].is_boolean())
    {
        runStorageSMTest = config["runStorageSMTest"];
    }
    runBinarySMTest = false;
    if (config.contains("runBinarySMTest") &&
        config["runBinarySMTest"].is_boolean())
    {
        runBinarySMTest = config["runBinarySMTest"];
    }
    runMemAlignSMTest = false;
    if (config.contains("runMemAlignSMTest") &&
        config["runMemAlignSMTest"].is_boolean())
    {
        runMemAlignSMTest = config["runMemAlignSMTest"];
    }
    runStarkTest = false;
    if (config.contains("runStarkTest") &&
        config["runStarkTest"].is_boolean())
    {
        runStarkTest = config["runStarkTest"];
    }
    useMainExecGenerated = false;
    if (config.contains("useMainExecGenerated") &&
        config["useMainExecGenerated"].is_boolean())
    {
        useMainExecGenerated = config["useMainExecGenerated"];
    }
    executeInParallel = false;
    if (config.contains("executeInParallel") &&
        config["executeInParallel"].is_boolean())
    {
        executeInParallel = config["executeInParallel"];
    }
    proverServerPort = 50051;
    if (config.contains("proverServerPort") &&
        config["proverServerPort"].is_number())
    {
        proverServerPort = config["proverServerPort"];
    }
    proverServerMockPort = 50052;
    if (config.contains("proverServerMockPort") &&
        config["proverServerMockPort"].is_number())
    {
        proverServerMockPort = config["proverServerMockPort"];
    }
    proverServerMockTimeout = 60 * 1000 * 1000;
    if (config.contains("proverServerMockTimeout") &&
        config["proverServerMockTimeout"].is_number())
    {
        proverServerMockTimeout = config["proverServerMockTimeout"];
    }
    proverClientPort = 50051;
    if (config.contains("proverClientPort") &&
        config["proverClientPort"].is_number())
    {
        proverClientPort = config["proverClientPort"];
    }
    proverClientHost = "127.0.0.1";
    if (config.contains("proverClientHost") &&
        config["proverClientHost"].is_string())
    {
        proverClientHost = config["proverClientHost"];
    }
    executorServerPort = 50071;
    if (config.contains("executorServerPort") &&
        config["executorServerPort"].is_number())
    {
        executorServerPort = config["executorServerPort"];
    }
    executorROMLineTraces = false;
    if (config.contains("executorROMLineTraces") &&
        config["executorROMLineTraces"].is_boolean())
    {
        executorROMLineTraces = config["executorROMLineTraces"];
    }
    executorClientPort = 50071;
    if (config.contains("executorClientPort") &&
        config["executorClientPort"].is_number())
    {
        executorClientPort = config["executorClientPort"];
    }
    executorClientHost = "127.0.0.1";
    if (config.contains("executorClientHost") &&
        config["executorClientHost"].is_string())
    {
        executorClientHost = config["executorClientHost"];
    }
    stateDBServerPort = 50061;
    if (config.contains("stateDBServerPort") &&
        config["stateDBServerPort"].is_number())
    {
        stateDBServerPort = config["stateDBServerPort"];
    }
    stateDBURL = "local";
    if (config.contains("stateDBURL") &&
        config["stateDBURL"].is_string())
    {
        stateDBURL = config["stateDBURL"];
    }
    if (config.contains("inputFile") &&
        config["inputFile"].is_string())
    {
        inputFile = config["inputFile"];
    }
    if (config.contains("romFile") &&
        config["romFile"].is_string())
    {
        romFile = config["romFile"];
    }
    if (config.contains("outputPath") &&
        config["outputPath"].is_string())
    {
        outputPath = config["outputPath"];
    }
    if (config.contains("cmPolsFile") &&
        config["cmPolsFile"].is_string())
    {
        cmPolsFile = config["cmPolsFile"];
    }
    if (config.contains("constPolsFile") &&
        config["constPolsFile"].is_string())
    {
        constPolsFile = config["constPolsFile"];
    }
    if (config.contains("constPolsC12aFile") &&
        config["constPolsC12aFile"].is_string())
    {
        constPolsC12aFile = config["constPolsC12aFile"];
    }
    if (config.contains("constPolsC12bFile") &&
        config["constPolsC12bFile"].is_string())
    {
        constPolsC12bFile = config["constPolsC12bFile"];
    }
    mapConstPolsFile = true;
    if (config.contains("mapConstPolsFile") &&
        config["mapConstPolsFile"].is_boolean())
    {
        mapConstPolsFile = config["mapConstPolsFile"];
    }
    if (config.contains("constantsTreeFile") &&
        config["constantsTreeFile"].is_string())
    {
        constantsTreeFile = config["constantsTreeFile"];
    }
    if (config.contains("constantsTreeC12aFile") &&
        config["constantsTreeC12aFile"].is_string())
    {
        constantsTreeC12aFile = config["constantsTreeC12aFile"];
    }
    if (config.contains("constantsTreeC12bFile") &&
        config["constantsTreeC12bFile"].is_string())
    {
        constantsTreeC12bFile = config["constantsTreeC12bFile"];
    }
    mapConstantsTreeFile = true;
    if (config.contains("mapConstantsTreeFile") &&
        config["mapConstantsTreeFile"].is_boolean())
    {
        mapConstantsTreeFile = config["mapConstantsTreeFile"];
    }
    if (config.contains("starkFile") &&
        config["starkFile"].is_string())
    {
        starkFile = config["starkFile"];
    }
    if (config.contains("starkFilec12a") &&
        config["starkFilec12a"].is_string())
    {
        starkFilec12a = config["starkFilec12a"];
    }
    if (config.contains("starkFilec12b") &&
        config["starkFilec12b"].is_string())
    {
        starkFilec12b = config["starkFilec12b"];
    }
    if (config.contains("starkZkIn") &&
        config["starkZkIn"].is_string())
    {
        starkZkIn = config["starkZkIn"];
    }
    if (config.contains("starkZkInC12a") &&
        config["starkZkInC12a"].is_string())
    {
        starkZkInC12a = config["starkZkInC12a"];
    }
    if (config.contains("starkZkInC12b") &&
        config["starkZkInC12b"].is_string())
    {
        starkZkInC12b = config["starkZkInC12b"];
    }
    if (config.contains("verifierFile") &&
        config["verifierFile"].is_string())
    {
        verifierFile = config["verifierFile"];
    }
    if (config.contains("verifierFileC12a") &&
        config["verifierFileC12a"].is_string())
    {
        verifierFileC12a = config["verifierFileC12a"];
    }
    if (config.contains("verifierFileC12b") &&
        config["verifierFileC12b"].is_string())
    {
        verifierFileC12b = config["verifierFileC12b"];
    }
    if (config.contains("witnessFile") &&
        config["witnessFile"].is_string())
    {
        witnessFile = config["witnessFile"];
    }
    if (config.contains("witnessFileC12a") &&
        config["witnessFileC12a"].is_string())
    {
        witnessFileC12a = config["witnessFileC12a"];
    }
    if (config.contains("witnessFileC12b") &&
        config["witnessFileC12b"].is_string())
    {
        witnessFileC12b = config["witnessFileC12b"];
    }
    if (config.contains("execC12aFile") &&
        config["execC12aFile"].is_string())
    {
        execC12aFile = config["execC12aFile"];
    }
    if (config.contains("execC12bFile") &&
        config["execC12bFile"].is_string())
    {
        execC12bFile = config["execC12bFile"];
    }
    if (config.contains("starkVerifierFile") &&
        config["starkVerifierFile"].is_string())
    {
        starkVerifierFile = config["starkVerifierFile"];
    }
    if (config.contains("proofFile") &&
        config["proofFile"].is_string())
    {
        proofFile = config["proofFile"];
    }
    if (config.contains("publicStarkFile") &&
        config["publicStarkFile"].is_string())
    {
        publicStarkFile = config["publicStarkFile"];
    }
    if (config.contains("publicFile") &&
        config["publicFile"].is_string())
    {
        publicFile = config["publicFile"];
    }
    if (config.contains("keccakScriptFile") &&
        config["keccakScriptFile"].is_string())
    {
        keccakScriptFile = config["keccakScriptFile"];
    }
    if (config.contains("keccakPolsFile") &&
        config["keccakPolsFile"].is_string())
    {
        keccakPolsFile = config["keccakPolsFile"];
    }
    if (config.contains("keccakConnectionsFile") &&
        config["keccakConnectionsFile"].is_string())
    {
        keccakConnectionsFile = config["keccakConnectionsFile"];
    }
    if (config.contains("storageRomFile") &&
        config["storageRomFile"].is_string())
    {
        storageRomFile = config["storageRomFile"];
    }
    if (config.contains("starkInfoFile") &&
        config["starkInfoFile"].is_string())
    {
        starkInfoFile = config["starkInfoFile"];
    }
    if (config.contains("starkInfoC12aFile") &&
        config["starkInfoC12aFile"].is_string())
    {
        starkInfoC12aFile = config["starkInfoC12aFile"];
    }
    if (config.contains("starkInfoC12bFile") &&
        config["starkInfoC12bFile"].is_string())
    {
        starkInfoC12bFile = config["starkInfoC12bFile"];
    }
    if (config.contains("stateDBURL") &&
        config["stateDBURL"].is_string())
    {
        stateDBURL = config["stateDBURL"];
    }
    if (config.contains("databaseURL") &&
        config["databaseURL"].is_string())
    {
        databaseURL = config["databaseURL"];
    }
    if (config.contains("dbTableName") &&
        config["dbTableName"].is_string())
    {
        dbTableName = config["dbTableName"];
    }
    dbAsyncWrite = false;
    if (config.contains("dbAsyncWrite") &&
        config["dbAsyncWrite"].is_boolean())
    {
        dbAsyncWrite = config["dbAsyncWrite"];
    }
    if (config.contains("cleanerPollingPeriod") &&
        config["cleanerPollingPeriod"].is_number())
    {
        cleanerPollingPeriod = config["cleanerPollingPeriod"];
    }
    if (config.contains("requestsPersistence") &&
        config["requestsPersistence"].is_number())
    {
        requestsPersistence = config["requestsPersistence"];
    }
}

void Config::print(void)
{
    cout << "Configuration:" << endl;
    if (runProverServer)
        cout << "runProverServer=true" << endl;
    if (runProverServerMock)
        cout << "runProverServerMock=true" << endl;
    if (runProverClient)
        cout << "runProverClient=true" << endl;
    if (runExecutorServer)
        cout << "runExecutorServer=true" << endl;
    if (runExecutorClient)
        cout << "runExecutorClient=true" << endl;
    if (runStateDBServer)
        cout << "runStateDBServer=true" << endl;
    if (runStateDBTest)
        cout << "runStateDBTest=true" << endl;
    if (runFile)
        cout << "runFile=true" << endl;
    if (runFileFast)
        cout << "runFileFast=true" << endl;
    if (runKeccakScriptGenerator)
        cout << "runKeccakScriptGenerator=true" << endl;
    if (runKeccakTest)
        cout << "runKeccakTest=true" << endl;
    if (runStorageSMTest)
        cout << "runStorageSMTest=true" << endl;
    if (runBinarySMTest)
        cout << "runBinarySMTest=true" << endl;
    if (runMemAlignSMTest)
        cout << "runMemAlignSMTest=true" << endl;
    if (runStarkTest)
        cout << "runStarkTest=true" << endl;
    if (executeInParallel)
        cout << "executeInParallel=true" << endl;
    if (useMainExecGenerated)
        cout << "useMainExecGenerated=true" << endl;
    cout << "proverServerPort=" << to_string(proverServerPort) << endl;
    cout << "proverServerMockPort=" << to_string(proverServerMockPort) << endl;
    cout << "proverClientPort=" << to_string(proverClientPort) << endl;
    cout << "proverClientHost=" << proverClientHost << endl;
    cout << "executorServerPort=" << to_string(executorServerPort) << endl;
    cout << "executorClientPort=" << to_string(executorClientPort) << endl;
    cout << "executorClientHost=" << executorClientHost << endl;
    cout << "stateDBServerPort=" << to_string(stateDBServerPort) << endl;
    cout << "stateDBURL=" << stateDBURL << endl;
    cout << "inputFile=" << inputFile << endl;
    cout << "outputPath=" << outputPath << endl;
    cout << "romFile=" << romFile << endl;
    cout << "cmPolsFile=" << cmPolsFile << endl;
    cout << "constPolsFile=" << constPolsFile << endl;
    cout << "constPolsC12aFile=" << constPolsC12aFile << endl;
    if (mapConstPolsFile)
        cout << "mapConstPolsFile=true" << endl;
    cout << "constantsTreeFile=" << constantsTreeFile << endl;
    cout << "constantsTreeC12aFile=" << constantsTreeC12aFile << endl;
    if (mapConstantsTreeFile)
        cout << "mapConstantsTreeFile=true" << endl;
    cout << "starkFile=" << starkFile << endl;
    cout << "starkFilec12a=" << starkFilec12a << endl;
    cout << "starkFilec12b=" << starkFilec12b << endl;
    cout << "starkZkIn=" << starkZkIn << endl;
    cout << "starkZkInC12a=" << starkZkInC12a << endl;
    cout << "starkZkInC12b=" << starkZkInC12b << endl;
    cout << "verifierFile=" << verifierFile << endl;
    cout << "verifierFileC12a=" << verifierFileC12a << endl;
    cout << "verifierFileC12b=" << verifierFileC12b << endl;
    cout << "witnessFile=" << witnessFile << endl;
    cout << "witnessFileC12a=" << witnessFileC12a << endl;
    cout << "witnessFileC12b=" << witnessFileC12b << endl;
    cout << "starkVerifierFile=" << starkVerifierFile << endl;
    cout << "publicStarkFile=" << publicStarkFile << endl;
    cout << "publicFile=" << publicFile << endl;
    cout << "proofFile=" << proofFile << endl;
    cout << "keccakScriptFile=" << keccakScriptFile << endl;
    cout << "keccakPolsFile=" << keccakPolsFile << endl;
    cout << "keccakConnectionsFile=" << keccakConnectionsFile << endl;
    cout << "storageRomFile=" << storageRomFile << endl;
    cout << "starkInfoFile=" << starkInfoFile << endl;
    cout << "starkInfoC12aFile=" << starkInfoC12aFile << endl;
    cout << "databaseURL=" << databaseURL << endl;
    cout << "dbTableName=" << dbTableName << endl;
    cout << "dbAsyncWrite=" << to_string(dbAsyncWrite) << endl;
    cout << "cleanerPollingPeriod=" << cleanerPollingPeriod << endl;
    cout << "requestsPersistence=" << requestsPersistence << endl;
}