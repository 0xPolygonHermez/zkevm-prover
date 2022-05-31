#include <string>
#include <nlohmann/json.hpp>
#include "config.hpp"

using namespace std;
using json = nlohmann::json;

void Config::load(json &config)
{
    runServer = false;
    if (config.contains("runServer") && 
        config["runServer"].is_boolean())
    {
        runServer = config["runServer"];
    }
    runServerMock = false;
    if (config.contains("runServerMock") && 
        config["runServerMock"].is_boolean())
    {
        runServerMock = config["runServerMock"];
    }
    runClient = false;
    if (config.contains("runClient") && 
        config["runClient"].is_boolean())
    {
        runClient = config["runClient"];
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
    executeInParallel = false;
    if (config.contains("executeInParallel") && 
        config["executeInParallel"].is_boolean())
    {
        executeInParallel = config["executeInParallel"];
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
    serverPort = 50051;
    if (config.contains("serverPort") && 
        config["serverPort"].is_number())
    {
        serverPort = config["serverPort"];
    }
    serverMockPort = 50052;
    if (config.contains("serverMockPort") && 
        config["serverMockPort"].is_number())
    {
        serverMockPort = config["serverMockPort"];
    }
    clientPort = 50051;
    if (config.contains("clientPort") && 
        config["clientPort"].is_number())
    {
        clientPort = config["clientPort"];
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
    if (config.contains("pilFile") && 
        config["pilFile"].is_string())
    {
        pilFile = config["pilFile"];
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
    if (config.contains("constantsTreeFile") && 
        config["constantsTreeFile"].is_string())
    {
        constantsTreeFile = config["constantsTreeFile"];
    }
    if (config.contains("scriptFile") && 
        config["scriptFile"].is_string())
    {
        scriptFile = config["scriptFile"];
    }
    if (config.contains("starkFile") && 
        config["starkFile"].is_string())
    {
        starkFile = config["starkFile"];
    }
    if (config.contains("verifierFile") && 
        config["verifierFile"].is_string())
    {
        verifierFile = config["verifierFile"];
    }
    if (config.contains("witnessFile") && 
        config["witnessFile"].is_string())
    {
        witnessFile = config["witnessFile"];
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
    if (config.contains("storagePilFile") && 
        config["storagePilFile"].is_string())
    {
        storagePilFile = config["storagePilFile"];
    }
    if (config.contains("storagePolsFile") && 
        config["storagePolsFile"].is_string())
    {
        storagePolsFile = config["storagePolsFile"];
    }
    if (config.contains("memoryPilFile") && 
        config["memoryPilFile"].is_string())
    {
        memoryPilFile = config["memoryPilFile"];
    }
    if (config.contains("memoryPolsFile") && 
        config["memoryPolsFile"].is_string())
    {
        memoryPolsFile = config["memoryPolsFile"];
    }
    if (config.contains("binaryPilFile") && 
        config["binaryPilFile"].is_string())
    {
        binaryPilFile = config["binaryPilFile"];
    }
    if (config.contains("binaryPolsFile") && 
        config["binaryPolsFile"].is_string())
    {
        binaryPolsFile = config["binaryPolsFile"];
    }
    if (config.contains("binaryConstPolsFile") && 
        config["binaryConstPolsFile"].is_string())
    {
        binaryConstPolsFile = config["binaryConstPolsFile"];
    }
    if (config.contains("dbHost") && 
        config["dbHost"].is_string())
    {
        dbHost = config["dbHost"];
    }
    if (config.contains("dbPort") && 
        config["dbPort"].is_number())
    {
        dbPort = config["dbPort"];
    }
    if (config.contains("dbUser") && 
        config["dbUser"].is_string())
    {
        dbUser = config["dbUser"];
    }
    if (config.contains("dbPassword") && 
        config["dbPassword"].is_string())
    {
        dbPassword = config["dbPassword"];
    }
    if (config.contains("dbDatabaseName") && 
        config["dbDatabaseName"].is_string())
    {
        dbDatabaseName = config["dbDatabaseName"];
    }
    if (config.contains("dbTableName") && 
        config["dbTableName"].is_string())
    {
        dbTableName = config["dbTableName"];
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