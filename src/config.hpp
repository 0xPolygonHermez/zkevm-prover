#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <string>
#include <iostream>
#include <nlohmann/json.hpp>

using namespace std;
using json = nlohmann::json;

class Config
{
public:
    bool runServer;
    bool runServerMock;
    bool runClient;
    bool runFile;
    bool runFileFast;
    bool runKeccakScriptGenerator;
    bool runKeccakTest;
    bool runStorageSMTest;
    bool runBinarySMTest;
    bool runMemAlignSMTest;
    bool runFiniteFieldTest;
    bool runStarkTest;
    bool executeInParallel;
    bool useMainExecGenerated;
    uint16_t proverServerPort;
    uint16_t proverServerMockPort;
    uint16_t proverClientPort;
    string inputFile;
    string outputPath;
    string romFile;
    string pilFile;
    string cmPolsFile;
    string constPolsFile;
    string constantsTreeFile;
    string scriptFile;
    string starkFile;
    string verifierFile;
    string witnessFile;
    string starkVerifierFile;
    string publicFile;
    string proofFile;
    string keccakScriptFile;
    string keccakPolsFile;
    string keccakConnectionsFile;
    string storageRomFile;
    string storagePilFile;
    string storagePolsFile;
    string memoryPilFile;
    string memoryPolsFile;
    string binaryPilFile;
    string binaryPolsFile;
    string binaryConstPolsFile;
    string dbHost;
    uint16_t dbPort;
    string dbUser;
    string dbPassword;
    string dbDatabaseName;
    string dbTableName;
    uint64_t cleanerPollingPeriod;
    uint64_t requestsPersistence;
    void load(json &config);
};

#endif