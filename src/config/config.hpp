#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <string>
#include <iostream>
#include <nlohmann/json.hpp>
#include "definitions.hpp"

using namespace std;
using json = nlohmann::json;

class Config
{
public:
    bool runProverServer;
    bool runProverServerMock;
    bool runProverClient;
    bool runExecutorServer;
    bool runExecutorClient;
    bool runStateDBServer;
    bool runStateDBTest;
    bool runFile;
    bool runFileFast;
    bool runKeccakScriptGenerator;
    bool runKeccakTest;
    bool runStorageSMTest;
    bool runBinarySMTest;
    bool runMemAlignSMTest;
    bool runStarkTest;
    bool executeInParallel;
    bool useMainExecGenerated;
    uint16_t proverServerPort;
    uint16_t proverServerMockPort;
    uint16_t proverClientPort;
    string proverClientHost;
    uint16_t executorServerPort;
    uint16_t executorClientPort;
    string executorClientHost;
    uint16_t stateDBServerPort;
    string stateDBURL;
    string inputFile;
    string outputPath;
    string romFile;
    string cmPolsFile;
    string constPolsFile;
    bool mapConstPolsFile;
    string constantsTreeFile;
    bool mapConstantsTreeFile;
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
    string starkInfoFile;
    string databaseURL;
    string dbTableName;
    bool dbAsyncWrite;
    uint64_t cleanerPollingPeriod;
    uint64_t requestsPersistence;
    void load (json &config);
    bool generateProof (void) const { return runProverServer || runFile; }
    void print (void);
};

#endif