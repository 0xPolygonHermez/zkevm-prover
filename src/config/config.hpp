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
    uint64_t proverServerMockTimeout;
    uint16_t proverClientPort;
    string proverClientHost;
    uint16_t executorServerPort;
    bool executorROMLineTraces;
    uint16_t executorClientPort;
    string executorClientHost;
    uint16_t stateDBServerPort;
    string stateDBURL;
    string inputFile;
    string outputPath;
    string romFile;
    string cmPolsFile;
    string constPolsFile;
    string constPolsC12File;
    bool mapConstPolsFile;
    string constantsTreeFile;
    string constantsTreeC12File;
    bool mapConstantsTreeFile;
    string starkFile;
    string starkZkIn;
    string starkZkInC12;
    string verifierFile;
    string verifierFileC12;
    string witnessFile;
    string witnessFileC12;
    string execFile;
    string starkVerifierFile;
    string publicStarkFile;
    string publicFile;
    string proofFile;
    string keccakScriptFile;
    string keccakPolsFile;
    string keccakConnectionsFile;
    string storageRomFile;
    string starkInfoFile;
    string starkInfoC12File;
    string databaseURL;
    string dbTableName;
    bool dbAsyncWrite;
    uint64_t cleanerPollingPeriod;
    uint64_t requestsPersistence;
    void load(json &config);
    bool generateProof(void) const { return runProverServer || runFile; }
    void print(void);
};

#endif