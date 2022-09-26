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
    bool runFileFastMultithread;
    bool runKeccakScriptGenerator;
    bool runKeccakTest;
    bool runStorageSMTest;
    bool runBinarySMTest;
    bool runMemAlignSMTest;
    bool runStarkTest;
    bool executeInParallel;
    bool useMainExecGenerated;
    bool saveDbReadsToFile;
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
    string cmPolsFileC12a;
    string cmPolsFileC12b;
    string constPolsFile;
    string constPolsC12aFile;
    string constPolsC12bFile;
    bool mapConstPolsFile;
    string constantsTreeFile;
    string constantsTreeC12aFile;
    string constantsTreeC12bFile;
    bool mapConstantsTreeFile;
    string starkFile;
    string starkFilec12a;
    string starkFilec12b;
    string starkZkIn;
    string starkZkInC12a;
    string starkZkInC12b;
    string verifierFile;
    string verifierFileC12a;
    string verifierFileC12b;
    string witnessFile;
    string witnessFileC12a;
    string witnessFileC12b;
    string execC12aFile;
    string execC12bFile;
    string starkVerifierFile;
    string publicStarkFile;
    string publicFile;
    string proofFile;
    string keccakScriptFile;
    string keccakPolsFile;
    string keccakConnectionsFile;
    string storageRomFile;
    string starkInfoFile;
    string starkInfoC12aFile;
    string starkInfoC12bFile;
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