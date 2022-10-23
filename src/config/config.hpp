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
    bool runExecutorClientMultithread;
    bool runStateDBServer;
    bool runStateDBTest;
    bool runAggregatorServer;
    bool runAggregatorClient;

    bool runFileGenProof;                   // Full proof = Executor + Stark + StarkC12a + StarkC12b + Groth16 (Snark)
    bool runFileGenBatchProof;              // Proof of 1 batch = Executor + Stark + StarkC12a
    bool runFileGenAggregatedProof;         // Proof of 2 batches = StarckC12a (of the 2 batches StarkC12a)
    bool runFileGenFinalProof;              // Final proof of an aggregated proof = StarkC12b + Groth16 (Snark)
    bool runFileProcessBatch;               // Executor (only main SM)
    bool runFileProcessBatchMultithread;    // Executor (only main SM) in parallel

    bool runKeccakScriptGenerator;
    bool runKeccakTest;
    bool runStorageSMTest;
    bool runBinarySMTest;
    bool runMemAlignSMTest;
    bool runStarkTest;
    
    bool executeInParallel;
    bool useMainExecGenerated;

    bool saveRequestToFile;
    bool saveInputToFile;
    bool saveDbReadsToFile;
    bool saveDbReadsToFileOnChange;
    bool saveResponseToFile;
    bool loadDBToMemCache;
    bool opcodeTracer;
    bool logRemoteDbReads;
    bool logExecutorServerResponses;

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

    uint16_t aggregatorServerPort;
    uint16_t aggregatorClientPort;
    string aggregatorClientHost;

    string inputFile;
    string inputFile2; // Used as the second input in genAggregatedProof
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
    string dbNodesTableName;
    string dbProgramTableName;
    bool dbAsyncWrite;
    uint64_t cleanerPollingPeriod;
    uint64_t requestsPersistence;
    uint64_t maxExecutorThreads;
    uint64_t maxProverThreads;
    uint64_t maxStateDBThreads;
    void load(json &config);
    bool generateProof(void) const { return runProverServer || runFileGenProof || runFileGenBatchProof || runFileGenAggregatedProof || runFileGenFinalProof || runAggregatorClient; }
    void print(void);
};

#endif