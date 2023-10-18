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
    string proverID; // UUID assigned to this process instance, i.e. to this zkProver execution

    bool runExecutorServer;
    bool runExecutorClient;
    bool runExecutorClientMultithread;
    bool runHashDBServer;
    bool runHashDBTest;
    bool runAggregatorServer;
    bool runAggregatorClient;
    bool runAggregatorClientMock;    

    bool runFileGenBatchProof;              // Proof of 1 batch = Executor + Stark + StarkC12a + Recursive1
    bool runFileGenAggregatedProof;         // Proof of 2 batches = Recursive2 (of the 2 batches StarkC12a)
    bool runFileGenFinalProof;              // Final proof of an aggregated proof = RecursiveF + Groth16 (Snark)
    bool runFileProcessBatch;               // Executor (only main SM)
    bool runFileProcessBatchMultithread;    // Executor (only main SM) in parallel
    bool runFileExecute;                    // Executor (all SMs)

    bool runKeccakScriptGenerator;
    bool runSHA256ScriptGenerator;
    bool runKeccakTest;
    bool runStorageSMTest;
    bool runBinarySMTest;
    bool runMemAlignSMTest;
    bool runSHA256Test;
    bool runBlakeTest;
    bool runECRecoverTest;
    bool runDatabaseCacheTest;
    bool runCheckTreeTest;
    string checkTreeRoot;
    bool runDatabasePerformanceTest;
    bool runPageManagerTest;
    bool runKeyValueTreeTest;
    bool runSMT64Test;
    bool runUnitTest;
    
    bool executeInParallel;
    bool useMainExecGenerated;
    bool useMainExecC;

    bool saveRequestToFile; // Saves the grpc service request, in text format
    bool saveInputToFile; // Saves the grpc input data, in json format
    bool saveDbReadsToFile; // Saves the grpc input data, including database reads done during execution, in json format
    bool saveDbReadsToFileOnChange; // Same as saveDbReadsToFile, but saving the file at every read (slow, but useful if executor crashes)
    bool saveOutputToFile; // Saves the grpc output data, in json format
    bool saveProofToFile; // Saves the proof, in json format
    bool saveResponseToFile; // Saves the grpc service response, in text format
    bool saveFilesInSubfolders; // Saves output files in folders per hour, e.g. output/2023/01/10/18

    bool loadDBToMemCache;
    bool loadDBToMemCacheInParallel;
    uint64_t loadDBToMemTimeout;
    int64_t dbMTCacheSize; // Size in MBytes for the cache to store MT records
    bool useAssociativeCache; // Use the associative cache for MT records?
    int64_t log2DbMTAssociativeCacheSize; // log2 of the size in entries of the DatabaseMTAssociativeCache. Note 1 cache entry = 128 bytes
    int64_t log2DbMTAssociativeCacheIndexesSize; // log2 of the size in entries of the DatabaseMTAssociativeCache indexes. Note index entry = 4 bytes
    int64_t log2DbKVAssociativeCacheSize; // log2 of the size in entries of the DatabaseKVAssociativeCache. Note 1 cache entry = 80 bytes
    int64_t log2DbKVAssociativeCacheIndexesSize; // log2 of the size in entries of the DatabaseKVAssociativeCache indexes. Note index entry = 4 bytes
    int64_t log2DbVersionsAssociativeCacheSize; // log2 of the size in entries of the DatabaseVersionsAssociativeCache. Note 1 cache entry = 40 bytes
    int64_t log2DbVersionsAssociativeCacheIndexesSize; // log2 of the size in entries of the DatabaseVersionsAssociativeCache indexes. Note index entry = 4 bytes
    int64_t dbProgramCacheSize; // Size in MBytes for the cache to store Program (SC) records
    
    // Executor service
    uint16_t executorServerPort;
    uint16_t executorClientPort;
    string executorClientHost;
    uint64_t executorClientLoops;
    bool executorClientCheckNewStateRoot;

    // HashDB service
    uint16_t hashDBServerPort;
    string hashDBURL;
    bool hashDB64;
    uint64_t kvDBMaxVersions;
    string dbCacheSynchURL;

    // Aggregator service (client)
    uint16_t aggregatorServerPort;
    uint16_t aggregatorClientPort;
    string aggregatorClientHost;
    uint64_t aggregatorClientMockTimeout;
    uint64_t aggregatorClientWatchdogTimeout;
    uint64_t aggregatorClientMaxStreams; // Max number of streams, used to limit E2E test execution; if 0 then there is no limit

    // Executor debugging
    bool executorROMLineTraces;
    bool executorTimeStatistics;
    bool opcodeTracer;
    bool logRemoteDbReads;
    bool logExecutorServerInput; // Logs all inputs, before processing 
    bool logExecutorServerInputJson; // Logs all inputs in input.json format, before processing
    uint64_t logExecutorServerInputGasThreshold; // Logs input if gas/s < this value, active if this value is > 0
    bool logExecutorServerResponses;
    bool logExecutorServerTxs;
    bool dontLoadRomOffsets;

    // Files
    string inputFile;
    string inputFile2; // Used as the second input in genAggregatedProof
    string outputPath;
    string configPath;
    string rom;
    string zkevmCmPols; // Maps commit pols memory into file, which slows down a bit the executor
    string zkevmCmPolsAfterExecutor; // Saves commit pols into file after the executor has completed, avoiding having to map it from the beginning
    string c12aCmPols;
    string recursive1CmPols;
    string zkevmConstPols;
    string c12aConstPols;
    string recursive1ConstPols;
    string recursive2ConstPols;
    string recursivefConstPols;
    bool mapConstPolsFile;
    string zkevmConstantsTree;
    string c12aConstantsTree;
    string recursive1ConstantsTree;
    string recursive2ConstantsTree;
    string recursivefConstantsTree;
    bool mapConstantsTreeFile;
    string finalVerkey;
    string zkevmVerifier;
    string recursive1Verifier;
    string recursive2Verifier;
    string recursivefVerifier;
    string zkevmVerkey;    
    string c12aVerkey;    
    string recursive1Verkey;    
    string recursive2Verkey;    
    string recursivefVerkey;    
    string finalVerifier;
    string c12aExec;
    string recursive1Exec;
    string recursive2Exec;
    string recursivefExec;
    string finalStarkZkey;
    string publicsOutput;
    string proofFile;
    string keccakScriptFile;
    string sha256ScriptFile;
    string keccakPolsFile;
    string sha256PolsFile;
    string keccakConnectionsFile;
    string sha256ConnectionsFile;
    string storageRomFile;
    string zkevmStarkInfo;
    string c12aStarkInfo;
    string recursive1StarkInfo;
    string recursive2StarkInfo;
    string recursivefStarkInfo;

    // Database
    string databaseURL;
    string dbNodesTableName;
    string dbProgramTableName;
    string dbKeyValueTableName;
    string dbVersionTableName;
    string dbLatestVersionTableName;
    bool dbMultiWrite;
    uint64_t dbMultiWriteSingleQuerySize;
    bool dbConnectionsPool;
    uint64_t dbNumberOfPoolConnections;
    bool dbMetrics;
    bool dbClearCache;
    bool dbGetTree;
    bool dbReadOnly;
    uint64_t dbReadRetryCounter;
    uint64_t dbReadRetryDelay;

    // State manager
    bool stateManager;
    bool stateManagerPurge;
    bool stateManagerPurgeTxs;

    // Infrastructure
    uint64_t cleanerPollingPeriod;
    uint64_t requestsPersistence;
    uint64_t maxExecutorThreads;
    uint64_t maxProverThreads;
    uint64_t maxHashDBThreads;
    string proverName;
    uint64_t fullTracerTraceReserveSize;

    // EC Recover
    bool ECRecoverPrecalc;
    uint64_t ECRecoverPrecalcNThreads;

    // Logs format
    bool jsonLogs;

    void load(json &config);
    bool generateProof(void) const { return runFileGenBatchProof || runFileGenAggregatedProof || runFileGenFinalProof || runAggregatorClient; }
    void print(void);
    bool check(void); // Checks that the loaded configuration is correct; returns true if there is at least one error
};

#endif
