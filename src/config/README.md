# Configuration Parameters

The zkProver supports a number of configuration parameters.
These parameters have a default value.
This default value can be modified through a configuration json file using the -c option: zkProver -c <config.json>.
They can also be modified usin environment variables, which have more priority than the config file modifications.
The configuration parameters can be of different uses:
- **Production**: the ones normally required to run the zkProver
- Test: used only while testing; most users can ignore them
- Tools: used only during the development; most users can ignore them

|Parameter|Use|Type|Description |Default|Environment variable|
|---------|---|----|------------|-------|--------------------|
|**`runExecutorServer`**|production|boolean|Enables Executor GRPC service, provides a service to process transaction batches; used by the Sequencer, Synchronizer and RPC|true|RUN_EXECUTOR_SERVER|
|`runExecutorClient`|test|boolean|Runs an executor client to test the executor service|false|RUN_EXECUTOR_CLIENT|
|`runExecutorClientMultithread`|test|boolean|Runs an multithread Executor client to test the Executor service|false|RUN_EXECUTOR_CLIENT_MULTITHREAD|
|**`runHashDBServer`**|production|boolean|Enables HashDB GRPC service, provides SMT (Sparse Merkle Tree) and Database access; used by the Synchronizer to create the genesis|true|RUN_HASHDB_SERVER|
|`runHashDBTest`|test|boolean|Runs a HashDB test to validate the HashDB service|false|RUN_HASHDB_TEST|
|`runAggregatorServer`|test|boolean|Runs an Aggregator service to test the Aggregator client|false|RUN_AGGREGATOR_SERVER|
|**`runAggregatorClient`**|production|boolean|Enables Aggregator GRPC client, connects to the Aggregator and process its proof generation requests; requires 512GB of RAM|false|RUN_AGGREGATOR_CLIENT|
|`runAggregatorClientMock`|test|boolean|Runs an Aggregator client mock that generates fake proofs|false|RUN_AGGREGATOR_CLIENT_MOCK|
|`runFileGenBatchProof`|test|boolean|Submits an input json file, defined in the `inputFile` parameter, to generate a regursive proof|false|RUN_FILE_GEN_BATCH_PROOF|
|`runFileGenAggregatedProof`|test|boolean|Submits two recursive proof files, defined in the `inputFile` and `inputFile2` parameters, to generate a recursive proof|false|RUN_FILE_GEN_AGGREGATED_PROOF|
|`runFileGenFinalProof`|test|boolean|Submits a recursive proof file, defined in the `inputFile` parameter, to generate a final proof|false|RUN_FILE_GEN_FINAL_PROOF|
|`runFileProcessBatch`|test|boolean|Submits an input json file, defined in the `inputFile` parameter, to process a batch|false|RUN_FILE_PROCESS_BATCH|
|`runFileProcessBatchMultithread`|test|boolean|Submits an input json file, defined in the `inputFile` parameter, to process a batch, multiple times in parallel|false|RUN_FILE_PROCESS_BATCH_MULTITHREAD|
|`runFileExecute`|test|boolean|Submits an input json file, defined in the `inputFile` parameter, to process a batch, including all secondary state machines|false|RUN_FILE_EXECUTE|
|`runKeccakScriptGenerator`|tools|boolean|Runs a Keccak-f hash that generates a keccak script json file|false|RUN_KECCAK_SCRIPT_GENERATOR|
|`runSHA256ScriptGenerator`|tools|boolean|Runs a SHA-256 hash that generates a SHA-256 script json file|false|RUN_SHA256_SCRIPT_GENERATOR|
|`runKeccakTest`|test|boolean|Runs a Keccak-f hash test|false|RUN_KECCAK_TEST|
|`runStorageSMTest`|test|boolean|Runs a storage state machine test|false|RUN_STORAGE_SM_TEST|
|`runBinarySMTest`|test|boolean|Runs a binary state machine test|false|RUN_BINARY_SM_TEST|
|`runMemAlignSMTest`|test|boolean|Runs a memory alignment state machine test|false|RUN_MEM_ALIGN_SM_TEST|
|`runSHA256Test`|test|boolean|Runs a SHA-256 hash test|false|RUN_SHA256_TEST|
|`runBlakeTest`|test|boolean|Runs a Blake hash test|false|RUN_BLAKE_TEST|
|`runECRecoverTest`|test|boolean|Runs an ECRecover test|false|RUN_ECRECOVER_TEST|
|`runDatabaseCacheTest`|test|boolean|Runs a database cache test|false|RUN_DATABASE_CACHE_TEST|
|`runCheckTreeTest`|test|boolean|Runs a check SMT tree test|false|RUN_CHECK_TREE_TEST|
|`checkTreeRoot`|test|string|State root used to check the tree, or automatically detect the last written one|"auto"|CHECK_TREE_ROOT|
|`runDatabasePerformanceTest`|test|boolean|Runs a database performance test|false|RUN_DATABASE_PERFORMANCE_TEST|
|`runPageManagerTest`|test|boolean|Runs a page manager test|false|RUN_PAGE_MANAGER_TEST|
|`runSMT64Test`|test|boolean|Runs a SMT64 test|false|RUN_SMT64_TEST|
|`runUnitTest`|test|boolean|Runs a unit test|false|RUN_UNIT_TEST|
|**`executeInParallel`**|production|boolean|Executes secondary state machines in parallel, when possible|true|EXECUTE_IN_PARALLEL|
|**`useMainExecGenerated`**|production|boolean|Executes main state machines in generated code, which is faster than native code|true|USE_MAIN_EXEC_GENERATED|
|`useMainExecC`|tools|boolean|Executes main state machines in C code, instead of executing the ROM (do not use in production, under development)|false|USE_MAIN_EXEC_C|
|`saveRequestToFile`|test|boolean|Saves executor GRPC requests to file, in text format|false|SAVE_REQUESTS_TO_FILE|
|`saveInputToFile`|test|boolean|Saves executor GRPC input to file, in JSON format|false|SAVE_INPUT_TO_FILE|
|`saveDbReadsToFile`|test|boolean|Saves executor reads to database to file, together with the input, in JSON format; the resulting file can be used as a self-contained input file that does not depend on any external database|false|SAVE_DB_READS_TO_FILE|
|`saveDbReadsToFileOnChange`|test|boolean|Saves executor reads to database to file, together with the input, in JSON format, updating the file every time a new read happens, which is useful to reproduce main executor errors; the resulting file can be used as a self-contained input file that does not depend on any external database|false|SAVE_DB_READS_TO_FILE_ON_CHANGE|
|`saveOutputToFile`|test|boolean|Saves executor GRPC output to file, in JSON format|false|SAVE_OUTPUT_TO_FILE|
|`saveResponseToFile`|test|boolean|Saves executor GRPC response to file, in text format|false|SAVE_RESPONSE_TO_FILE|
|`saveProofToFile`|test|boolean|Saves generated proof to file, in JSON format|false|SAVE_PROOF_TO_FILE|
|`saveFilesInSubfolders`|test|boolean|Saves files in folders per hour, e.g. `output/2023/01/10/18`|false|SAVE_FILES_IN_SUBFOLDERS|
|`loadDBToMemCache`|test|boolean|Fill database cache with content during initialization|false|LOAD_DB_TO_MEM_CACHE|
|`loadDBToMemCacheInParallel`|test|boolean|Fill database cache in parallel with the normal execution|false|LOAD_DB_TO_MEM_CACHE_IN_PARALLEL|
|`loadDBToMemTimeout`|test|u64|Fill database cache up to a certain time, in microseconds|30000000 (30 seconds)|LOAD_DB_TO_MEM_TIMEOUT|
|**`dbMTCacheSize`**|production|s64|Database MT cache size, in MB|8*1024 (8 GB)|DB_MT_CACHE_SIZE|
|**`useAssociativeCache`**|production|boolean|Use associative cache as Database MT cache, which is faster|false|USE_ASSOCIATIVE_CACHE|
|`log2DbMTAssociativeCacheSize`|production|s64|log2 of the size in entries of the DatabaseMTAssociativeCache; note that 1 cache entry = 128 bytes|25|LOG2_DB_MT_ASSOCIATIVE_CACHE_SIZE|
|`log2DbMTAssociativeCacheIndexesSize`|production|s64|log2 of the size in entries of the DatabaseMTAssociativeCache indexes; note that 1 cache entry = 4 bytes|28|LOG2_DB_MT_ASSOCIATIVE_CACHE_INDEXES_SIZE|
|`log2DbKVAssociativeCacheSize`|production|s64|log2 of the size in entries of the DatabaseKVAssociativeCache; note that 1 cache entry = 80 bytes|25|LOG2_DB_KV_ASSOCIATIVE_CACHE_SIZE|
|`log2DbKVAssociativeCacheIndexesSize`|production|s64|log2 of the size in entries of the DatabaseKVAssociativeCache indexes; note that 1 cache entry = 4 bytes|28|LOG2_DB_KV_ASSOCIATIVE_CACHE_INDEXES_SIZE|
|`log2DbVersionsAssociativeCacheSize`|production|s64|log2 of the size in entries of the DatabaseVersionsAssociativeCache; note that 1 cache entry = 40 bytes|25|LOG2_DB_VERSIONS_ASSOCIATIVE_CACHE_SIZE|
|`log2DbVersionsAssociativeCacheIndexesSize`|production|s64|log2 of the size in entries of the DatabaseVersionsAssociativeCache indexes; note that 1 cache entry = 4 bytes|28|LOG2_DB_VERSIONS_ASSOCIATIVE_CACHE_INDEXES_SIZE|
|**`dbProgramCacheSize`**|production|s64|Size for the cache to store Program (SC) records, in MB|1*1024 (1 GB)|DB_PROGRAM_CACHE_SIZE|
|**`executorServerPort`**|production|u16|Executor server GRPC port|50071|EXECUTOR_SERVER_PORT|
|`executorClientPort`|test|u16|Executor client GRPC port it connects to|50071|EXECUTOR_CLIENT_PORT|
|`executorClientHost`|test|string|Executor client host it connects to|"127.0.0.1"|EXECUTOR_CLIENT_HOST|
|`executorClientLoops`|test|u64|Executor client iterations|1|EXECUTOR_CLIENT_LOOPS|
|`executorClientCheckNewStateRoot`|test|bool|Executor client checks the new state root returned in the response using CheckTree|false|EXECUTOR_CLIENT_CHECK_NEW_STATE_ROOT|
|**`hashDBServerPort`**|production|u16|HashDB server GRPC port|50061|HASHDB_SERVER_PORT|
|**`hashDBURL`**|production|string|URL used by the Executor to connect to the HashDB service, e.g. "127.0.0.1:50061"; if set to "local", no GRPC is used and it connects to the local HashDB interface using direct calls to the HashDB classes; if your zkProver instance does not need to use a remote HashDB service for a good reason (e.g. not having direct access to the database) then even if it exports this service to other clients we recommend to use "local" since the performance is better|"local"|HASHDB_URL|
|`hashDB64`|test|boolean|Use HashDB64 new database (do not use in  production, under development)|false|HASHDB64|
|`kvDBMaxVersions`|production|u64|Maximum number of KV versionn in Database|131072|HASHDB64_MAX_VERSIONS|
|`dbCacheSynchURL`|test|string|URL of the HashDB service to synchronize the Database cache (experimental)|""|DB_CACHE_SYNCH_URL|
|`aggregatorServerPort`|test|u16|Aggregator server GRPC port|50081|AGGREGATOR_SERVER_PORT|
|**`aggregatorClientPort`**|production|u16|Aggregator client GRPC port to connect to|50081|AGGREGATOR_SERVER_PORT|
|**`aggregatorClientHost`**|production|string|Aggregator client GRPC host name to connect to, i.e. Aggregator server host name|"127.0.0.1"|AGGREGATOR_CLIENT_HOST|
|`aggregatorClientMockTimeout`|test|u64|Aggregator client mock timeout, in microseconds|60000000 (60 seconds)|AGGREGATOR_CLIENT_MOCK_TIMEOUT|
|**`aggregatorClientWatchdogTimeout`**|production|u64|Aggregator client watchdog timeout, in microseconds|60000000 (60 seconds)|AGGREGATOR_CLIENT_WATCHDOG_TIMEOUT|
|`aggregatorClientMaxStreams`|test|u64|Max number of aggregator client streams, used to limit E2E test execution; if 0 then there is no limit|0|AGGREGATOR_CLIENT_MAX_STREAMS|
|`executorROMLineTraces`|test|boolean|If true, the main state machine executor will log the content of every executed ROM program line|false|EXECUTOR_ROM_LINE_TRACES|
|`executorTimeStatistics`|test|boolean|If true, the main state machine executor will log the time metrics statistics of external calls|false|EXECUTOR_TIME_STATISTICS|
|`opcodeTracer`|test|boolean|Generate main state machine executor opcode statistics|false|OPCODE_TRACER|
|`logRemoteDbReads`|test|boolean|Log main state machine executor remote Database reads|false|LOG_REMOTE_DB_READS|
|`logExecutorServerInput`|test|boolean|Log main state machine executor input data|false|LOG_EXECUTOR_SERVER_INPUT|
|`logExecutorServerInputJson`|test|boolean|Log main state machine executor input data, in JSON format|false|LOG_EXECUTOR_SERVER_INPUT_JSON|
|`logExecutorServerInputGasThreshold`|test|u64|Log main state machine executor input if gas/s < this value; active if this value is > 0|0|LOG_EXECUTOR_SERVER_INPUT_GAS_THRESHOLD|
|`logExecutorServerResponses`|test|bool|Log executor server resonses|false|LOG_EXECUTOR_SERVER_RESPONSES|
|`logExecutorServerTxs`|test|bool|Log executor server transactins details|true|LOG_EXECUTOR_SERVER_TXS|
|`dontLoadRomOffsets`|test|bool|Avoid loading ROM offsets; used with experimental or testing ROM.json files|false|DONT_LOAD_ROM_OFFSETS|
|`inputFile`|test|string|Input file in some tests|"testvectors/batchProof/input_executor_0.json"|INPUT_FILE|
|`inputFile2`|test|string|Second input file, used as the second input in genAggregatedProof|""|INPUT_FILE_2|
|`outputPath`|test|string|Output directory for saved files|"output"|OUTPUT_PATH|
|**`configPath`**|production|string|Default input directory for configuration and constant files|"config"|CONFIG_PATH|
|`rom`|production|string|Prover rom.json file location|"src/main_sm/<fork>/scripts/rom.json"|ROM|
|`zkevmCmPols`|test|string|Maps commit pols memory into file, which slows down a bit the executor|""|ZKEVM_CM_POLS|
|`zkevmCmPolsAfterExecutor`|test|string|Saves commit pols into file after the executor has completed, avoiding having to map it from the beginning|""|ZKEVM_CM_POLS_AFTER_EXECUTOR|
|`recursive1CmPols`|test|string|Saves recursive 1 commit polynomials into file|""|RECURSIVE1_CM_POLS|
|`zkevmConstPols`|production|string|Saves constant polynomials into file|config + "/zkevm/zkevm.const"|ZKEVM_CONST_POLS|
|`mapConstPolsFile`|test|boolean|Maps constant polynomials file to memory|false|MAP_CONST_POLS_FILE|
|`c12aCmPols`|test|string|Saves C12A commit polynomials into file|""|C12A_CM_POLS|
|`c12aConstPols`|production|string|C12A constant polynomials file|config + "/c12a/c12a.const"|C12A_CONST_POLS|
|`c12aConstantsTree`|production|string|C12A constant tree polynomials file|config + "/c12a/c12a.consttree"|C12A_CONSTANTS_TREE|
|`c12aExec`|production|string|C12A exec file|config + "/c12a/c12a.exec"|C12A_EXEC|
|`c12aStarkInfo`|production|string|C12A STARK info file|config + "/c12a/c12a.starkinfo.json"|C12A_STARK_INFO|
|`recursive1ConstPols`|production|string|Recursive 1 constant polynomials file|config + "/recursive1/recursive1.const"|RECURSIVE1_CONST_POLS|
|`recursive2ConstPols`|production|string|Recursive 2 constant polynomials file|config + "/recursive2/recursive2.const"|RECURSIVE2_CONST_POLS|
|`recursivefConstPols`|production|string|Recursive final constant polynomials file|config + "/recursivef/recursivef.const"|RECURSIVEF_CONST_POLS|
|`recursive1Verifier`|production|string|Recursive 1 verifier data file|config + "/recursive1/recursive1.verifier.dat"|RECURSIVE1_VERIFIER|
|`recursive2Verifier`|production|string|Recursive 2 verifier data file|config + "/recursive2/recursive2.verifier.dat"|RECURSIVE2_VERIFIER|
|`recursivefVerifier`|production|string|Recursive final verifier data file|config + "/recursivef/recursivef.verifier.dat"|RECURSIVEF_VERIFIER|
|`zkevmConstantsTree`|production|string|Constant polynomials tree file|config + "/zkevm/zkevm.consttree"|ZKEVM_CONSTANTS_TREE|
|`mapConstantsTreeFile`|test|boolean|Maps constant polynomials tree file to memory|false|MAP_CONSTANTS_TREE_FILE|
|`recursive1StarkInfo`|production|string|Recursive 1 STARK info file|config + "/recursive1/recursive1.starkinfo.json"|RECURSIVE1_STARK_INFO|
|`recursive2StarkInfo`|production|string|Recursive 2 STARK info file|config + "/recursive2/recursive2.starkinfo.json"|RECURSIVE2_STARK_INFO|
|`recursivefStarkInfo`|production|string|Recursive final STARK info file|config + "/recursivef/recursivef.starkinfo.json"|RECURSIVEF_STARK_INFO|
|`finalVerkey`|production|string|Final verification key file|config + "/final/final.fflonk.verkey.json"|FINAL_VERKEY|
|`zkevmVerifier`|production|string|zkEVM verifier data file|config + "/zkevm/zkevm.verifier.dat"|ZKEVM_VERIFIER|
|`zkevmVerkey`|production|string|ZKEVM verification key file|config + "/zkevm/zkevm.verkey.json"|ZKEVM_VERKEY|
|`c12aVerkey`|production|string|C12A verification key file|config + "/c12a/c12a.verkey.json"|C12A_VERKEY|
|`recursive1Verkey`|production|string|Recursive 1 verification key file|config + "/recursive1/recursive1.verkey.json"|RECURSIVE1_VERKEY|
|`recursive2Verkey`|production|string|Recursive 2 verification key file|config + "/recursive2/recursive2.verkey.json"|RECURSIVE2_VERKEY|
|`recursivefVerkey`|production|string|Recursive F verification key file|config + "/recursivef/recursivef.verkey.json"|RECURSIVEF_VERKEY|
|`finalVerifier`|production|string|Final verifier data file|config + "/final/final.verifier.dat"|FINAL_VERIFIER|
|`recursive1Exec`|production|string|Recursive 1 exec file|config + "/recursive1/recursive1.exec"|RECURSIVE1_EXEC|
|`recursive2Exec`|production|string|Recursive 2 exec file|config + "/recursive2/recursive2.exec"|RECURSIVE2_EXEC|
|`recursivefExec`|production|string|Recursive final exec file|config + "/recursivef/recursivef.exec"|RECURSIVEF_EXEC|
|`finalStarkZkey`|production|string|Final STARK zkey file|config + "/final/final.fflonk.zkey"|FINAL_STARK_ZKEY|
|`publicsOutput`|production|string|Public data output file|"public.json"|PUBLICS_OUTPUT|
|`proofFile`|production|string|Proof data output file|"proof.json"|PROOF_FILE|
|`keccakScriptFile`|production|string|Keccak-f script file|config + "/scripts/keccak_script.json"|KECCAK_SCRIPT_FILE|
|`sha256ScriptFile`|production|string|SHA-256 script file|config + "/scripts/sha256_script.json"|SHA256_SCRIPT_FILE|
|`keccakPolsFile`|production|string|Keccak-f polynomials file|"keccak_pols.json"|KECCAK_POLS_FILE|
|`keccakConnectionsFile`|production|string|Keccak-f connections file|"keccak_connections.json"|KECCAK_CONNECTIONS_FILE|
|`keccakConnectionsFile`|production|string|Keccak-f connections file|"keccak_connections.json"|KECCAK_CONNECTIONS_FILE|
|`zkevmStarkInfo`|production|string|zkEVN STARK info file|config + "/zkevm/zkevm.starkinfo.json"|ZKEVM_STARK_INFO|
|`storageRomFile`|production|string|Storage ROM file|config + "/scripts/storage_sm_rom.json"|STORAGE_ROM_FILE|
|`recursive1ConstantsTree`|production|string|Recursive 1 contant polynomials tree file|config + "/recursive1/recursive1.consttree"|
|`recursive2ConstantsTree`|production|string|Recursive 2 contant polynomials tree file|config + "/recursive2/recursive2.consttree"|
|`recursivefConstantsTree`|production|string|Recursive final contant polynomials tree file|config + "/recursivef/recursivef.consttree"|RECURSIVE1_CONSTANTS_TREE|
|`sha256PolsFile`|production|string|SHA-256 polynomials file|"sha256_pols.json"|SHA256_POLS_FILE|
|`sha256ConnectionsFile`|production|string|SHA-256 connections file|"sha256_connections.json"|SHA256_CONNECTIONS_FILE|
|**`databaseURL`**|production|string|URL of the external database, e.g. postgresql://statedb:statedb@127.0.0.1:5432/testdb, or "local" if no external database is used (data will be stored in cache)|"local"|DATABASE_URL|
|`dbNodesTableName`|production|string|Name of the nodes table in the external database|"state.nodes"|DB_NODES_TABLE_NAME|
|`dbProgramTableName`|production|string|Name of the programs (smart contracts) table in the external database|"state.program"|DB_PROGRAM_TABLE_NAME|
|`dbKeyValueTableName`|production|string|Name of the key-value table in the external database|"state.keyvalue"|DB_KEYVALUE_TABLE_NAME|
|`dbVersionTableName`|production|string|Name of the version table in the external database|"state.version"|DB_VERSION_TABLE_NAME|
|`dbLatestVersionTableName`|production|string|Name of the latest version table in the external database|"state.latestversion"|DB_LATEST_VERSION_TABLE_NAME|
|`dbMultiWrite`|production|boolean|Use Database multi-write mechanism to send multiple write queries to database|true|DB_MULTIWRITE|
|`dbMultiWriteSingleQuerySize`|production|u64|Threshold of single Database query size when writing multi-write queries, in bytes|20*1024*1024 (20 MB)|DB_MULTIWRITE_SINGLE_QUERY_SIZE|
|`dbConnectionsPool`|production|boolean|Use a Database connections pool|true|DB_CONNECTIONS_POOL|
|`dbNumberOfPoolConnections`|production|u64|Number of Database pool of connections|30|DB_NUMBER_OF_POOL_CONNECTIONS|
|`dbMetrics`|test|boolean|Log Database metrics after each batch process|true|DB_METRICS|
|`dbClearCache`|test|boolean|Clear the cache after loading the Database input data|false|DB_CLEAR_CACHE|
|`dbGetTree`|production|boolean|Use Database PL-SQL GetTree function|true|DB_GET_TREE|
|`dbReadOnly`|production|boolean|Don't write any data to the external Database; used in RPC executors|false|DB_READ_ONLY|
|`dbReadRetryCounter`|production|u64|Number of Database retries, in case an error happens|10|DB_READ_RETRY_COUNTER|
|`dbReadRetryDelay`|production|u64|Delay between Database retries, in miliseconds|100*1000|DB_READ_RETRY_DELAY|
|`stateManager`|production|boolean|Use State Manager to consolidate states before writing to Database|true|STATE_MANAGER|
|`stateManagerPurge`|production|boolean|Purge State Manager sub-states|true|STATE_MANAGER_PURGE|
|`stateManagerPurgeTxs`|production|boolean|Purge State Manager transactions|true|STATE_MANAGER_PURGE_TXS|
|`cleanerPollingPeriod`|production|u64|Polling period of the cleaner thread that deletes completed Prover batches, in seconds|600|CLEANER_POLLING_PERIOD|
|`requestsPersistence`|production|u64|Time that completed batches stay before being cleaned up|3600|REQUESTS_PERSISTENCE|
|`maxExecutorThreads`|production|u64|Maximum numbre of GRPC Executor service threads|20|MAX_EXECUTOR_THREADS|
|`maxProverThreads`|test|u64|Maximum numbre of GRPC Aggregator service threads|8|MAX_PROVER_THREADS|
|`maxHashDBThreads`|production|u64|Maximum numbre of GRPC HashDB service threads|8|MAX_HASHDB_THREADS|
|`fullTracerTraceReserveSize`|production|u64|Full tracer number of reserved traces|256*1024|FULL_TRACER_TRACE_RESERVE_SIZE|
|`proverName`|production|string|Prover name, used to identy the prover when connecting to the Aggregator service|"UNSPECIFIED"|PROVER_NAME|
|`ECRecoverPrecalc`|production|boolean|Use ECRecover precalculation to improve main state machine executor performance (do not use in production, under development)|false|ECRECOVER_PRECALC|
|`ECRecoverPrecalcNThreads`|production|u64|Number of threads used to perform the ECRecover precalculation|16|ECRECOVER_PRECALC_N_THREADS|
|`jsonLogs`|production|boolean|Generate logs in JSON format, compatible with Datadog service|true|JSON_LOGS|
