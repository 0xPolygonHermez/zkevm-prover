#ifndef DEFINITIONS_HPP
#define DEFINITIONS_HPP

#define CONCATENATE_DEFINES_AUX(x, y) x##y
#define CONCATENATE_DEFINES(x, y) CONCATENATE_DEFINES_AUX(x, y)

/* Only one fork can generate proofs */
#ifndef PROVER_FORK_ID
#define PROVER_FORK_ID 13
#endif

/* Regardless of the PROVER_FORK_ID, proof generation code uses the same namespace from the parent fork ID */
#define PROVER_FORK_NAMESPACE fork_13
#define PROVER_FORK_NAMESPACE_STRING "fork_13"
#define USING_PROVER_FORK_NAMESPACE using namespace fork_13 // Same pols definition for all proof generation files

/* Log traces selector: uncomment to enable the corresponding trace */
//#define LOG_START_STEPS
//#define LOG_COMPLETED_STEPS
//#define LOG_START_STEPS_TO_FILE
//#define LOG_COMPLETED_STEPS_TO_FILE
//#define LOG_PRINT_ROM_LINES
//#define LOG_INX
//#define LOG_ADDR
//#define LOG_ASSERT
//#define LOG_SETX
//#define LOG_JMP
//#define LOG_STORAGE
//#define LOG_MEMORY
//#define LOG_HASH
//#define LOG_POLS
//#define LOG_VARIABLES // If defined, logs variable declaration, get and set actions
//#define LOG_FILENAME // If defined, logs ROM compilation file name and line number
#define LOG_TIME // If defined, logs time differences to measure performance
//#define LOG_TIME_STATISTICS // If defined, generates main executor statistics for main operations
//#define LOG_TIME_STATISTICS_STATEDB_REMOTE // If defined, generates remote statedb statistics
//#define LOG_TIME_STATISTICS_STATEDB // If defined, generates statedb statistics
//#define LOG_TIME_STATISTICS_MAIN_EXECUTOR
//#define LOG_TIME_STATISTICS_STATE_MANAGER
//#define TIME_METRIC_TABLE
//#define LOG_TXS
//#define LOG_SERVICE
#define LOG_SERVICE_EXECUTOR_INPUT
#define LOG_SERVICE_EXECUTOR_OUTPUT
//#define LOG_BME
//#define LOG_BME_HASH
//#define LOG_SCRIPT_OUTPUT
//#define LOG_SMT
//#define LOG_SMT_SET_PRINT_TREE
//#define LOG_STORAGE_EXECUTOR
//#define LOG_STORAGE_EXECUTOR_ROM_LINE
//#define LOG_MEMORY_EXECUTOR
//#define LOG_BINARY_EXECUTOR
//#define LOG_HASHK
//#define LOG_HASH
//#define LOG_DB_READ
//#define LOG_DB_WRITE
//#define LOG_DB_WRITE_QUERY
//#define LOG_DB_SEND_DATA
//#define LOG_DB_DELETE_NODES
//#define LOG_DB_FLUSH
//#define LOG_DB_SEMI_FLUSH
//#define LOG_DB_SENDER_THREAD
//#define LOG_DB_WRITE_REMOTE
//#define LOG_DB_ACCEPT_INTRAY
//#define LOG_HASHDB_SERVICE
//#define LOG_STATE_MANAGER
//#define LOG_FULL_TRACER
#define LOG_FULL_TRACER_ON_ERROR
//#define LOG_TX_HASH
//#define LOG_INPUT
//#define LOG_ASSOCIATIVE_CACHE
//LOG_SMT_KEY_DETAILS

/* Prover defines */
//#define PROVER_USE_PROOF_GOOD_JSON
//#define PROVER_INJECT_ZKIN_JSON

/* Hash DB*/
//#define HASHDB_LOCK // If defined, the HashDB class will use a lock in all its methods, i.e. they will be serialized
//#define DATABASE_COMMIT // If defined, the Database class can be configured to autocommit, or explicitly commit(); used for testing only
#define DATABASE_USE_CACHE // If defined, the Database class uses a cache
#define USE_NEW_KVTREE

#define MAIN_SM_EXECUTOR_GENERATED_CODE
#define MAIN_SM_PROVER_GENERATED_CODE

#define LOAD_CONST_FILES false

#ifndef REDUCE_ZKEVM_MEMORY
#define REDUCE_ZKEVM_MEMORY true
#endif

#define USE_GENERIC_PARSER true

#define NROWS_PACK 4

#ifdef __USE_CUDA__
#define TRANSPOSE_TMP_POLS false
#else
#define TRANSPOSE_TMP_POLS true
#endif

//#define MULTI_ROM_TEST

//#define ENABLE_EXPERIMENTAL_CODE


#endif
