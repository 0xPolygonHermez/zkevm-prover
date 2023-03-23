#ifndef DEFINITIONS_HPP
#define DEFINITIONS_HPP

#define ADDRESS_GLOBAL_EXIT_ROOT_MANAGER_L2 "0xAE4bB80bE56B819606589DE61d5ec3b522EEB032"

#define PROVER_FORK_ID 4
#define PROVER_FORK_NAMESPACE fork_4
#define PROVER_FORK_NAMESPACE_STRING "fork_4"
#define USING_PROVER_FORK_NAMESPACE using namespace fork_4

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
#define LOG_TIME_STATISTICS_MAIN_EXECUTOR
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
//#define LOG_DB_READ
//#define LOG_DB_WRITE
//#define LOG_STATEDB_SERVICE
//#define LOG_FULL_TRACER
#define LOG_FULL_TRACER_ON_ERROR
//#define LOG_TX_HASH
//#define LOG_INPUT

/* Prover defines */
//#define PROVER_USE_PROOF_GOOD_JSON
//#define PROVER_INJECT_ZKIN_JSON

/* State DB*/
//#define STATEDB_LOCK // If defined, the StateDB class will use a lock in all its methods, i.e. they will be serialized
//#define DATABASE_COMMIT // If defined, the Database class can be configured to autocommit, or explicitly commit(); used for testing only
#define DATABASE_USE_CACHE // If defined, the Database class uses a cache

#endif
