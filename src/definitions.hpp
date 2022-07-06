#ifndef DEFINITIONS_HPP
#define DEFINITIONS_HPP

#define NEVALUATIONS 65536 //1<<16, i.e. 1<<NBITS
#define NPOLS 119
#define NCONSTPOLS 61
#define ARITY 4
#define NBITS 16
#define EXTENDED_BITS 1
#define MERKLE_ARITY 16

#define ADDRESS_GLOBAL_EXIT_ROOT_MANAGER_L2 "0xAE4bB80bE56B819606589DE61d5ec3b522EEB032"

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
//#define LOG_TXS
#define LOG_SERVICE
//#define LOG_BME
//#define LOG_BME_HASH
//#define LOG_SCRIPT_OUTPUT
#define LOG_RPC_INPUT
#define LOG_RPC_OUTPUT
//#define LOG_SMT
//#define LOG_STORAGE_EXECUTOR
//#define LOG_STORAGE_EXECUTOR_ROM_LINE
//#define LOG_MEMORY_EXECUTOR
//#define LOG_BINARY_EXECUTOR
//#define LOG_HASHK
#define LOG_STATEDB_SERVICE

/* Executor defines */
//#define USE_LOCAL_STORAGE // If defined, use ctx.sto[], else, use smt.get()

/* Prover defines */
//#define PROVER_USE_PROOF_GOOD_JSON
//#define PROVER_INJECT_ZKIN_JSON

#endif