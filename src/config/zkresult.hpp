#ifndef ZKRESULT_HPP
#define ZKRESULT_HPP

typedef enum : int
{
    ZKR_UNSPECIFIED = 0,
    ZKR_SUCCESS = 1,
    ZKR_DB_KEY_NOT_FOUND = 2, // Requested key was not found in database
    ZKR_DB_ERROR = 3, // Error connecting to database, or processing request
    ZKR_INTERNAL_ERROR = 4,
    ZKR_SM_MAIN_ASSERT = 5, // Main state machine executor assert failed
    ZKR_SM_MAIN_ARITH = 6, // Main state machine executor arith condition failed
    ZKR_SM_MAIN_BINARY = 7, // Main state machine executor binary condition failed
    ZKR_SM_MAIN_HASHP = 8, // Main state machine executor hash poseidon condition failed
    ZKR_SM_MAIN_HASHK = 9, // Main state machine executor hash Keccak condition failed
    ZKR_SM_MAIN_STORAGE = 10, // Main state machine executor storage condition failed
    ZKR_SM_MAIN_MEMORY = 11, // Main state machine executor memory condition failed
    ZKR_SM_MAIN_MEMALIGN = 12, // Main state machine executor memalign condition failed
    ZKR_SM_MAIN_ADDRESS = 13, // Main state machine executor address condition failed
    ZKR_SMT_INVALID_DATA_SIZE = 14, // Invalid size data for a MT node
    ZKR_SM_MAIN_BATCH_L2_DATA_TOO_BIG = 15, // Input batch L2 data is too big
    ZKR_AGGREGATED_PROOF_INVALID_INPUT = 16, // Aggregated proof input is incorrect
    ZKR_SM_MAIN_OOC_ARITH = 17, // Incremented arith counters exceeded the maximum
    ZKR_SM_MAIN_OOC_BINARY = 18, // Incremented binary counters exceeded the maximum
    ZKR_SM_MAIN_OOC_MEM_ALIGN = 19, // Incremented mem align counters exceeded the maximum
    ZKR_SM_MAIN_OOC_KECCAK_F = 20, // Incremented keccak-f counters exceeded the maximum
    ZKR_SM_MAIN_OOC_PADDING_PG = 21, // Incremented padding pg counters exceeded the maximum
    ZKR_SM_MAIN_OOC_POSEIDON_G = 22, // Incremented poseidon g counters exceeded the maximum
    ZKR_STATEDB_GRPC_ERROR = 23, // Error making GRPC call to stateDB service
    ZKR_SM_MAIN_OUT_OF_STEPS = 24, // Main state machine executor did not complete the execution within available steps
    ZKR_SM_MAIN_INVALID_FORK_ID = 25, // Main state machine executor does not support the requested fork ID
    ZKR_SM_MAIN_INVALID_UNSIGNED_TX = 26, // Main state machine executor cannot process unsigned TXs in prover mode
    ZKR_SM_MAIN_BALANCE_MISMATCH = 27 // Main state machine executor found that total tranferred balances are not zero
} zkresult;

const char* zkresult2string (int code);

#endif