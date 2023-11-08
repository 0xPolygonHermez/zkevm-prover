#ifndef ZKRESULT_HPP
#define ZKRESULT_HPP

#include <string>

using namespace std;

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
    ZKR_SM_MAIN_STORAGE_INVALID_KEY = 10, // Main state machine executor storage condition failed
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
    ZKR_HASHDB_GRPC_ERROR = 23, // Error making GRPC call to hash DB service
    ZKR_SM_MAIN_OUT_OF_STEPS = 24, // Main state machine executor did not complete the execution within available steps
    ZKR_SM_MAIN_INVALID_FORK_ID = 25, // Main state machine executor does not support the requested fork ID
    ZKR_SM_MAIN_INVALID_UNSIGNED_TX = 26, // Main state machine executor cannot process unsigned TXs in prover mode
    ZKR_SM_MAIN_BALANCE_MISMATCH = 27, // Main state machine executor found that total tranferred balances are not zero
    ZKR_SM_MAIN_FEA2SCALAR = 28, // Main state machine executor failed calling fea2scalar()
    ZKR_SM_MAIN_TOS32 = 29, // Main state machine executor failed calling fr.toS32()
    ZKR_SM_MAIN_BATCH_INVALID_INPUT = 30, // Process batch input is incorrect
    ZKR_SM_MAIN_S33 = 31, // Main state machine executor failed getting an S33 value from op
    ZKR_STATE_MANAGER = 32, // State root error
    ZKR_SM_MAIN_INVALID_NO_COUNTERS = 33, // No counters received outside of a process batch request
    ZKR_SM_MAIN_ARITH_ECRECOVER_DIVIDE_BY_ZERO = 34, // Main state machine arith operation during ECRecover found a divide by zero situation
    ZKR_SM_MAIN_ADDRESS_OUT_OF_RANGE = 35, // Main state machine address out of valid memory space range
    ZKR_SM_MAIN_ADDRESS_NEGATIVE = 36, // Main state machine address is negative
    ZKR_SM_MAIN_HASHK_SIZE_OUT_OF_RANGE = 37, // Main state Keccak hash size is out of range
    ZKR_SM_MAIN_HASHK_POSITION_NEGATIVE = 38, // Main state Keccak hash position is negative
    ZKR_SM_MAIN_HASHK_POSITION_PLUS_SIZE_OUT_OF_RANGE = 39, // Main state Keccak hash position + size is out of range
    ZKR_SM_MAIN_HASHKDIGEST_ADDRESS_NOT_FOUND = 40, // Main state Keccak hash digest address is not found
    ZKR_SM_MAIN_HASHKDIGEST_NOT_COMPLETED = 41, // Main state Keccak hash digest called when hash has not been completed
    ZKR_SM_MAIN_HASHP_SIZE_OUT_OF_RANGE = 42, // Main state Poseidon hash size is out of range
    ZKR_SM_MAIN_HASHP_POSITION_NEGATIVE = 43, // Main state Poseidon hash position is negative
    ZKR_SM_MAIN_HASHP_POSITION_PLUS_SIZE_OUT_OF_RANGE = 44, // Main state Poseidon hash position + size is out of range
    ZKR_SM_MAIN_HASHPDIGEST_ADDRESS_NOT_FOUND = 45, // Main state Poseidon hash digest address is not found
    ZKR_SM_MAIN_HASHPDIGEST_NOT_COMPLETED = 46, // Main state Poseidon hash digest called when hash has not been completed
    ZKR_SM_MAIN_MEMALIGN_OFFSET_OUT_OF_RANGE = 47, // Main state mem align offset is out of range
    ZKR_SM_MAIN_MULTIPLE_FREEIN = 48, // Main state got multiple free inputs in one sigle ROM instruction
    ZKR_SM_MAIN_STORAGE_READ_MISMATCH = 49, // Main state read ROM operation check failed
    ZKR_SM_MAIN_STORAGE_WRITE_MISMATCH = 50, // Main state write ROM operation check failed
    ZKR_SM_MAIN_HASHK_VALUE_MISMATCH = 51, // Main state Keccak hash ROM operation value check failed
    ZKR_SM_MAIN_HASHK_PADDING_MISMATCH = 52, // Main state Keccak hash ROM operation padding check failed
    ZKR_SM_MAIN_HASHK_SIZE_MISMATCH = 53, // Main state Keccak hash ROM operation size check failed
    ZKR_SM_MAIN_HASHKLEN_LENGTH_MISMATCH = 54, // Main state Keccak hash ROM operation data length check failed
    ZKR_SM_MAIN_HASHKLEN_CALLED_TWICE = 55, // Main state Keccak hash ROM operation called once check failed
    ZKR_SM_MAIN_HASHKDIGEST_NOT_FOUND = 56, // Main state Keccak hash digest ROM operation slot not found
    ZKR_SM_MAIN_HASHKDIGEST_DIGEST_MISMATCH = 57, // Main state Keccak hash digest ROM operation digest mismatch
    ZKR_SM_MAIN_HASHKDIGEST_CALLED_TWICE = 58, // Main state Keccak hash digest ROM operation called twice
    ZKR_SM_MAIN_HASHP_VALUE_MISMATCH = 59, // Main state Poseidon hash ROM operation value check failed
    ZKR_SM_MAIN_HASHP_PADDING_MISMATCH = 60, // Main state Poseidon hash ROM operation padding check failed
    ZKR_SM_MAIN_HASHP_SIZE_MISMATCH = 61, // Main state Poseidon hash ROM operation size check failed
    ZKR_SM_MAIN_HASHPLEN_LENGTH_MISMATCH = 62, // Main state Poseidon hash ROM operation data length check failed
    ZKR_SM_MAIN_HASHPLEN_CALLED_TWICE = 63, // Main state Poseidon hash ROM operation called once check failed
    ZKR_SM_MAIN_HASHPDIGEST_DIGEST_MISMATCH = 64, // Main state Poseidon hash digest ROM operation digest mismatch
    ZKR_SM_MAIN_HASHPDIGEST_CALLED_TWICE = 65, // Main state Poseidon hash digest ROM operation called twice
    ZKR_SM_MAIN_ARITH_MISMATCH = 66, // Main state arithmetic ROM operation check failed
    ZKR_SM_MAIN_ARITH_ECRECOVER_MISMATCH = 67, // Main state arithmetic ECRecover ROM operation check failed
    ZKR_SM_MAIN_BINARY_ADD_MISMATCH = 68, // Main state binary add ROM operation check failed
    ZKR_SM_MAIN_BINARY_SUB_MISMATCH = 69, // Main state binary sub ROM operation check failed
    ZKR_SM_MAIN_BINARY_LT_MISMATCH = 70, // Main state binary less than ROM operation check failed
    ZKR_SM_MAIN_BINARY_SLT_MISMATCH = 71, // Main state binary signed less than ROM operation check failed
    ZKR_SM_MAIN_BINARY_EQ_MISMATCH = 72, // Main state binary equal ROM operation check failed
    ZKR_SM_MAIN_BINARY_AND_MISMATCH = 73, // Main state binary and ROM operation check failed
    ZKR_SM_MAIN_BINARY_OR_MISMATCH = 74, // Main state binary or ROM operation check failed
    ZKR_SM_MAIN_BINARY_XOR_MISMATCH = 75, // Main state binary XOR ROM operation check failed
    ZKR_SM_MAIN_MEMALIGN_WRITE_MISMATCH = 76, // Main state memory align write ROM operation check failed
    ZKR_SM_MAIN_MEMALIGN_WRITE8_MISMATCH = 77, // Main state memory align write 8 ROM operation check failed
    ZKR_SM_MAIN_MEMALIGN_READ_MISMATCH = 78, // Main state memory align read ROM operation check failed
    ZKR_SM_MAIN_HASHK_READ_OUT_OF_RANGE = 79, // Main state Keccak hash check found read out of range
    ZKR_SM_MAIN_HASHP_READ_OUT_OF_RANGE = 80, // Main state Poseidon hash check found read out of range
    ZKR_DB_VERSION_NOT_FOUND_KVDB = 81, // Version not found in KeyValue database
    ZKR_DB_VERSION_NOT_FOUND_GLOBAL = 82, // Version not found in KeyValue database and not present in hashDB neither

    
} zkresult;

string zkresult2string (int code);

#endif