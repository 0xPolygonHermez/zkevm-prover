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
    ZKR_SM_MAIN_STORAGE_INVALID_KEY = 8, // Main state machine executor storage condition failed
    ZKR_SM_MAIN_MEMORY = 9, // Main state machine executor memory condition failed
    ZKR_SM_MAIN_MEMALIGN = 10, // Main state machine executor memalign condition failed
    ZKR_SM_MAIN_ADDRESS = 11, // Main state machine executor address condition failed
    ZKR_SMT_INVALID_DATA_SIZE = 12, // Invalid size data for a MT node
    ZKR_SM_MAIN_BATCH_L2_DATA_TOO_BIG = 13, // Input batch L2 data is too big
    ZKR_AGGREGATED_PROOF_INVALID_INPUT = 14, // Aggregated proof input is incorrect
    ZKR_SM_MAIN_OOC_ARITH = 15, // Incremented arith counters exceeded the maximum
    ZKR_SM_MAIN_OOC_BINARY = 16, // Incremented binary counters exceeded the maximum
    ZKR_SM_MAIN_OOC_MEM_ALIGN = 17, // Incremented mem align counters exceeded the maximum
    ZKR_SM_MAIN_OOC_KECCAK_F = 18, // Incremented keccak-f counters exceeded the maximum
    ZKR_SM_MAIN_OOC_PADDING_PG = 19, // Incremented padding pg counters exceeded the maximum
    ZKR_SM_MAIN_OOC_SHA256_F = 20, // Incremented SHA-256-f counters exceeded the maximum
    ZKR_SM_MAIN_OOC_POSEIDON_G = 21, // Incremented poseidon g counters exceeded the maximum
    ZKR_HASHDB_GRPC_ERROR = 22, // Error making GRPC call to hash DB service
    ZKR_SM_MAIN_OUT_OF_STEPS = 23, // Main state machine executor did not complete the execution within available steps
    ZKR_SM_MAIN_INVALID_FORK_ID = 24, // Main state machine executor does not support the requested fork ID
    ZKR_SM_MAIN_INVALID_UNSIGNED_TX = 25, // Main state machine executor cannot process unsigned TXs in prover mode
    ZKR_SM_MAIN_BALANCE_MISMATCH = 26, // Main state machine executor found that total tranferred balances are not zero
    ZKR_SM_MAIN_FEA2SCALAR = 27, // Main state machine executor failed calling fea2scalar()
    ZKR_SM_MAIN_TOS32 = 28, // Main state machine executor failed calling fr.toS32()
    ZKR_SM_MAIN_BATCH_INVALID_INPUT = 29, // Process batch input is incorrect
    ZKR_SM_MAIN_S33 = 30, // Main state machine executor failed getting an S33 value from op
    ZKR_STATE_MANAGER = 31, // State root error
    ZKR_SM_MAIN_INVALID_NO_COUNTERS = 32, // No counters received outside of a process batch request
    ZKR_SM_MAIN_ARITH_ECRECOVER_DIVIDE_BY_ZERO = 33, // Main state machine arith operation during ECRecover found a divide by zero situation
    ZKR_SM_MAIN_ADDRESS_OUT_OF_RANGE = 34, // Main state machine address out of valid memory space range
    ZKR_SM_MAIN_ADDRESS_NEGATIVE = 35, // Main state machine address is negative
    ZKR_SM_MAIN_MEMALIGN_OFFSET_OUT_OF_RANGE = 36, // Main state mem align offset is out of range
    ZKR_SM_MAIN_MULTIPLE_FREEIN = 37, // Main state got multiple free inputs in one sigle ROM instruction
    ZKR_SM_MAIN_STORAGE_READ_MISMATCH = 38, // Main state read ROM operation check failed
    ZKR_SM_MAIN_STORAGE_WRITE_MISMATCH = 39, // Main state write ROM operation check failed
    ZKR_SM_MAIN_ARITH_MISMATCH = 40, // Main state arithmetic ROM operation check failed
    ZKR_SM_MAIN_ARITH_ECRECOVER_MISMATCH = 41, // Main state arithmetic ECRecover ROM operation check failed
    ZKR_SM_MAIN_BINARY_ADD_MISMATCH = 42, // Main state binary add ROM operation check failed
    ZKR_SM_MAIN_BINARY_SUB_MISMATCH = 43, // Main state binary sub ROM operation check failed
    ZKR_SM_MAIN_BINARY_LT_MISMATCH = 44, // Main state binary less than ROM operation check failed
    ZKR_SM_MAIN_BINARY_SLT_MISMATCH = 45, // Main state binary signed less than ROM operation check failed
    ZKR_SM_MAIN_BINARY_EQ_MISMATCH = 46, // Main state binary equal ROM operation check failed
    ZKR_SM_MAIN_BINARY_AND_MISMATCH = 47, // Main state binary and ROM operation check failed
    ZKR_SM_MAIN_BINARY_OR_MISMATCH = 48, // Main state binary or ROM operation check failed
    ZKR_SM_MAIN_BINARY_XOR_MISMATCH = 49, // Main state binary XOR ROM operation check failed
    ZKR_SM_MAIN_BINARY_LT4_MISMATCH = 50, // Main state binary less than 4 ROM operation check failed
    ZKR_SM_MAIN_MEMALIGN_WRITE_MISMATCH = 51, // Main state memory align write ROM operation check failed
    ZKR_SM_MAIN_MEMALIGN_WRITE8_MISMATCH = 52, // Main state memory align write 8 ROM operation check failed
    ZKR_SM_MAIN_MEMALIGN_READ_MISMATCH = 53, // Main state memory align read ROM operation check failed
    ZKR_DB_VERSION_NOT_FOUND_KVDB = 54, // Version not found in KeyValue database
    ZKR_DB_VERSION_NOT_FOUND_GLOBAL = 55, // Version not found in KeyValue database and not present in hashDB neither

    ZKR_SM_MAIN_HASHK = 56, // Main state machine executor hash Keccak condition failed
    ZKR_SM_MAIN_HASHK_SIZE_OUT_OF_RANGE = 57, // Main state Keccak hash size is out of range
    ZKR_SM_MAIN_HASHK_POSITION_NEGATIVE = 58, // Main state Keccak hash position is negative
    ZKR_SM_MAIN_HASHK_POSITION_PLUS_SIZE_OUT_OF_RANGE = 59, // Main state Keccak hash position + size is out of range
    ZKR_SM_MAIN_HASHKDIGEST_ADDRESS_NOT_FOUND = 60, // Main state Keccak hash digest address is not found
    ZKR_SM_MAIN_HASHKDIGEST_NOT_COMPLETED = 61, // Main state Keccak hash digest called when hash has not been completed
    ZKR_SM_MAIN_HASHK_VALUE_MISMATCH = 62, // Main state Keccak hash ROM operation value check failed
    ZKR_SM_MAIN_HASHK_PADDING_MISMATCH = 63, // Main state Keccak hash ROM operation padding check failed
    ZKR_SM_MAIN_HASHK_SIZE_MISMATCH = 64, // Main state Keccak hash ROM operation size check failed
    ZKR_SM_MAIN_HASHKLEN_LENGTH_MISMATCH = 65, // Main state Keccak hash ROM operation data length check failed
    ZKR_SM_MAIN_HASHKLEN_CALLED_TWICE = 66, // Main state Keccak hash ROM operation called once check failed
    ZKR_SM_MAIN_HASHKDIGEST_NOT_FOUND = 67, // Main state Keccak hash digest ROM operation slot not found
    ZKR_SM_MAIN_HASHKDIGEST_DIGEST_MISMATCH = 68, // Main state Keccak hash digest ROM operation digest mismatch
    ZKR_SM_MAIN_HASHKDIGEST_CALLED_TWICE = 69, // Main state Keccak hash digest ROM operation called twice
    ZKR_SM_MAIN_HASHK_READ_OUT_OF_RANGE = 70, // Main state Keccak hash check found read out of range
    
    ZKR_SM_MAIN_HASHP = 71, // Main state machine executor hash poseidon condition failed
    ZKR_SM_MAIN_HASHP_SIZE_OUT_OF_RANGE = 72, // Main state Poseidon hash size is out of range
    ZKR_SM_MAIN_HASHP_POSITION_NEGATIVE = 73, // Main state Poseidon hash position is negative
    ZKR_SM_MAIN_HASHP_POSITION_PLUS_SIZE_OUT_OF_RANGE = 74, // Main state Poseidon hash position + size is out of range
    ZKR_SM_MAIN_HASHPDIGEST_ADDRESS_NOT_FOUND = 75, // Main state Poseidon hash digest address is not found
    ZKR_SM_MAIN_HASHPDIGEST_NOT_COMPLETED = 76, // Main state Poseidon hash digest called when hash has not been completed
    ZKR_SM_MAIN_HASHP_VALUE_MISMATCH = 77, // Main state Poseidon hash ROM operation value check failed
    ZKR_SM_MAIN_HASHP_PADDING_MISMATCH = 78, // Main state Poseidon hash ROM operation padding check failed
    ZKR_SM_MAIN_HASHP_SIZE_MISMATCH = 79, // Main state Poseidon hash ROM operation size check failed
    ZKR_SM_MAIN_HASHPLEN_LENGTH_MISMATCH = 80, // Main state Poseidon hash ROM operation data length check failed
    ZKR_SM_MAIN_HASHPLEN_CALLED_TWICE = 81, // Main state Poseidon hash ROM operation called once check failed
    ZKR_SM_MAIN_HASHPDIGEST_DIGEST_MISMATCH = 82, // Main state Poseidon hash digest ROM operation digest mismatch
    ZKR_SM_MAIN_HASHPDIGEST_CALLED_TWICE = 83, // Main state Poseidon hash digest ROM operation called twice
    ZKR_SM_MAIN_HASHP_READ_OUT_OF_RANGE = 84, // Main state Poseidon hash check found read out of range
    
    ZKR_SM_MAIN_HASHS = 85, // Main state machine executor hash SHA-256 condition failed
    ZKR_SM_MAIN_HASHS_SIZE_OUT_OF_RANGE = 86, // Main state SHA-256 hash size is out of range
    ZKR_SM_MAIN_HASHS_POSITION_NEGATIVE = 87, // Main state SHA-256 hash position is negative
    ZKR_SM_MAIN_HASHS_POSITION_PLUS_SIZE_OUT_OF_RANGE = 88, // Main state SHA-256 hash position + size is out of range
    ZKR_SM_MAIN_HASHSDIGEST_ADDRESS_NOT_FOUND = 89, // Main state SHA-256 hash digest address is not found
    ZKR_SM_MAIN_HASHSDIGEST_NOT_COMPLETED = 90, // Main state SHA-256 hash digest called when hash has not been completed
    ZKR_SM_MAIN_HASHS_VALUE_MISMATCH = 91, // Main state SHA-256 hash ROM operation value check failed
    ZKR_SM_MAIN_HASHS_PADDING_MISMATCH = 92, // Main state SHA-256 hash ROM operation padding check failed
    ZKR_SM_MAIN_HASHS_SIZE_MISMATCH = 93, // Main state SHA-256 hash ROM operation size check failed
    ZKR_SM_MAIN_HASHSLEN_LENGTH_MISMATCH = 94, // Main state SHA-256 hash ROM operation data length check failed
    ZKR_SM_MAIN_HASHSLEN_CALLED_TWICE = 95, // Main state SHA-256 hash ROM operation called once check failed
    ZKR_SM_MAIN_HASHSDIGEST_NOT_FOUND = 96, // Main state SHA-256 hash digest ROM operation slot not found
    ZKR_SM_MAIN_HASHSDIGEST_DIGEST_MISMATCH = 97, // Main state SHA-256 hash digest ROM operation digest mismatch
    ZKR_SM_MAIN_HASHSDIGEST_CALLED_TWICE = 98, // Main state SHA-256 hash digest ROM operation called twice
    ZKR_SM_MAIN_HASHS_READ_OUT_OF_RANGE = 99, // Main state SHA-256 hash check found read out of range

} zkresult;

string zkresult2string (int code);

#endif