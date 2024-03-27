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
    ZKR_SM_MAIN_STORAGE_INVALID_KEY = 6, // Main state machine executor storage condition failed
    ZKR_SM_MAIN_MEMORY = 7, // Main state machine executor memory condition failed
    ZKR_SMT_INVALID_DATA_SIZE = 8, // Invalid size data for a MT node
    ZKR_AGGREGATED_PROOF_INVALID_INPUT = 9, // Aggregated proof input is incorrect
    ZKR_SM_MAIN_OOC_ARITH = 10, // Incremented arith counters exceeded the maximum
    ZKR_SM_MAIN_OOC_BINARY = 11, // Incremented binary counters exceeded the maximum
    ZKR_SM_MAIN_OOC_MEM_ALIGN = 12, // Incremented mem align counters exceeded the maximum
    ZKR_SM_MAIN_OOC_KECCAK_F = 13, // Incremented keccak-f counters exceeded the maximum
    ZKR_SM_MAIN_OOC_PADDING_PG = 14, // Incremented padding pg counters exceeded the maximum
    ZKR_SM_MAIN_OOC_SHA256_F = 15, // Incremented SHA-256-f counters exceeded the maximum
    ZKR_SM_MAIN_OOC_POSEIDON_G = 16, // Incremented poseidon g counters exceeded the maximum
    ZKR_HASHDB_GRPC_ERROR = 17, // Error making GRPC call to hash DB service
    ZKR_SM_MAIN_OUT_OF_STEPS = 18, // Main state machine executor did not complete the execution within available steps
    ZKR_SM_MAIN_INVALID_FORK_ID = 19, // Main state machine executor does not support the requested fork ID
    ZKR_SM_MAIN_INVALID_UNSIGNED_TX = 20, // Main state machine executor cannot process unsigned TXs in prover mode
    ZKR_SM_MAIN_BALANCE_MISMATCH = 21, // Main state machine executor found that total tranferred balances are not zero
    ZKR_SM_MAIN_FEA2SCALAR = 22, // Main state machine executor failed calling fea2scalar()
    ZKR_SM_MAIN_TOS32 = 23, // Main state machine executor failed calling fr.toS32()
    ZKR_SM_MAIN_S33 = 24, // Main state machine executor failed getting an S33 value from op
    ZKR_STATE_MANAGER = 25, // State root error
    ZKR_SM_MAIN_INVALID_NO_COUNTERS = 26, // No counters received outside of a process batch request
    ZKR_SM_MAIN_ARITH_ECRECOVER_DIVIDE_BY_ZERO = 27, // Main state machine arith operation during ECRecover found a divide by zero situation
    ZKR_SM_MAIN_ADDRESS_OUT_OF_RANGE = 28, // Main state machine address out of valid memory space range
    ZKR_SM_MAIN_ADDRESS_NEGATIVE = 29, // Main state machine address is negative
    ZKR_SM_MAIN_MEMALIGN_OFFSET_OUT_OF_RANGE = 30, // Main state machine mem align offset is out of range
    ZKR_SM_MAIN_MULTIPLE_FREEIN = 31, // Main state machine got multiple free inputs in one sigle ROM instruction
    ZKR_SM_MAIN_STORAGE_READ_MISMATCH = 32, // Main state machine read ROM operation check failed
    ZKR_SM_MAIN_STORAGE_WRITE_MISMATCH = 33, // Main state machine write ROM operation check failed
    ZKR_SM_MAIN_ARITH_MISMATCH = 34, // Main state machine arithmetic ROM operation check failed
    ZKR_SM_MAIN_ARITH_ECRECOVER_MISMATCH = 35, // Main state machine arithmetic ECRecover ROM operation check failed
    ZKR_SM_MAIN_BINARY_ADD_MISMATCH = 36, // Main state machine binary add ROM operation check failed
    ZKR_SM_MAIN_BINARY_SUB_MISMATCH = 37, // Main state machine binary sub ROM operation check failed
    ZKR_SM_MAIN_BINARY_LT_MISMATCH = 38, // Main state machine binary less than ROM operation check failed
    ZKR_SM_MAIN_BINARY_SLT_MISMATCH = 39, // Main state machine binary signed less than ROM operation check failed
    ZKR_SM_MAIN_BINARY_EQ_MISMATCH = 40, // Main state machine binary equal ROM operation check failed
    ZKR_SM_MAIN_BINARY_AND_MISMATCH = 41, // Main state machine binary and ROM operation check failed
    ZKR_SM_MAIN_BINARY_OR_MISMATCH = 42, // Main state machine binary or ROM operation check failed
    ZKR_SM_MAIN_BINARY_XOR_MISMATCH = 43, // Main state machine binary XOR ROM operation check failed
    ZKR_SM_MAIN_BINARY_LT4_MISMATCH = 44, // Main state machine binary less than 4 ROM operation check failed
    ZKR_SM_MAIN_MEMALIGN_WRITE_MISMATCH = 45, // Main state machine memory align write ROM operation check failed
    ZKR_SM_MAIN_MEMALIGN_WRITE8_MISMATCH = 46, // Main state machine memory align write 8 ROM operation check failed
    ZKR_SM_MAIN_MEMALIGN_READ_MISMATCH = 47, // Main state machine memory align read ROM operation check failed
    ZKR_DB_VERSION_NOT_FOUND_KVDB = 48, // Version not found in KeyValue database
    ZKR_DB_VERSION_NOT_FOUND_GLOBAL = 49, // Version not found in KeyValue database and not present in hashDB neither

    ZKR_SM_MAIN_HASHK = 50, // Main state machine executor hash Keccak condition failed
    ZKR_SM_MAIN_HASHK_SIZE_OUT_OF_RANGE = 51, // Main state machine Keccak hash size is out of range
    ZKR_SM_MAIN_HASHK_POSITION_NEGATIVE = 52, // Main state machine Keccak hash position is negative
    ZKR_SM_MAIN_HASHK_POSITION_PLUS_SIZE_OUT_OF_RANGE = 53, // Main state Keccak hash position + size is out of range
    ZKR_SM_MAIN_HASHKDIGEST_ADDRESS_NOT_FOUND = 54, // Main state machine Keccak hash digest address is not found
    ZKR_SM_MAIN_HASHKDIGEST_NOT_COMPLETED = 55, // Main state machine Keccak hash digest called when hash has not been completed
    ZKR_SM_MAIN_HASHK_VALUE_MISMATCH = 56, // Main state machine Keccak hash ROM operation value check failed
    ZKR_SM_MAIN_HASHK_PADDING_MISMATCH = 57, // Main state machine Keccak hash ROM operation padding check failed
    ZKR_SM_MAIN_HASHK_SIZE_MISMATCH = 58, // Main state machine Keccak hash ROM operation size check failed
    ZKR_SM_MAIN_HASHKLEN_LENGTH_MISMATCH = 59, // Main state machine Keccak hash ROM operation data length check failed
    ZKR_SM_MAIN_HASHKLEN_CALLED_TWICE = 60, // Main state machine Keccak hash ROM operation called once check failed
    ZKR_SM_MAIN_HASHKDIGEST_NOT_FOUND = 61, // Main state machine Keccak hash digest ROM operation slot not found
    ZKR_SM_MAIN_HASHKDIGEST_DIGEST_MISMATCH = 62, // Main state machine Keccak hash digest ROM operation digest mismatch
    ZKR_SM_MAIN_HASHKDIGEST_CALLED_TWICE = 63, // Main state machine Keccak hash digest ROM operation called twice
    ZKR_SM_MAIN_HASHK_READ_OUT_OF_RANGE = 64, // Main state machine Keccak hash check found read out of range
    
    ZKR_SM_MAIN_HASHP = 65, // Main state machine executor hash poseidon condition failed
    ZKR_SM_MAIN_HASHP_SIZE_OUT_OF_RANGE = 66, // Main state machine Poseidon hash size is out of range
    ZKR_SM_MAIN_HASHP_POSITION_NEGATIVE = 67, // Main state machine Poseidon hash position is negative
    ZKR_SM_MAIN_HASHP_POSITION_PLUS_SIZE_OUT_OF_RANGE = 68, // Main state machine Poseidon hash position + size is out of range
    ZKR_SM_MAIN_HASHPDIGEST_ADDRESS_NOT_FOUND = 69, // Main state machine Poseidon hash digest address is not found
    ZKR_SM_MAIN_HASHPDIGEST_NOT_COMPLETED = 70, // Main state machine Poseidon hash digest called when hash has not been completed
    ZKR_SM_MAIN_HASHP_VALUE_MISMATCH = 71, // Main state machine Poseidon hash ROM operation value check failed
    ZKR_SM_MAIN_HASHP_PADDING_MISMATCH = 72, // Main state machine Poseidon hash ROM operation padding check failed
    ZKR_SM_MAIN_HASHP_SIZE_MISMATCH = 73, // Main state machine Poseidon hash ROM operation size check failed
    ZKR_SM_MAIN_HASHPLEN_LENGTH_MISMATCH = 74, // Main state machine Poseidon hash ROM operation data length check failed
    ZKR_SM_MAIN_HASHPLEN_CALLED_TWICE = 75, // Main state machine Poseidon hash ROM operation called once check failed
    ZKR_SM_MAIN_HASHPDIGEST_DIGEST_MISMATCH = 76, // Main state machine Poseidon hash digest ROM operation digest mismatch
    ZKR_SM_MAIN_HASHPDIGEST_CALLED_TWICE = 77, // Main state machine Poseidon hash digest ROM operation called twice
    ZKR_SM_MAIN_HASHP_READ_OUT_OF_RANGE = 78, // Main state machine Poseidon hash check found read out of range
    
    ZKR_SM_MAIN_HASHS = 79, // Main state machine executor hash SHA-256 condition failed
    ZKR_SM_MAIN_HASHS_SIZE_OUT_OF_RANGE = 80, // Main state machine SHA-256 hash size is out of range
    ZKR_SM_MAIN_HASHS_POSITION_NEGATIVE = 81, // Main state machine SHA-256 hash position is negative
    ZKR_SM_MAIN_HASHS_POSITION_PLUS_SIZE_OUT_OF_RANGE = 82, // Main state machine SHA-256 hash position + size is out of range
    ZKR_SM_MAIN_HASHSDIGEST_ADDRESS_NOT_FOUND = 83, // Main state machine SHA-256 hash digest address is not found
    ZKR_SM_MAIN_HASHSDIGEST_NOT_COMPLETED = 84, // Main state machine SHA-256 hash digest called when hash has not been completed
    ZKR_SM_MAIN_HASHS_VALUE_MISMATCH = 85, // Main state machine SHA-256 hash ROM operation value check failed
    ZKR_SM_MAIN_HASHS_PADDING_MISMATCH = 86, // Main state machine SHA-256 hash ROM operation padding check failed
    ZKR_SM_MAIN_HASHS_SIZE_MISMATCH = 87, // Main state machine SHA-256 hash ROM operation size check failed
    ZKR_SM_MAIN_HASHSLEN_LENGTH_MISMATCH = 88, // Main state machine SHA-256 hash ROM operation data length check failed
    ZKR_SM_MAIN_HASHSLEN_CALLED_TWICE = 89, // Main state machine SHA-256 hash ROM operation called once check failed
    ZKR_SM_MAIN_HASHSDIGEST_NOT_FOUND = 90, // Main state machine SHA-256 hash digest ROM operation slot not found
    ZKR_SM_MAIN_HASHSDIGEST_DIGEST_MISMATCH = 91, // Main state machine SHA-256 hash digest ROM operation digest mismatch
    ZKR_SM_MAIN_HASHSDIGEST_CALLED_TWICE = 92, // Main state machine SHA-256 hash digest ROM operation called twice
    ZKR_SM_MAIN_HASHS_READ_OUT_OF_RANGE = 93, // Main state machine SHA-256 hash check found read out of range

    ZKR_SM_MAIN_INVALID_L1_INFO_TREE_INDEX = 94, // Main state machine ROM requests a L1InfoTree not present in input
    ZKR_SM_MAIN_INVALID_L1_INFO_TREE_SMT_PROOF_VALUE = 95, // Main state machine ROM requests a L1InfoTree SMT proof value not present in input
    ZKR_SM_MAIN_INVALID_WITNESS = 96, // Main state machine input witness is invalid or corrupt
    ZKR_CBOR_INVALID_DATA = 97, // CBOR data is invalid
    ZKR_DATA_STREAM_INVALID_DATA = 98, // Data stream data is invalid
    
    ZKR_SM_MAIN_INVALID_TX_STATUS_ERROR = 99, // Invalid TX status-error combination

} zkresult;

string zkresult2string (int code);

#endif